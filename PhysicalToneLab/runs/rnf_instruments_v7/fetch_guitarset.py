"""Fetch only what is needed from GuitarSet (Zenodo 3371780, CC BY 4.0) over a slow link:
the annotation zip (resumed, parallel ranges) and N solo microphone recordings read straight out of the remote zip.
Writes only to data/guitarset/.   python fetch_guitarset.py [how many solo files]
"""
import sys, os, io, struct, zlib, zipfile, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

OUT = Path(__file__).resolve().parent / 'data/guitarset'
URL = 'https://zenodo.org/records/3371780/files/{}?download=1'


def fetch(name, a, b, tries=40):
    for k in range(tries):
        try:
            req = urllib.request.Request(URL.format(name), headers={'User-Agent': 'python-urllib', 'Range': f'bytes={a}-{b}'})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            if len(data) == b - a + 1: return data
        except Exception: time.sleep(2)
    raise RuntimeError(f'could not fetch {name} {a}-{b}')


def size(name):
    with urllib.request.urlopen(urllib.request.Request(URL.format(name), headers={'User-Agent': 'python-urllib'}, method='HEAD'), timeout=60) as r: return int(r.headers['Content-Length'])


class Remote(io.RawIOBase):
    """Read-only file over HTTP ranges, enough for zipfile to read the directory at the end of the archive."""
    def __init__(self, name): self.name_, self.n, self.pos, self.tail = name, size(name), 0, None
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0): self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.n + off; return self.pos
    def read(self, k=-1):
        k = self.n - self.pos if k < 0 else min(k, self.n - self.pos)
        if k <= 0: return b''
        if self.tail is None: start = max(0, self.n - 400000); self.tail = (start, fetch(self.name_, start, self.n - 1))
        s, buf = self.tail
        data = buf[self.pos - s:self.pos - s + k] if self.pos >= s else fetch(self.name_, self.pos, self.pos + k - 1)
        self.pos += len(data); return data


def main(count=48):
    OUT.mkdir(parents=True, exist_ok=True); ann = OUT / 'annotation.zip'; total = size('annotation.zip'); have = ann.stat().st_size if ann.exists() else 0
    if have < total:
        step = 1 << 20; parts = [(a, min(a + step, total) - 1) for a in range(have, total, step)]
        with ThreadPoolExecutor(8) as ex, open(ann, 'ab') as f:
            for chunk in ex.map(lambda p: fetch('annotation.zip', *p), parts): f.write(chunk)
    print('annotation.zip', ann.stat().st_size, 'of', total, flush=True)
    z = zipfile.ZipFile(Remote('audio_mono-mic.zip')); solos = sorted((i for i in z.infolist() if i.filename.endswith('_solo_mic.wav')), key=lambda i: i.filename); pick = solos[::max(1, len(solos) // count)][:count]
    print('solo recordings in archive', len(solos), 'fetching', len(pick), 'about', round(sum(i.compress_size for i in pick) / 1e6), 'MB', flush=True); (OUT / 'wav').mkdir(exist_ok=True)

    def one(i):
        dst = OUT / 'wav' / i.filename.split('/')[-1]
        if dst.exists() and dst.stat().st_size == i.file_size: return dst.name
        head = fetch('audio_mono-mic.zip', i.header_offset, i.header_offset + 29); nlen, elen = struct.unpack('<HH', head[26:30]); a = i.header_offset + 30 + nlen + elen
        raw = fetch('audio_mono-mic.zip', a, a + i.compress_size - 1); data = raw if i.compress_type == 0 else zlib.decompress(raw, -15)
        assert len(data) == i.file_size and zlib.crc32(data) == i.CRC; dst.write_bytes(data); return dst.name
    with ThreadPoolExecutor(8) as ex:
        for k, name in enumerate(ex.map(one, pick)): print(k + 1, name, flush=True)
    print('FETCH_DONE')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 48)

"""Fetch only the fingerstyle single notes (PS/FS) of IDMT-SMT-Bass (Zenodo 7188892, CC BY-NC-ND 4.0: research use,
not redistributed) straight out of the remote 1.6 GB zip.  Writes only to data/bass/."""
import sys, struct, zlib, zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_guitarset as fg
fg.URL = 'https://zenodo.org/records/7188892/files/{}?download=1'
OUT = Path(__file__).resolve().parent / 'data/bass'; OUT.mkdir(parents=True, exist_ok=True); Z = 'IDMT-SMT-BASS.zip'
z = zipfile.ZipFile(fg.Remote(Z)); pick = [i for i in z.infolist() if '/FS/' in '/' + i.filename and i.filename.lower().endswith('.wav') and '/PS/' in '/' + i.filename]
print('fetching', len(pick), 'files, about', round(sum(i.compress_size for i in pick) / 1e6), 'MB', flush=True)
def one(i):
    dst = OUT / i.filename.split('/')[-1]
    if dst.exists() and dst.stat().st_size == i.file_size: return 1
    head = fg.fetch(Z, i.header_offset, i.header_offset + 29); nlen, elen = struct.unpack('<HH', head[26:30]); a = i.header_offset + 30 + nlen + elen
    raw = fg.fetch(Z, a, a + i.compress_size - 1); data = raw if i.compress_type == 0 else zlib.decompress(raw, -15); assert len(data) == i.file_size and zlib.crc32(data) == i.CRC; dst.write_bytes(data); return 1
with ThreadPoolExecutor(12) as ex:
    for k, _ in enumerate(ex.map(one, pick)):
        if k % 60 == 0: print(k, flush=True)
print('FETCH_DONE', len(list(OUT.glob('*.wav'))))

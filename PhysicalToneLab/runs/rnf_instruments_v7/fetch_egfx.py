"""Fetch one pickup position of EGFxSet (Zenodo 7044411, CC BY 4.0): clean notes and the same notes through three
real distortion pedals, read straight out of the remote zips.  Writes only to data/egfx/.   python fetch_egfx.py [pickup]
"""
import sys, struct, zlib, zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_guitarset as fg
fg.URL = 'https://zenodo.org/records/7044411/files/{}?download=1'
OUT = Path(__file__).resolve().parent / 'data/egfx'; PICKUP = sys.argv[1] if len(sys.argv) > 1 else 'Bridge'
for tone in ('Clean', 'RAT', 'TubeScreamer', 'BluesDriver'):
    Z = tone + '.zip'; pick = [i for i in zipfile.ZipFile(fg.Remote(Z)).infolist() if i.filename.lower().endswith('.wav') and i.filename.split('/')[-2] == PICKUP]
    dst_dir = OUT / tone; dst_dir.mkdir(parents=True, exist_ok=True); print(tone, len(pick), 'files', round(sum(i.compress_size for i in pick) / 1e6), 'MB', flush=True)

    def one(i):
        dst = dst_dir / i.filename.split('/')[-1]
        if dst.exists() and dst.stat().st_size == i.file_size: return 1
        head = fg.fetch(Z, i.header_offset, i.header_offset + 29); nlen, elen = struct.unpack('<HH', head[26:30]); a = i.header_offset + 30 + nlen + elen
        raw = fg.fetch(Z, a, a + i.compress_size - 1); data = raw if i.compress_type == 0 else zlib.decompress(raw, -15); assert len(data) == i.file_size and zlib.crc32(data) == i.CRC; dst.write_bytes(data); return 1
    with ThreadPoolExecutor(12) as ex: list(ex.map(one, pick))
    print(tone, 'done', len(list(dst_dir.glob('*.wav'))), flush=True)
print('FETCH_DONE')

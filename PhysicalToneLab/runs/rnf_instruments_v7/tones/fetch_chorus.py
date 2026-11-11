"""Fetch the bridge-pickup Chorus recordings of EGFxSet (Zenodo 7044411, CC BY 4.0) into data/egfx/Chorus."""
import sys, struct, zlib, zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
V7 = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(V7))
import fetch_guitarset as fg
fg.URL = 'https://zenodo.org/records/7044411/files/{}?download=1'; OUT = V7 / 'data/egfx'
for tone in sys.argv[1:] or ['Chorus']:
    Z = tone + '.zip'; pick = [i for i in zipfile.ZipFile(fg.Remote(Z)).infolist() if i.filename.lower().endswith('.wav') and i.filename.split('/')[-2] == 'Bridge']
    d = OUT / tone; d.mkdir(parents=True, exist_ok=True); print(tone, len(pick), 'files', round(sum(i.compress_size for i in pick) / 1e6), 'MB', flush=True)
    def one(i):
        dst = d / i.filename.split('/')[-1]
        if dst.exists() and dst.stat().st_size == i.file_size: return 1
        head = fg.fetch(Z, i.header_offset, i.header_offset + 29); nlen, elen = struct.unpack('<HH', head[26:30]); a = i.header_offset + 30 + nlen + elen
        raw = fg.fetch(Z, a, a + i.compress_size - 1); data = raw if i.compress_type == 0 else zlib.decompress(raw, -15); assert len(data) == i.file_size and zlib.crc32(data) == i.CRC; dst.write_bytes(data); return 1
    with ThreadPoolExecutor(12) as ex: list(ex.map(one, pick))
    print(tone, 'done', len(list(d.glob('*.wav'))), flush=True)
print('FETCH_DONE')

"""Fetch real crash cymbal hits: University of Iowa Musical Instrument Samples (free, unrestricted use; anechoic),
suspended crash cymbals 13/17/18/20 inch, stick, 'normal' and 'shoulder' strikes, pp/mf/ff.  Writes only data/crash/."""
import urllib.request, urllib.parse, re
from pathlib import Path
OUT = Path(__file__).resolve().parent / 'data/crash'; OUT.mkdir(parents=True, exist_ok=True); base = 'https://theremin.music.uiowa.edu/'
get = lambda u: urllib.request.urlopen(urllib.request.Request(u, headers={'User-Agent': 'python-urllib'}), timeout=120).read()
links = [l for l in re.findall(r'href="([^"]+)"', get(base + 'MIScymbals.html').decode('latin-1')) if re.search(r'/\d\dcrash\.stick\.(normal|shoulder)\.(pp|mf|ff)\.aif$', l)]
print('files listed', len(links))
for l in links:
    dst = OUT / l.split('/')[-1]
    if not dst.exists(): dst.write_bytes(get(base + urllib.parse.quote(l)))
    print(dst.name, dst.stat().st_size // 1000, 'kB', flush=True)

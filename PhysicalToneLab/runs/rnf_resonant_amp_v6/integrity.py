"""Hash every pre-existing file before training; verify afterwards. Read-only."""
from pathlib import Path
import hashlib, json, sys
HERE = Path(__file__).resolve().parent
LAB = HERE.parents[1]; BASE = LAB.parent
SNAP = HERE / 'integrity_before.json'

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()

def collect():
    files = {}
    for root in (LAB, BASE / 'NeuralInstrumentStudio_Experiments' / 'energy_v3'):
        for p in root.rglob('*'):
            if p.is_file() and HERE not in p.parents: files[str(p)] = sha(p)
    old = json.loads((LAB / 'runs/guitar_real_v1/protected_files.json').read_text(encoding='utf-8'))
    listed = {'checked': 0, 'missing': [], 'mismatch_vs_original_list': []}
    for name, row in old.items():
        p = Path(name)
        if not p.is_file(): listed['missing'].append(name); continue
        digest = sha(p); files[str(p)] = digest; listed['checked'] += 1
        if digest != row['sha256']: listed['mismatch_vs_original_list'].append(name)
    return files, listed

if __name__ == '__main__':
    files, listed = collect()
    if sys.argv[1:] == ['snapshot']:
        if SNAP.exists(): raise SystemExit('snapshot already exists; not overwriting')
        SNAP.write_text(json.dumps({'files': files, 'original_list': listed}, indent=1), encoding='utf-8')
        print(json.dumps({'snapshot_files': len(files), 'original_list_checked': listed['checked'],
                          'missing': len(listed['missing']), 'mismatch': len(listed['mismatch_vs_original_list'])}))
    else:
        before = json.loads(SNAP.read_text(encoding='utf-8'))['files']
        changed = [k for k, v in before.items() if files.get(k) != v]
        added = [k for k in files if k not in before]
        out = {'files_before': len(before), 'changed_or_deleted': changed, 'new_files_outside_run_folder': added,
               'original_list_mismatch': listed['mismatch_vs_original_list'], 'intact': not changed}
        (HERE / 'integrity_after.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
        print(json.dumps({k: (v if not isinstance(v, list) else v[:10]) for k, v in out.items()}, ensure_ascii=True))

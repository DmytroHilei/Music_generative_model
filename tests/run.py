"""Run every CPU test without pytest: python tests/run.py [name filter]  (pytest tests/ works too)."""

import importlib.util
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

pattern = sys.argv[1] if len(sys.argv) > 1 else ''
failed, n = [], 0
for path in sorted(HERE.glob('*/test_*.py')):
    spec = importlib.util.spec_from_file_location(f'{path.parent.name}_{path.stem}', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name, fn in vars(module).items():
        if not name.startswith('test_') or not callable(fn) or pattern not in f'{path.relative_to(HERE)}::{name}':
            continue
        n += 1
        t0 = time.time()
        try:
            fn()
            print(f'ok    {path.relative_to(HERE)}::{name} ({time.time() - t0:.1f}s)')
        except Exception:
            failed.append(f'{path.relative_to(HERE)}::{name}')
            print(f'FAIL  {path.relative_to(HERE)}::{name}')
            traceback.print_exc()
print(f'\n{n - len(failed)}/{n} passed' + (''.join(f'\n  failed: {f}' for f in failed)))
sys.exit(1 if failed else 0)

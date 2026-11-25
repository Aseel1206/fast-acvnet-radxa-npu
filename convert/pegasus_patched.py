import sys

# pegasus.py imports modules that live next to it (e.g. `custom`), so its directory must be on the path
# when this wrapper is run from somewhere else (such as this repository).
ACUITY_BIN = '/root/acuity-toolkit-whl-6.30.22/bin'
sys.path.insert(0, ACUITY_BIN)

# Inject fake torch module to prevent gridsample.so from crashing
# Export path doesn't use grid_sample so this is safe
import types
fake_torch = types.ModuleType('torch')
fake_torch.__version__ = '1.5.1'
fake_torch._C = types.ModuleType('torch._C')
sys.modules['torch'] = fake_torch
sys.modules['torch._C'] = fake_torch._C

# Now run pegasus normally
import runpy
runpy.run_path(ACUITY_BIN + '/pegasus.py', run_name='__main__')

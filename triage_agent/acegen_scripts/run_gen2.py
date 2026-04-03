"""
run_gen2.py — Generation 2 launcher for TRI_FLAG + ACEGEN

Changes from run_gen1.py:
  - BATCH_ID          = "gen_002"
  - GENERATION_NUMBER = "2"
  - SureChEMBL fix deployed (new async POST /search/structure API)

Run from triage_agent/ directory:
    python acegen_scripts/run_gen2.py
"""
import os
import sys
import ctypes

# Must be set before ANY torch import
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ---------------------------------------------------------------------------
# Path resolution
# _HERE  = triage_agent/acegen_scripts/   (this file's directory)
# _ROOT  = triage_agent/                  (project root for all imports)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)   # one level up — triage_agent/

# ---------------------------------------------------------------------------
# Windows DLL fix — patch torch's _load_dll_libraries before it runs.
# ---------------------------------------------------------------------------
_SITE_PACKAGES = os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages")
_TORCH_LIB     = os.path.join(_SITE_PACKAGES, "torch", "lib")
_CONDA_LIB_BIN = os.path.join(os.path.dirname(sys.executable), "Library", "bin")
_CONDA_ROOT    = os.path.dirname(sys.executable)

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

for _dir in [_TORCH_LIB, _CONDA_LIB_BIN, _CONDA_ROOT,
             os.path.join(_CONDA_ROOT, "Library", "mingw-w64", "bin"),
             os.path.join(_CONDA_ROOT, "Scripts")]:
    if os.path.isdir(_dir):
        kernel32.AddDllDirectory(ctypes.c_wchar_p(_dir))
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(_dir)

import glob as _glob
for _dll in _glob.glob(os.path.join(_TORCH_LIB, "*.dll")):
    try:
        ctypes.WinDLL(_dll)
    except OSError:
        pass

# ---------------------------------------------------------------------------
# sys.path — both _ROOT and its parent for dotted-path import resolution
# ---------------------------------------------------------------------------
_PARENT = os.path.dirname(_ROOT)   # parent of triage_agent/
for _p in [_ROOT, _PARENT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Scorer config via environment variables
# ---------------------------------------------------------------------------
os.environ["TRIFLAG_BATCH_ID"]              = "gen_002"
os.environ["TRIFLAG_GENERATION_NUMBER"]     = "2"
os.environ["TRIFLAG_SKIP_SIMILARITY"]       = "0"   # live API calls enabled
os.environ["TRIFLAG_ENABLE_DEEPPURPOSE"]    = "1"   # S_act active (BACE1 pIC50)
os.environ["TRIFLAG_DB_PATH"]               = os.path.join(_ROOT, "runs", "triflag.db")

print(f"[run_gen2] Scorer configured:")
print(f"  BATCH_ID           = {os.environ['TRIFLAG_BATCH_ID']}")
print(f"  GENERATION_NUMBER  = {os.environ['TRIFLAG_GENERATION_NUMBER']}")
print(f"  SKIP_SIMILARITY    = {os.environ['TRIFLAG_SKIP_SIMILARITY']} (0=False, live API)")
print(f"  ENABLE_DEEPPURPOSE = {os.environ['TRIFLAG_ENABLE_DEEPPURPOSE']} (1=True, S_act active)")
print(f"  DB_PATH            = {os.environ['TRIFLAG_DB_PATH']}")

# ---------------------------------------------------------------------------
# Locate and launch ACEGEN reinvent script
# ---------------------------------------------------------------------------
_ACEGEN_SCRIPT = os.path.normpath(os.path.join(
    _ROOT,       # triage_agent/
    "..",        # Lab Code/TRI_FLAG/
    "..",        # Lab Code/
    "acegen-open",
    "scripts", "reinvent", "reinvent.py"
))

if not os.path.exists(_ACEGEN_SCRIPT):
    print(f"\n[run_gen2] ERROR: ACEGEN script not found at:\n  {_ACEGEN_SCRIPT}")
    print("Edit _ACEGEN_SCRIPT path in run_gen2.py to point at your acegen-open clone.")
    sys.exit(1)

print(f"\n[run_gen2] Launching ACEGEN reinvent script:")
print(f"  {_ACEGEN_SCRIPT}")
print(f"  --config-path loop --config-name config_triflag\n")

import torch    # noqa: F401 — intentional pre-load (populates DLL cache)
import torchrl  # noqa: F401

sys.argv = [
    _ACEGEN_SCRIPT,
    "--config-path", os.path.join(_ROOT, "loop"),
    "--config-name", "config_triflag",
]

with open(_ACEGEN_SCRIPT) as f:
    _script_src = f.read()

_script_src = _script_src.replace(
    'os.chdir("/tmp")',
    'os.chdir(os.environ.get("TEMP", os.environ.get("TMP", os.path.expanduser("~"))))'
)

exec(compile(_script_src, _ACEGEN_SCRIPT, "exec"), {"__name__": "__main__", "__file__": _ACEGEN_SCRIPT})
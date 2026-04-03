"""
run_generation.py — Universal generation launcher for TRI_FLAG + ACEGEN

Replaces the per-generation run_gen0.py / run_gen1.py / run_gen2.py files.
The only thing that changes between generations is BATCH_ID and
GENERATION_NUMBER — this script handles both automatically.

Usage:
    # Auto-increment (reads max generation from DB and adds 1) — default:
    python acegen_scripts/run_generation.py

    # Explicit generation number:
    python acegen_scripts/run_generation.py --gen 3

Run from triage_agent/ directory.
"""
import os
import sys
import ctypes
import argparse

# Must be set before ANY torch import
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
_HERE   = os.path.dirname(os.path.abspath(__file__))
_ROOT   = os.path.dirname(_HERE)       # triage_agent/
_PARENT = os.path.dirname(_ROOT)       # parent of triage_agent/

for _p in [_ROOT, _PARENT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Windows DLL fix
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
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Launch a TRI_FLAG + ACEGEN generation run.",
    epilog="Omit --gen to auto-increment based on what's already in the database.",
)
parser.add_argument(
    "--gen", type=int, default=None,
    help="Generation number to run (e.g. 3). Omit to auto-increment.",
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Resolve generation number from DB if not provided
# ---------------------------------------------------------------------------
_DB_PATH = os.path.join(_ROOT, "runs", "triflag.db")

def _get_next_generation() -> int:
    """Return max(generation_number) + 1 from triage_runs, or 0 if DB is empty."""
    try:
        import sqlite3
        if not os.path.exists(_DB_PATH):
            return 0
        conn = sqlite3.connect(_DB_PATH)
        row = conn.execute(
            "SELECT MAX(generation_number) FROM triage_runs "
            "WHERE generation_number IS NOT NULL"
        ).fetchone()
        conn.close()
        current_max = row[0] if (row and row[0] is not None) else -1
        return current_max + 1
    except Exception as exc:
        print(f"[run_generation] Warning: could not read DB ({exc}), defaulting to gen 0")
        return 0

if args.gen is not None:
    generation_number = args.gen
else:
    generation_number = _get_next_generation()

batch_id = f"gen_{generation_number:03d}"

# ---------------------------------------------------------------------------
# Pre-flight EBI health check
# ---------------------------------------------------------------------------
def _ebi_is_up() -> bool:
    try:
        import requests
        r = requests.get(
            "https://www.ebi.ac.uk/chembl/api/data/spore",
            timeout=10,
        )
        return r.status_code == 200
    except Exception:
        return False

print(f"\n[run_generation] Checking EBI / ChEMBL availability...")
if _ebi_is_up():
    print(f"[run_generation] EBI is UP — live similarity screening enabled")
    skip_similarity = "0"
else:
    print(f"[run_generation] WARNING: EBI is DOWN")
    print(f"[run_generation] Setting SKIP_SIMILARITY=1 to avoid timeout hangs")
    print(f"[run_generation] Re-run once EBI recovers for full IP screening")
    skip_similarity = "1"

# ---------------------------------------------------------------------------
# Scorer config via environment variables
# ---------------------------------------------------------------------------
os.environ["TRIFLAG_BATCH_ID"]           = batch_id
os.environ["TRIFLAG_GENERATION_NUMBER"]  = str(generation_number)
os.environ["TRIFLAG_SKIP_SIMILARITY"]    = skip_similarity
os.environ["TRIFLAG_ENABLE_DEEPPURPOSE"] = "1"
os.environ["TRIFLAG_DB_PATH"]            = _DB_PATH

print(f"\n[run_generation] Configuration:")
print(f"  BATCH_ID           = {batch_id}")
print(f"  GENERATION_NUMBER  = {generation_number}")
print(f"  SKIP_SIMILARITY    = {skip_similarity}  (0=live API, 1=offline)")
print(f"  ENABLE_DEEPPURPOSE = 1  (S_act active, BACE1 pIC50)")
print(f"  DB_PATH            = {_DB_PATH}")

# ---------------------------------------------------------------------------
# Locate and launch ACEGEN reinvent script
# ---------------------------------------------------------------------------
_ACEGEN_SCRIPT = os.path.normpath(os.path.join(
    _ROOT, "..", "..", "acegen-open", "scripts", "reinvent", "reinvent.py"
))

if not os.path.exists(_ACEGEN_SCRIPT):
    print(f"\n[run_generation] ERROR: ACEGEN script not found at:\n  {_ACEGEN_SCRIPT}")
    sys.exit(1)

print(f"\n[run_generation] Launching ACEGEN:")
print(f"  Script: {_ACEGEN_SCRIPT}")
print(f"  Config: loop/config_triflag\n")

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
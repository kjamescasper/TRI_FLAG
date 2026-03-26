"""
run_gen0.py — Generation 0 launcher for TRI_FLAG + ACEGEN

Sets required environment variables BEFORE torch is imported anywhere,
then hands off to the ACEGEN reinvent script.

Run from triage_agent/ directory:
    python run_gen0.py
"""
import os
import sys
import ctypes

# Must be set before ANY torch import
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ---------------------------------------------------------------------------
# Windows DLL fix — patch torch's _load_dll_libraries before it runs.
#
# torch uses LoadLibraryExW with flag 0x00001100 (LOAD_LIBRARY_SEARCH_DEFAULT_DIRS)
# which ignores PATH and only looks in directories registered via the Win32
# AddDllDirectory() API. Python's os.add_dll_directory() wraps this correctly,
# but we must call it before torch.__init__ runs _load_dll_libraries().
#
# We do this by finding the torch __init__.py, injecting our add_dll_directory
# calls via a sitecustomize-style hook, then importing torch normally.
# ---------------------------------------------------------------------------

# Step 1: Find torch lib dir without importing torch
_SITE_PACKAGES = os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages")
_TORCH_LIB = os.path.join(_SITE_PACKAGES, "torch", "lib")
_CONDA_LIB_BIN = os.path.join(os.path.dirname(sys.executable), "Library", "bin")
_CONDA_ROOT = os.path.dirname(sys.executable)

# Step 2: Register directories with Win32 AddDllDirectory via ctypes directly
# This is the same underlying call as os.add_dll_directory() but we also
# need to pre-load the DLLs torch needs using LoadLibraryW (flag 0, no search restriction)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

# Register all relevant dirs with AddDllDirectory (the Win32 API torch uses)
for _dir in [_TORCH_LIB, _CONDA_LIB_BIN, _CONDA_ROOT,
             os.path.join(_CONDA_ROOT, "Library", "mingw-w64", "bin"),
             os.path.join(_CONDA_ROOT, "Scripts")]:
    if os.path.isdir(_dir):
        kernel32.AddDllDirectory(ctypes.c_wchar_p(_dir))
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(_dir)

# Step 3: Pre-load shm.dll and its siblings using LoadLibraryW (unrestricted search)
# so they're already in the process DLL cache when torch tries LoadLibraryExW
import glob as _glob
for _dll in _glob.glob(os.path.join(_TORCH_LIB, "*.dll")):
    try:
        ctypes.WinDLL(_dll)
    except OSError:
        pass  # Some DLLs may fail — that's OK, we just want the critical ones cached

# Ensure triage_agent/ is on PYTHONPATH so TRI_FLAG imports resolve.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)   # parent of triage_agent/

# triage_agent.loop.triflag_scorer requires TWO things on sys.path:
# 1. _HERE (triage_agent/) so that internal imports like `from tools.x import X` work
# 2. _PARENT so that `import triage_agent.loop.triflag_scorer` resolves as a package
for _p in [_HERE, _PARENT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Configure the scorer via environment variables — these are read by
# triflag_scorer.py at module import time, so they work even when ACEGEN
# imports a fresh copy of the module rather than using our pre-imported one.
# ---------------------------------------------------------------------------
os.environ["TRIFLAG_BATCH_ID"] = "gen_000"
os.environ["TRIFLAG_GENERATION_NUMBER"] = "0"
os.environ["TRIFLAG_SKIP_SIMILARITY"] = "1"   # CRITICAL: no live API calls for gen 0
os.environ["TRIFLAG_DB_PATH"] = os.path.join(_HERE, "runs", "triflag.db")

print(f"[run_gen0] Scorer configured:")
print(f"  BATCH_ID          = {os.environ['TRIFLAG_BATCH_ID']}")
print(f"  GENERATION_NUMBER = {os.environ['TRIFLAG_GENERATION_NUMBER']}")
print(f"  SKIP_SIMILARITY   = {os.environ['TRIFLAG_SKIP_SIMILARITY']} (1=True)")
print(f"  DB_PATH           = {os.environ['TRIFLAG_DB_PATH']}")

# ---------------------------------------------------------------------------
# Hand off to ACEGEN reinvent script
# ---------------------------------------------------------------------------
# Locate the reinvent script relative to the cloned acegen-open repo.
# Adjust this path if you cloned acegen-open somewhere else.
_ACEGEN_SCRIPT = os.path.join(
    os.path.dirname(_HERE),          # Lab Code/TRI_FLAG/
    "..",                            # Lab Code/
    "acegen-open",                   # Lab Code/acegen-open/
    "scripts", "reinvent", "reinvent.py"
)
_ACEGEN_SCRIPT = os.path.normpath(_ACEGEN_SCRIPT)

if not os.path.exists(_ACEGEN_SCRIPT):
    print(f"\n[run_gen0] ERROR: ACEGEN script not found at:\n  {_ACEGEN_SCRIPT}")
    print("Edit _ACEGEN_SCRIPT path in run_gen0.py to point at your acegen-open clone.")
    sys.exit(1)

print(f"\n[run_gen0] Launching ACEGEN reinvent script:")
print(f"  {_ACEGEN_SCRIPT}")
print(f"  --config-path loop --config-name config_triflag\n")

# Pre-import torch so that when reinvent.py does `import torch` it hits
# sys.modules cache and skips _load_dll_libraries() entirely.
# This is necessary because exec() runs in the same process but the DLL
# search path additions above don't propagate to the loader in that context.
import torch  # noqa: F401 — intentional pre-load
import torchrl  # noqa: F401 — pre-load torchrl too

# Inject the config args as if they were sys.argv, then exec the script
# in the current process — this avoids spawning a subprocess where env
# vars might not propagate correctly on Windows.
sys.argv = [
    _ACEGEN_SCRIPT,
    "--config-path", os.path.join(_HERE, "loop"),
    "--config-name", "config_triflag",
]

# Patch the reinvent script source before exec-ing it.
# Line: os.chdir("/tmp")  — hardcoded Linux path, breaks on Windows.
# Hydra writes its output logs there; redirect to Windows temp instead.
with open(_ACEGEN_SCRIPT) as f:
    _script_src = f.read()

_script_src = _script_src.replace(
    'os.chdir("/tmp")',
    'os.chdir(os.environ.get("TEMP", os.environ.get("TMP", os.path.expanduser("~"))))'
)

exec(compile(_script_src, _ACEGEN_SCRIPT, "exec"), {"__name__": "__main__", "__file__": _ACEGEN_SCRIPT})
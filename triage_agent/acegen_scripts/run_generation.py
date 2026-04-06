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
import pathlib

_ACEGEN_SCRIPT = os.path.normpath(os.path.join(
    _ROOT, "..", "..", "acegen-open", "scripts", "reinvent", "reinvent.py"
))

if not os.path.exists(_ACEGEN_SCRIPT):
    print(f"\n[run_generation] ERROR: ACEGEN script not found at:\n  {_ACEGEN_SCRIPT}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Warm-start checkpoint resolution
# ---------------------------------------------------------------------------
_CKPT_DIR  = pathlib.Path(_ROOT) / "runs" / "checkpoints"
_CKPT_DIR.mkdir(parents=True, exist_ok=True)

_PREV_CKPT = _CKPT_DIR / f"gen_{generation_number - 1:03d}_policy.pt"
_CURR_CKPT = _CKPT_DIR / f"gen_{generation_number:03d}_policy.pt"

_warm_start = _PREV_CKPT.exists() and generation_number > 0

if _warm_start:
    print(f"\n[run_generation] Warm-start: checkpoint found → {_PREV_CKPT}")
else:
    print(f"\n[run_generation] Cold-start: no checkpoint for gen {generation_number - 1}, using ascii.pt prior")

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

# ---------------------------------------------------------------------------
# Warm-start patch — replace ckpt_path in-place after models registry unpack
#
# reinvent.py unpacks: (create_actor, _, _, voc_path, ckpt_path, tokenizer) = models[cfg.model]
# then immediately does: ckpt = torch.load(ckpt_path, ...)
# We insert a single assignment between those two lines to redirect ckpt_path
# to our saved policy. adapt_state_dict() still runs as normal — same
# architecture, identical keys, no schema mismatch possible.
# ---------------------------------------------------------------------------
_WS_ANCHOR = '(create_actor, _, _, voc_path, ckpt_path, tokenizer) = models[cfg.model]'

if _warm_start:
    if _WS_ANCHOR in _script_src:
        _ws_injection = (
            f'\n    # [TRI_FLAG warm-start] override ckpt_path with trained policy from previous generation\n'
            f'    ckpt_path = r"{_PREV_CKPT}"\n'
            f'    print(f"[warm_start] Loading policy weights from: {{ckpt_path}}")\n'
        )
        _script_src = _script_src.replace(_WS_ANCHOR, _WS_ANCHOR + _ws_injection)
        print(f"[run_generation] Warm-start patch applied — ckpt_path redirected to gen_{generation_number - 1:03d} policy")
    else:
        print(f"[run_generation] WARNING: Warm-start anchor not found in reinvent.py — falling back to cold-start")
        print(f"[run_generation]   Expected: '{_WS_ANCHOR}'")

# ---------------------------------------------------------------------------
# Checkpoint-save patch — expose actor_training after the training loop ends
#
# run_reinvent() in reinvent.py ends after the while loop with no return value.
# We inject a global assignment at the bottom of run_reinvent() so that after
# exec() completes, _exec_ns contains _triflag_actor_training and we can call
# .state_dict() on it to save the checkpoint.
#
# Injection point: the blank line immediately before def get_log_prob(...),
# which is the first function defined after run_reinvent() closes.
# ---------------------------------------------------------------------------
_SAVE_ANCHOR = '\ndef get_log_prob(data, model):'

if _SAVE_ANCHOR in _script_src:
    _save_injection = (
        '\n\n    # [TRI_FLAG checkpoint] expose trained actor for post-exec save\n'
        '    global _triflag_actor_training\n'
        '    _triflag_actor_training = actor_training\n'
    )
    _script_src = _script_src.replace(_SAVE_ANCHOR, _save_injection + _SAVE_ANCHOR)
    print(f"[run_generation] Checkpoint-save patch applied — actor_training will be captured after run")
else:
    print(f"[run_generation] WARNING: Checkpoint-save anchor not found in reinvent.py — checkpoint will not be saved")
    print(f"[run_generation]   Expected: '{_SAVE_ANCHOR.strip()}'")

# ---------------------------------------------------------------------------
# Execute ACEGEN
# ---------------------------------------------------------------------------
_exec_ns = {"__name__": "__main__", "__file__": _ACEGEN_SCRIPT}
exec(compile(_script_src, _ACEGEN_SCRIPT, "exec"), _exec_ns)

# ---------------------------------------------------------------------------
# Save policy checkpoint after run completes
# ---------------------------------------------------------------------------
_actor = _exec_ns.get("_triflag_actor_training")

if _actor is not None and hasattr(_actor, "state_dict"):
    import torch as _torch
    _torch.save(_actor.state_dict(), str(_CURR_CKPT))
    print(f"\n[run_generation] Checkpoint saved → {_CURR_CKPT}")
    print(f"[run_generation] Gen {generation_number + 1} will warm-start from this checkpoint")
else:
    # Fallback: scan gc for the largest nn.Module in memory
    print(f"\n[run_generation] actor_training not in exec namespace — trying gc fallback...")
    import gc
    import torch.nn as _nn
    import torch as _torch
    _modules = [
        obj for obj in gc.get_objects()
        if isinstance(obj, _nn.Module)
        and sum(p.numel() for p in obj.parameters()) > 50_000
    ]
    if _modules:
        _policy = max(_modules, key=lambda m: sum(p.numel() for p in m.parameters()))
        _torch.save(_policy.state_dict(), str(_CURR_CKPT))
        print(f"[run_generation] Checkpoint saved via gc fallback → {_CURR_CKPT}")
        print(f"[run_generation] Gen {generation_number + 1} will warm-start from this checkpoint")
    else:
        print(f"[run_generation] WARNING: No checkpoint saved for gen {generation_number}")
        print(f"[run_generation] Gen {generation_number + 1} will cold-start from ascii.pt prior")
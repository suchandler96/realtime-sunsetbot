"""Run main.py for a validated request and collect its text output and figure(s)."""
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOBS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jobs")
PYTHON = sys.executable
JOB_MAX_AGE_SECONDS = 7 * 24 * 60 * 60
JOB_MAX_COUNT = 100


@dataclass
class RunResult:
    ok: bool
    output: str
    figures: list = field(default_factory=list)   # absolute paths
    seconds: float = 0.0


def _pngs():
    return {f: os.path.getmtime(os.path.join(ROOT, f)) for f in os.listdir(ROOT)
            if f.endswith(".png") and not f.startswith("nsmc_captcha")}


def cleanup_jobs(now=None):
    """Remove job directories older than seven days, then keep only the newest 100 by name."""
    if not os.path.isdir(JOBS_DIR):
        return
    now = time.time() if now is None else now
    jobs = sorted(name for name in os.listdir(JOBS_DIR)
                  if os.path.isdir(os.path.join(JOBS_DIR, name)))
    current = []
    for name in jobs:
        path = os.path.join(JOBS_DIR, name)
        if os.path.getmtime(path) < now - JOB_MAX_AGE_SECONDS:
            shutil.rmtree(path)
        else:
            current.append(name)
    for name in current[:-JOB_MAX_COUNT]:
        shutil.rmtree(os.path.join(JOBS_DIR, name))


def collect_figures(before, started):
    """Move the PNGs written since `started` from the project root into email/jobs/<timestamp>/."""
    figures = []
    job_dir = os.path.join(JOBS_DIR, time.strftime("%Y%m%d-%H%M%S"))
    for name, mtime in sorted(_pngs().items()):
        if before.get(name) != mtime and mtime >= started - 1:
            os.makedirs(job_dir, exist_ok=True)
            figures.append(shutil.move(os.path.join(ROOT, name), os.path.join(job_dir, name)))
    return figures


def run_main(argv, timeout=900, max_chars=20000, headless=True):
    """Run `main.py *argv` in the project folder (so its caches are shared); one job at a time.

    The figures it writes are moved into email/jobs/<timestamp>/ and returned.
    """
    cleanup_jobs()
    before = _pngs()
    started = time.time()
    # UTF-8 both ways: with a pipe, Windows would otherwise encode main.py's output (Chinese text, degree signs) as cp1252
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    if headless:  # no GUI backend: plt.show() would otherwise wait for a window nobody sees
        env["MPLBACKEND"] = "Agg"
    try:
        proc = subprocess.run([PYTHON, os.path.join(ROOT, "main.py"), *argv], cwd=ROOT, env=env,
                              stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=timeout)
        output, ok = proc.stdout, proc.returncode == 0
        if not ok:
            output += f"\n--- main.py exited with status {proc.returncode} ---\n" + proc.stderr[-3000:]
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or b"").decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        output += f"\n--- stopped after the {timeout} s time limit ---"
        ok = False
    figures = collect_figures(before, started)
    cleanup_jobs()
    if len(output) > max_chars:
        output = output[:max_chars] + f"\n... (output cut at {max_chars} characters)"
    return RunResult(ok and bool(figures), output, figures, time.time() - started)

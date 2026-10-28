"""Run main.py inside this Python process (no subprocess); same interface as runner.run_main.

main.py is a script, so "calling" it means executing it with `runpy` under a patched sys.argv, with its
stdout/stderr captured, SystemExit turned into a status and the matplotlib figures closed afterwards.
Differences from the subprocess runner: modules imported by main.py stay loaded between requests (faster,
shares process memory), but a hang cannot be killed, so there is no time limit (`timeout` is ignored),
and a hard crash of main.py's native libraries takes the whole service down.
"""
import contextlib
import io
import os
import runpy
import sys
import threading
import time
import traceback

from runner import ROOT, RunResult, _pngs, collect_figures

_lock = threading.Lock()   # main.py uses the process-wide cwd, sys.argv and pyplot state: one run at a time


def run_main(argv, timeout=None, max_chars=20000, headless=True):
    with _lock:
        before, started = _pngs(), time.time()
        if ROOT not in sys.path:
            sys.path.append(ROOT)
        if headless:
            import matplotlib
            matplotlib.use("Agg", force=True)
        out, err = io.StringIO(), io.StringIO()
        saved = sys.argv, os.getcwd()
        status = 0
        os.chdir(ROOT)  # relative cache folders (cams_ads, fy4b_nsmc, ...) live in the project folder
        sys.argv = [os.path.join(ROOT, "main.py"), *argv]
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                try:
                    runpy.run_path(sys.argv[0], run_name="__main__")
                except SystemExit as exc:  # main.py ends with sys.exit("message") on bad input
                    status = exc.code if isinstance(exc.code, int) else 1
                    if isinstance(exc.code, str):
                        print(exc.code)
                except Exception:
                    status = 1
                    traceback.print_exc()
        finally:
            sys.argv = saved[0]
            os.chdir(saved[1])
            import matplotlib.pyplot as plt
            plt.close("all")
        output = out.getvalue()
        if status:
            output += f"\n--- main.py exited with status {status} ---\n" + err.getvalue()[-3000:]
        figures = collect_figures(before, started)
        if len(output) > max_chars:
            output = output[:max_chars] + f"\n... (output cut at {max_chars} characters)"
        return RunResult(status == 0 and bool(figures), output, figures, time.time() - started)

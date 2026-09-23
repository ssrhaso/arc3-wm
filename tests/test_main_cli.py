"""``python -m arc3_wm`` info command.

The CLI is a pure-Python sanity probe: no environment_files, no JAX. We
test the in-process ``main()`` (return code + captured stdout) and a real
``python -m arc3_wm`` subprocess to prove the module is invocable.
"""
from __future__ import annotations

import subprocess
import sys

from arc3_wm import __version__
from arc3_wm.__main__ import main
from arc3_wm.action_space import N_ACTIONS
from arc3_wm.registration import PUBLIC_GAMES, env_id


def test_main_returns_zero(capsys):
    rc = main([])
    assert rc == 0
    out = capsys.readouterr().out
    assert f"arc3_wm {__version__}" in out
    assert f"{N_ACTIONS} actions" in out


def test_main_lists_every_public_game(capsys):
    main([])
    out = capsys.readouterr().out
    assert f"{len(PUBLIC_GAMES)} registered Gymnasium ids" in out
    for game in PUBLIC_GAMES:
        assert env_id(game) in out


def test_module_runs_as_subprocess():
    proc = subprocess.run(
        [sys.executable, "-m", "arc3_wm"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "arc3_wm" in proc.stdout
    assert "ARC3/vc33-v0" in proc.stdout

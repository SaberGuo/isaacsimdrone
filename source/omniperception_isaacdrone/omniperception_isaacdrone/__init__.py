"""
omniperception_isaacdrone package.

IMPORTANT:
- Do NOT eagerly import tasks on package import.
- Isaac Sim python packages (isaacsim.*) are only available after AppLauncher starts.
- Importing tasks too early will raise: ModuleNotFoundError: No module named 'isaacsim.core'
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


__all__ = ["import_tasks"]


def import_tasks() -> Any:
    """
    Lazily import tasks.

    Call this ONLY after IsaacLab AppLauncher has started, e.g.

        from isaaclab.app import AppLauncher
        app_launcher = AppLauncher(args_cli)
        simulation_app = app_launcher.app

        import omniperception_isaacdrone as op
        op.import_tasks()
    """
    return import_module("omniperception_isaacdrone.tasks")

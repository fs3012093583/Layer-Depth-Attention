from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

try:
    import swanlab
except ImportError:  # pragma: no cover - optional dependency
    swanlab = None


@dataclass
class SwanLabConfig:
    project: str = "Layer-Depth-Attention"
    experiment_name: str | None = None


class SwanLabMonitor:
    def __init__(self, config: SwanLabConfig) -> None:
        self.config = config
        self.enabled = swanlab is not None
        self._experiment = None

        if not self.enabled:
            return

        try:
            swanlab.login()
            print("[swanlab] login succeeded", file=sys.stderr)
        except Exception as exc:
            print(f"[swanlab] login failed: {exc!r}", file=sys.stderr)
            self.enabled = False

    def init_experiment(self, run_config: dict[str, Any] | None = None) -> Any:
        if not self.enabled:
            print("[swanlab] init skipped: monitor disabled", file=sys.stderr)
            return None

        try:
            self._experiment = swanlab.init(
                project=self.config.project,
                experiment_name=self.config.experiment_name,
                config=run_config,
            )
            print(
                f"[swanlab] init succeeded: project={self.config.project} "
                f"experiment={self.config.experiment_name}",
                file=sys.stderr,
            )
        except Exception as exc:
            print(f"[swanlab] init failed: {exc!r}", file=sys.stderr)
            self.enabled = False
            self._experiment = None
        return self._experiment

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if not self.enabled:
            return
        try:
            swanlab.log(metrics, step=step)
        except Exception as exc:
            print(f"[swanlab] log failed at step={step}: {exc!r}", file=sys.stderr)
            self.enabled = False

    def finish(self) -> None:
        if not self.enabled:
            return
        try:
            swanlab.finish()
            print("[swanlab] finish succeeded", file=sys.stderr)
        except Exception as exc:
            print(f"[swanlab] finish failed: {exc!r}", file=sys.stderr)
            self.enabled = False


class NullMonitor:
    enabled = False

    def init_experiment(self, run_config: dict[str, Any] | None = None) -> None:
        return None

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        return None

    def finish(self) -> None:
        return None


def build_monitor(
    backend: str,
    project: str,
    experiment_name: str | None,
) -> SwanLabMonitor | NullMonitor:
    if backend != "swanlab":
        return NullMonitor()
    return SwanLabMonitor(SwanLabConfig(project=project, experiment_name=experiment_name))

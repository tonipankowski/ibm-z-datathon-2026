"""AquaWatch AI: anomaly detection and pipe localization.

Exports are loaded lazily so `import ai` stays light and `python -m ai.detect`
doesn't import ai.detect twice.
"""

from __future__ import annotations

from typing import Any

__all__ = ["LeakDetector", "load_network"]


def __getattr__(name: str) -> Any:
    if name == "LeakDetector":
        from .detect import LeakDetector
        return LeakDetector
    if name == "load_network":
        from .topology import load_network
        return load_network
    raise AttributeError(f"module 'ai' has no attribute {name!r}")

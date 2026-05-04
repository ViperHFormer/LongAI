from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ToolConfig:
    backend: str = "local_hf"
    cache_dir: Path | None = None
    asr_model_size: str | None = None


class BaseTool:
    def __init__(self, config: ToolConfig):
        self.config = config

    def run(self, *args, **kwargs):
        raise NotImplementedError

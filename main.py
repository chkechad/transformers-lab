"""Main module of the Transformers Lab project."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


def define_env(env: Any) -> None:
    """Inject project metadata into MkDocs macros environment."""
    pyproject_path = Path("pyproject.toml")

    if not pyproject_path.exists():
        raise FileNotFoundError("pyproject.toml not found at project root.")

    data: dict[str, Any] = tomllib.loads(pyproject_path.read_text())
    version: str = data["project"]["version"]

    env.variables["project_version"] = version

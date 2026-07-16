"""Export the existing tourism-agent LangGraph as Mermaid artifacts."""

import sys
from pathlib import Path

from app.graph.workflow import build_graph
from app.utils.config import PROJECT_ROOT


ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MERMAID_PATH = ARTIFACTS_DIR / "tourism_agent_graph.mmd"
PNG_PATH = ARTIFACTS_DIR / "tourism_agent_graph.png"


def main() -> int:
    """Compile the graph and export Mermaid text plus an optional PNG."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    compiled_graph = build_graph()
    drawable_graph = compiled_graph.get_graph()

    MERMAID_PATH.write_text(
        drawable_graph.draw_mermaid(),
        encoding="utf-8",
    )
    exported_files: list[Path] = [MERMAID_PATH]

    try:
        PNG_PATH.write_bytes(drawable_graph.draw_mermaid_png())
    except Exception as exc:
        print(
            f"Warning: PNG rendering is unavailable: {exc}",
            file=sys.stderr,
        )
    else:
        exported_files.append(PNG_PATH)

    print("Exported graph files:")
    for path in exported_files:
        print(path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

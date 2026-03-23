from __future__ import annotations

import streamlit.web.cli as stcli
import sys
from pathlib import Path


if __name__ == "__main__":
    script = Path(__file__).parent / "app" / "ui" / "pages" / "dataset_builder.py"
    sys.argv = ["streamlit", "run", str(script)]
    sys.exit(stcli.main())

from __future__ import annotations

import streamlit.web.cli as stcli
import sys
from pathlib import Path


if __name__ == "__main__":
    app_script = Path(__file__).parent / "app" / "ui" / "app.py"
    sys.argv = ["streamlit", "run", str(app_script)]
    raise SystemExit(stcli.main())

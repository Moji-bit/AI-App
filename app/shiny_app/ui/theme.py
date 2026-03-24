from __future__ import annotations


def app_styles() -> str:
    return """
:root {
  --bg: #0b1020;
  --card: #121933;
  --card-soft: #1a2344;
  --text: #e8edff;
  --muted: #a8b1d6;
  --accent: #67b7ff;
  --accent-2: #8b7dff;
  --ok: #39d98a;
  --warn: #ffcf5c;
  --danger: #ff6b7a;
}
body {
  background: radial-gradient(circle at top right, #1d2b55 0%, var(--bg) 52%);
  color: var(--text);
}
.app-shell { max-width: 1400px; margin: 0 auto; padding: 1.4rem; }
.hero {
  padding: 1.25rem 1.6rem;
  border-radius: 16px;
  margin-bottom: 1rem;
  background: linear-gradient(120deg, rgba(103,183,255,.23), rgba(139,125,255,.2));
  border: 1px solid rgba(255,255,255,.14);
}
.hero h1 { margin: 0; font-size: 1.65rem; font-weight: 700; }
.hero p { margin: .35rem 0 0; color: var(--muted); }
.card {
  background: rgba(18, 25, 51, .92);
  border: 1px solid rgba(255,255,255,.1);
  border-radius: 14px;
  padding: .95rem;
  margin-bottom: .9rem;
  box-shadow: 0 8px 20px rgba(0,0,0,.25);
}
.kpis { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: .75rem; }
.kpi {
  background: rgba(26,35,68,.95);
  border: 1px solid rgba(255,255,255,.09);
  border-radius: 10px;
  padding: .6rem .75rem;
}
.kpi .label { color: var(--muted); font-size: .8rem; }
.kpi .value { font-size: 1.15rem; font-weight: 600; }
.navbar { border-bottom: 1px solid rgba(255,255,255,.08); }
.form-control, .form-select, .btn, .shiny-input-container { border-radius: 10px !important; }
@media (max-width: 1000px) {
  .kpis { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
"""

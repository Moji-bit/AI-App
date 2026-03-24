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
.card, .card h1, .card h2, .card h3, .card h4, .card h5, .card h6,
.card label, .card .form-label, .card .form-text, .card p, .card span {
  color: var(--text);
}
.card .text-muted, .card small, .card .help-block { color: var(--muted) !important; }
.form-control, .form-select {
  background: rgba(11,16,32,.7);
  color: var(--text);
  border: 1px solid rgba(255,255,255,.22);
}
.form-control::placeholder { color: var(--muted); }
.form-control:focus, .form-select:focus {
  background: rgba(11,16,32,.9);
  color: var(--text);
  border-color: rgba(103,183,255,.65);
  box-shadow: 0 0 0 .2rem rgba(103,183,255,.2);
}
.btn { color: #f7fbff; }
.btn-outline-secondary { color: var(--text); border-color: rgba(255,255,255,.3); }
.btn:disabled, .btn.disabled {
  color: rgba(232,237,255,.65) !important;
  border-color: rgba(255,255,255,.18) !important;
}
.shiny-text-output, .shiny-html-output, .shiny-input-container > label { color: var(--text); }
pre, .form-control[readonly], .shiny-text-output pre {
  color: var(--text);
  background: rgba(11,16,32,.75);
}
.table, .table th, .table td,
.dataframe, .dataframe th, .dataframe td {
  color: var(--text) !important;
}
.table-striped > tbody > tr:nth-of-type(odd) > * {
  color: var(--text) !important;
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

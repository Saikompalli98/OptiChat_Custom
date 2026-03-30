"""
Rolling Horizon Comparison Framework.

A standalone wrapper on top of OptiChat for comparing two optimization model
epochs side-by-side. Registered as a separate ADK app ('optichat_rh') so it
shares the same server process without touching OptiChat's state namespace.

Package layout:
  rh_comparison/
  ├── config/          — state keys, thresholds, path constants
  ├── data_store/      — EpochStore: data persistence + lazy accessors
  ├── analytics/       — QT1–QT4 pure-Python and solver-based analytics
  ├── agents/          — ADK agent factories and prompts
  └── tools/           — ADK tool wrappers and callbacks
"""

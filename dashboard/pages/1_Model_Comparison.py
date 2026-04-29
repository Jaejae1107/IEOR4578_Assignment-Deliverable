"""
Streamlit sub-page: Model / Cluster Comparison.

Lives at ``dashboard/pages/1_Model_Comparison.py`` so that, when the user
runs ``streamlit run dashboard/app.py``, this page becomes the second tab
in the sidebar.

It simply delegates to ``dashboard.render()`` to avoid duplicating layout
logic.  Shared CSS theme and the gradient hero banner are also rendered
inside ``render`` for visual parity with the chatbot page.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# The dashboard.py module lives one level up.
_THIS_FILE = Path(__file__).resolve()
_DASHBOARD_DIR = _THIS_FILE.parents[1]
_PROJECT_ROOT = _DASHBOARD_DIR.parent
for p in (_DASHBOARD_DIR, _PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dashboard import render as render_dashboard  # noqa: E402

st.set_page_config(
    page_title="FortuneTellers - Model Comparison",
    page_icon=None,
    layout="wide",
)

render_dashboard("Model / Cluster Comparison")

"""Streamlit entrypoint for Pathfinder.

Netlify can't run this (it's a Python server app), but platforms like Streamlit Community Cloud,
Render, Railway, Fly.io, etc. can.

Run locally:
  pip install -r requirements.txt
  streamlit run app.py

Optional:
  set CALL_INDEX_PATH=data/calls_index.csv
"""

import pathfinder_v4  # noqa: F401  (import runs the Streamlit app)

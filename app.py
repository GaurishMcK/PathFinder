import streamlit as st
import pathfinder_v4

st.set_page_config(page_title="Pathfinder", layout="wide")

# Call into the module explicitly
if hasattr(pathfinder_v4, "main"):
    pathfinder_v4.main()
else hasattr(pathfinder_v4, "run"):
    pathfinder_v4.run()

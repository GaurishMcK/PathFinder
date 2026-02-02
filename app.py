import streamlit as st
import pathfinder_v4

st.set_page_config(page_title="Pathfinder", layout="wide")

# Call into the module explicitly
if hasattr(pathfinder_v4, "main"):
    pathfinder_v4.main()
elif hasattr(pathfinder_v4, "run"):
    pathfinder_v4.run()
else:
    # Last resort: if pathfinder_v4 still has top-level Streamlit code, importing it would have shown UI already
    st.error("pathfinder_v4 imported, but no main() or run() was found, and it didn't render UI on import.")
    st.write("Open pathfinder_v4.py and either:")
    st.write("- put the Streamlit UI at top-level, OR")
    st.write("- wrap it in main() and call it from app.py (recommended).")

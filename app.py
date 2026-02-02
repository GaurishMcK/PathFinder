import streamlit as st

st.set_page_config(page_title="Pathfinder", layout="wide")
st.write("✅ Streamlit started. Importing app…")

try:
    import pathfinder_v4  # noqa: F401
    st.write("✅ Imported pathfinder_v4 successfully.")
except Exception as e:
    st.error("❌ App crashed during startup.")
    st.exception(e)
    st.stop()

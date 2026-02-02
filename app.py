import streamlit as st

st.set_page_config(page_title="Pathfinder", layout="wide")
st.write("Booting Pathfinder...")  # proves the app started

try:
    import pathfinder_v4  # noqa: F401
except Exception as e:
    st.error("App crashed during startup.")
    st.exception(e)
    st.stop()

"""Entry point — Genel Bakış'a yönlendirir, sidebar'dan 'app' gizlenir."""
import streamlit as st

st.set_page_config(page_title="IoT Intrusion Detection", page_icon="�️", layout="wide")

# Sidebar'dan "app" girişini gizle
st.markdown(
    """<style>[data-testid="stSidebarNav"] li:first-child {display: none;}</style>""",
    unsafe_allow_html=True,
)

st.switch_page("pages/1_Genel_Bakış.py")

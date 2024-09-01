import streamlit as st

pg = st.navigation([
    st.Page("pages/rag_page.py"),
])
pg.run()



import streamlit as st
from dotenv import load_dotenv

load_dotenv(
    override=True,
    verbose=True,
)
st.title("template-streamlit")
st.info("Select a code sample from the sidebar to run it")

import streamlit as st

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    st.video(uploaded_file, format="video/mp4", autoplay=True)
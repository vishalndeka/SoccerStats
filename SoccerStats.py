import cv2
import streamlit as st
import tempfile
from app import full_detection_tracking

if __name__=='__main__':
    uploaded_file = st.file_uploader("Upload Match Footage")
    temp_dir = "temp"
    if uploaded_file is not None:
        # trackFile = full_detection_tracking(uploaded_file.name, 50)
        # st.video(trackFile, autoplay=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name, dir=temp_dir) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
            st.text(tmp_file_path)
        
            track_file = full_detection_tracking(tmp_file_path)

            
        
        # full_detection_tracking(uploaded_file.name)


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import requests

st.set_page_config(page_title="EcoType: Forest Cover Predictor", page_icon=":)", layout="wide")

@st.cache_resource
def load_ml_pipeline():
    local_path = 'forest_cover_pipeline.joblib'
    
    # 🚨 PUSH YOUR ACTUAL ID HERE
    GOOGLE_DRIVE_FILE_ID = "1R3fw6pNzYZsyiOG2armiweF2hbDX7tke"
    
    if not os.path.exists(local_path):
        with st.spinner("🔄 Downloading machine learning model artifacts from cloud storage... This may take a moment."):
            try:
                # Use a requests session to capture Google's large file virus scan redirect token
                session = requests.Session()
                download_url = "https://docs.google.com/uc?export=download"
                
                # First request to check for verification token
                response = session.get(download_url, params={'id': GOOGLE_DRIVE_FILE_ID}, stream=True)
                
                token = None
                for key, value in response.cookies.items():
                    if key.startswith('download_warning'):
                        token = value
                        break
                
                # If a large file warning token is present, make a second request passing the token
                if token:
                    params = {'id': GOOGLE_DRIVE_FILE_ID, 'confirm': token}
                    response = session.get(download_url, params=params, stream=True)
                
                # Save the stream safely to disk
                with open(local_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=32768):
                        if chunk: 
                            f.write(chunk)
                            
                # Sanity check: Ensure we didn't just download a tiny error string or HTML file
                if os.path.getsize(local_path) < 100000: # less than 100KB means it failed
                    with open(local_path, 'r', errors='ignore') as f:
                        content_peek = f.read(500)
                    st.error("❌ The file downloaded from Google Drive is not your model. It appears to be an access error page.")
                    st.code(content_peek)
                    st.stop()
                    
            except Exception as e:
                st.error("⚠️ Failed to stream the model from cloud storage.")
                st.exception(e)
                st.stop()
                
    return joblib.load(local_path)

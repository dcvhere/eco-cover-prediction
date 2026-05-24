import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import requests

st.set_page_config(page_title="EcoType: Forest Cover Predictor", page_icon="🌿", layout="wide")

@st.cache_resource
def load_ml_pipeline():
    local_path = 'forest_cover_pipeline.joblib'
    
    # 🚨 REPLACE THIS WITH YOUR EXACT GOOGLE DRIVE FILE ID
    # (The alphanumeric string between /d/ and /view in your share link)
    GOOGLE_DRIVE_FILE_ID = "1R3fw6pNzYZsyiOG2armiweF2hbDX7tke"
    
    if not os.path.exists(local_path):
        with st.spinner("📥 Fetching 75MB model pipeline from Google Drive... This may take a moment."):
            try:
                # Direct endpoint for files under 100MB
                download_url = f"https://docs.google.com/uc?export=download&id={GOOGLE_DRIVE_FILE_ID}"
                
                # Use a request stream to write the binary chunks directly to the server disk
                response = requests.get(download_url, stream=True)
                response.raise_for_status() # Raise error if download link is broken
                
                with open(local_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            
                # Sanity check validation
                file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
                if file_size_mb < 50: # If it's way smaller than your 75MB file, something went wrong
                    st.error(f"❌ Download completed, but the file size ({file_size_mb:.2f} MB) is incorrect. Check your File ID.")
                    st.stop()
                    
            except Exception as e:
                st.error("⚠️ Streamlit failed to download the model from cloud storage.")
                st.exception(e)
                st.stop()
                
    return joblib.load(local_path)

# Main layout presentation and application code continues below...

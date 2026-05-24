import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="EcoType: Forest Cover Predictor", page_icon="", layout="wide")

# Cached loading routine to prevent reloading the pipeline on every widget interaction
@st.cache_resource
def load_ml_pipeline():
    # Look for the compressed joblib file directly in the repository root directory
    model_path = 'forest_cover_pipeline.joblib'
        
    if not os.path.exists(model_path):
        st.error(f"⚠️ Project artifact file '{model_path}' could not be discovered in the root directory.")
        st.info("💡 Please ensure you have uploaded your compressed 'forest_cover_pipeline.joblib' file directly to the main folder of your GitHub repository.")
        st.stop()
        
    return joblib.load(model_path)

# Main layout presentation and title
st.title("🌿 EcoType: Forest Cover Classification Tool")
st.markdown("Provide geographical, cartographic, and environmental inputs to evaluate the dominant forest cover type class designation.")
st.markdown("---")

# Load the compiled machine learning artifacts
pipeline = load_ml_pipeline()
model = pipeline['model']
scaler = pipeline['scaler']
encoders = pipeline['encoders']
feature_names = pipeline['feature_names']

# Build 3 layout presentation panels
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🗺️ Cartographic Parameters")
    elevation = st.number_input("Elevation (meters above sea level)", min_value=0, max_value=6000, value=2900)
    aspect = st.slider("Aspect Slope Orientation (degrees 0-360)", min_value=0, max_value=360, value=180)
    slope = st.slider("Terrain Slope Steepness (degrees)", min_value=0, max_value=90, value=15)

with col2:
    st.subheader("💧 Hydrology & Transport")
    h_dist_hydrology = st.number_input("Horizontal Dist to Surface Water (m)", min_value=0, value=250)
    v_dist_hydrology = st.number_input("Vertical Dist to Surface Water (m)", value=30)
    h_dist_roadways = st.number_input("Horizontal Dist to Public Roadways (m)", min_value=0, value=1500)
    h_dist_fire_points = st.number_input("Horizontal Dist to Wildfire Ignition Points (m)", min_value=0, value=2000)

with col3:
    st.subheader("☀️ Hillshade Illumination")
    hillshade_9am = st.slider("9:00 AM Illumination Index (0-255)", min_value=0, max_value=255, value=210)
    hillshade_noon = st.slider("12:00 PM Noon Illumination Index (0-255)", min_value=0, max_value=255, value=215)
    hillshade_3pm = st.slider("3:00 PM Illumination Index (0-255)", min_value=0, max_value=255, value=140)

# Sidebar for high cardinality index categories
st.sidebar.header("🏞️ Terrain Profiles")
wilderness_area = st.sidebar.selectbox("Designated Wilderness Zone Profile", options=list(range(0, 4)))
soil_type = st.sidebar.selectbox("Soil Classification Index Type", options=list(range(0, 40)))

st.markdown("---")

if st.button("🚀 Calculate Predictive Cover Mapping"):
    # Apply identical log transformations used during pipeline development 
    h_dist_hyd_trans = np.log1p(h_dist_hydrology)
    h_dist_road_trans = np.log1p(h_dist_roadways)
    h_dist_fire_trans = np.log1p(h_dist_fire_points)
    
    # Calculate identical engineered features
    total_dist_hydrology = np.sqrt(h_dist_hyd_trans**2 + v_dist_hydrology**2)
    shade_9am_to_noon = hillshade_9am - hillshade_noon
    shade_noon_to_3pm = hillshade_noon - hillshade_3pm
    
    # Assemble feature data dictionary mapping 
    input_payload = {
        'Elevation': elevation,
        'Aspect': aspect,
        'Slope': slope,
        'Horizontal_Distance_To_Hydrology': h_dist_hyd_trans,
        'Vertical_Distance_To_Hydrology': v_dist_hydrology,
        'Horizontal_Distance_To_Roadways': h_dist_road_trans,
        'Hillshade_9am': hillshade_9am,
        'Hillshade_Noon': hillshade_noon,
        'Hillshade_3pm': hillshade_3pm,
        'Horizontal_Distance_To_Fire_Points': h_dist_fire_trans,
        'Wilderness_Area': wilderness_area,
        'Soil_Type': soil_type,
        'Total_Distance_To_Hydrology': total_dist_hydrology,
        'Shade_9am_to_Noon_Diff': shade_9am_to_noon,
        'Shade_Noon_to_3pm_Diff': shade_noon_to_3pm
    }
    
    # Coerce structural dataframe matching original positioning parameters
    input_df = pd.DataFrame([input_payload])[feature_names]
    
    # Run data layout through fitted system standardizer matrix
    scaled_features = scaler.transform(input_df)
    
    # Predict target class
    encoded_prediction = model.predict(scaled_features)[0]
    
    # Check if target inverse transform available, otherwise map default strings
    if 'Cover_Type' in encoders:
        final_class_label = encoders['Cover_Type'].inverse_transform([encoded_prediction])[0]
    else:
        cover_mapping = {0: "Spruce/Fir", 1: "Lodgepole Pine", 2: "Ponderosa Pine", 
                         3: "Cottonwood/Willow", 4: "Aspen", 5: "Douglas-fir", 6: "Krummholz"}
        final_class_label = cover_mapping.get(encoded_prediction, f"Type Index {encoded_prediction}")
        
    st.balloons()
    st.success(f"🎯 **Model Predicted Forest Cover Outcome:** `{final_class_label}`")

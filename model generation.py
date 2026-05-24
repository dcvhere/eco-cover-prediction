## STEP 1: ENVIRONMENT SETUP & DEPENDENCY INSTALLATION
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Machine Learning Classification Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# Class Imbalance Handling
from imblearn.over_sampling import SMOTE

# to produce compressed/shrikned model file
import joblib

"""## 2 DATA COLLECTION & UNDERSTANDING"""

def load_and_understand_data(file_name):
    print("--- 📥 1. Loading Dataset ---")
    colab_path = f"/content/cover_type (1).csv"

    if os.path.exists(colab_path):
        df = pd.read_csv(colab_path)
    elif os.path.exists(file_name):
        df = pd.read_csv(file_name)
    else:
        raise FileNotFoundError(
            f"❌ Could not find '{file_name}'. Please click the folder icon on the left "
            "sidebar in Google Colab and upload your CSV file directly there."
        )

    print(f"✅ Data Loaded successfully! Shape: {df.shape[0]} rows, {df.shape[1]} columns.")

    print("\n--- 🔍 2. Understanding Data Structure ---")
    print("\nMissing Values per column:")
    print(df.isnull().sum())

    print("\nTarget Class Distribution ('Cover_Type'):")
    print(df['Cover_Type'].value_counts())

    return df

df = load_and_understand_data(uploaded_filename)
df

"""## 3 DATA CLEANING & TRANSFORMATION"""

def clean_and_transform(df):
    print("\n--- 🔧 3. Data Cleaning & Transformation ---")
    df_cleaned = df.copy()

    # 3.1 Handle missing values dynamically by filling with column median
    num_cols = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
    if 'Cover_Type' in num_cols:
        num_cols.remove('Cover_Type')

    for col in num_cols:
        if df_cleaned[col].isnull().sum() > 0:
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())

    # 3.2 Cap outliers using the Interquartile Range (IQR) method
    continuous_features = ['Elevation', 'Aspect', 'Slope',
                           'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
                           'Horizontal_Distance_To_Roadways', 'Horizontal_Distance_To_Fire_Points']

    print("Capping outliers using the IQR technique...")
    for col in continuous_features:
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_cleaned[col] = np.clip(df_cleaned[col], lower_bound, upper_bound)

    # 3.3 Treat skewness on highly skewed continuous distance features using log1p
    print("Treating feature skewness using log1p transformation...")
    skewed_cols = ['Horizontal_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 'Horizontal_Distance_To_Fire_Points']
    for col in skewed_cols:
        df_cleaned[col] = np.log1p(df_cleaned[col])

    return df_cleaned

"""## 4 FEATURE ENGINEERING"""

def feature_engineering(df):
    print("\n--- 🧠 4. Feature Engineering ---")
    df_fe = df.copy()

    # 4.1 Create derived columns to assist model performance
    print("Generating derived engineering features...")
    df_fe['Total_Distance_To_Hydrology'] = np.sqrt(
        df_fe['Horizontal_Distance_To_Hydrology']**2 + df_fe['Vertical_Distance_To_Hydrology']**2
    )
    df_fe['Shade_9am_to_Noon_Diff'] = df_fe['Hillshade_9am'] - df_fe['Hillshade_Noon']
    df_fe['Shade_Noon_to_3pm_Diff'] = df_fe['Hillshade_Noon'] - df_fe['Hillshade_3pm']

    # 4.2 Encode categorical columns if present as strings/objects
    categorical_cols = ['Wilderness_Area', 'Soil_Type']
    encoders = {}

    for col in categorical_cols:
        if df_fe[col].dtype == 'object':
            print(f"Encoding categorical column: {col}")
            le = LabelEncoder()
            df_fe[col] = le.fit_transform(df_fe[col].astype(str))
            encoders[col] = le

    # Always explicitly encode the target variable to avoid XGBoost indexing conflicts
    target_encoder = LabelEncoder()
    df_fe['Cover_Type'] = target_encoder.fit_transform(df_fe['Cover_Type'])
    encoders['Cover_Type'] = target_encoder

    return df_fe, encoders

"""## 5 SPLITTING, SAMPLING & SCALING"""

def prepare_data_splits(df):
    print("\n--- 📊 5, 6 & 7. Data Preparation Split & Scaling ---")
    X = df.drop(columns=['Cover_Type'])
    y = df['Cover_Type']

    # Train-test split with stratification to maintain class percentages
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 6️⃣ Address Class Imbalance using SMOTE on training data only
    print("Applying SMOTE to balance the target class representation...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # 7️⃣ Standard Scale features based on the training data statistics
    scaler = StandardScaler()
    X_train_res = scaler.fit_transform(X_train_res)
    X_test = scaler.transform(X_test)

    return X_train_res, X_test, y_train_res, y_test, scaler, X.columns

"""## 6 MODEL BUILDING & HYPERPARAMETER TUNING"""

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    print("\n--- 🤖 8. Model Building & Comparative Evaluation ---")

    # Defining the 5 algorithms required by specifications
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(random_state=42, eval_metric='mlogloss', n_jobs=-1)
    }

    best_acc = 0
    best_model_name = None
    best_model_obj = None

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"🎯 Classifier Model: {name:20} -> Test Accuracy Score: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_model_name = name
            best_model_obj = model

    print(f"\n🥇 Best baseline performing model architecture: {best_model_name}")

    # Tuning the selected optimal algorithm
    print(f"\n--- 🛠️ Hyperparameter Tuning on {best_model_name} ---")
    if best_model_name == "Random Forest":
        param_dist = {
            'n_estimators': [50, 100],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        }
    else:  # Backup default to XGBoost framework parameters
        param_dist = {
            'n_estimators': [50, 100],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.1, 0.2]
        }

    tuned_search = RandomizedSearchCV(best_model_obj, param_distributions=param_dist,
                                       n_iter=3, cv=3, random_state=42, n_jobs=-1)
    tuned_search.fit(X_train, y_train)
    final_model = tuned_search.best_estimator_

    print("\n📊 Final Tuned Model Evaluation Report:")
    final_preds = final_model.predict(X_test)
    print(classification_report(y_test, final_preds))

    return final_model

"""## 7 PIPELINE EXECUTION & ARTIFACT EXPORT"""

if __name__ == "__main__":
    # ⚠️ EDIT THIS STRING IF YOUR UPLOADED FILE HAS A DIFFERENT NAME
    uploaded_filename = "EcoType_ Forest Cover Classification.csv"

    # Run the full pipeline pipeline steps
    df = load_and_understand_data(uploaded_filename)
    df_cleaned = clean_and_transform(df)
    df_fe, saved_encoders = feature_engineering(df_cleaned)
    X_train, X_test, y_train, y_test, saved_scaler, feature_names = prepare_data_splits(df_fe)
    best_tuned_model = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    # Packaging the pipeline assets for deployment
    os.makedirs('models', exist_ok=True)
    artifacts = {
        'model': best_tuned_model,
        'scaler': saved_scaler,
        'encoders': saved_encoders,
        'feature_names': feature_names.tolist()
    }

    output_path = 'models/forest_cover_pipeline.joblib'
    with open(output_path, 'wb') as f:
        joblib.dump(artifacts, f, compress=("xz", 9))

    print(f"\n✅ SUCCESS! All processing components exported to: '{output_path}'")
    print("👉 Refresh your Colab files directory tab, find the 'models' folder, download 'forest_cover_pipeline.pkl' to your local system.")
print("scikit-learn version:", sklearn.__version__)
print("xgboost version:", xgboost.__version__)
print("joblib version:", joblib.__version__)
print("python version:", sys.version)

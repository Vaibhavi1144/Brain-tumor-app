import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- Page config ---
st.set_page_config(page_title="Brain Tumor Diagnostic Tool", page_icon="🧠", layout="centered")
st.title("🧠 Brain Tumor Diagnostic Tool")

# --- Sidebar navigation ---
page = st.sidebar.selectbox("Navigate", ["Home", "Upload Dataset & Train", "Predict Tumor", "Advice & Tips"])

# --- Home ---
if page == "Home":
    st.subheader("Welcome!")
    st.write("""
    This app allows you to train a machine learning model on your dataset and predict brain tumor type
    based on patient details and symptoms.
    """)

# --- Upload Dataset & Train ---
elif page == "Upload Dataset & Train":
    st.subheader("Upload your dataset")
    uploaded_file = st.file_uploader("Upload Excel (.xlsx) or CSV (.csv) file", type=["csv", "xlsx"])

    if uploaded_file:
        try:
            # Read file
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file, engine="openpyxl")
        except Exception as e:
            st.error(f"Error reading file: {e}")
        else:
            st.write("Dataset Preview:")
            st.dataframe(df.head())

            # --- Preprocessing ---
            st.write("Training model...")

            df_model = df.copy()

            # Encode categorical columns
            le_dict = {}
            for col in df_model.columns:
                if df_model[col].dtype == 'object' and col != 'Tumor_Type':
                    le = LabelEncoder()
                    df_model[col] = df_model[col].astype(str)
                    df_model[col] = le.fit_transform(df_model[col])
                    le_dict[col] = le

            # Encode target
            le_target = LabelEncoder()
            df_model['Tumor_Type'] = le_target.fit_transform(df_model['Tumor_Type'].astype(str))

            # Features and target
            X = df_model.drop(['Patient_ID', 'Tumor_Type'], axis=1, errors='ignore')
            y = df_model['Tumor_Type']

            # Fill missing values
            X = X.fillna(0)

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            st.success("✅ Model trained successfully!")

            # Save to session state
            st.session_state['model'] = model
            st.session_state['le_dict'] = le_dict
            st.session_state['le_target'] = le_target
            st.session_state['feature_cols'] = X.columns.tolist()

# --- Predict Tumor ---
elif page == "Predict Tumor":
    if 'model' not in st.session_state:
        st.warning("Please upload dataset and train the model first!")
    else:
        st.subheader("Enter patient details")

        input_data = {}
        for col in st.session_state['feature_cols']:
            if col.lower() == 'age':
                input_data[col] = st.slider("Age", 1, 100)
            elif col.lower() == 'gender':
                input_data[col] = st.selectbox("Gender", ["Male", "Female", "Other"])
            else:
                # For symptom columns
                input_data[col] = st.selectbox(f"{col}", ["No", "Yes"])

        if st.button("Predict"):
            # Convert input to numeric using saved LabelEncoders
            input_df = pd.DataFrame([input_data])
            for col, le in st.session_state['le_dict'].items():
                input_df[col] = le.transform(input_df[col].astype(str))

            # Predict
            model = st.session_state['model']
            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df).max() * 100
            tumor_label = st.session_state['le_target'].inverse_transform([pred])[0]

            st.success(f"Tumor Type: **{tumor_label}**")
            st.info(f"Prediction Confidence: **{prob:.2f}%**")

            # Advice section
            st.subheader("Advice & Tips")
            st.write("""
            - See a neurologist immediately  
            - Maintain healthy lifestyle  
            - Follow prescribed treatment  
            - Regular check-ups  
            - Early diagnosis improves recovery chances
            """)

# --- Advice & Tips ---
elif page == "Advice & Tips":
    st.subheader("Brain Tumor Advice & Tips")
    st.write("""
    1. **Immediate Consultation:** Always consult a neurologist if you notice symptoms like persistent headaches, seizures, or vision problems.  
    2. **Lifestyle:** Eat healthy, exercise regularly, avoid smoking/alcohol.  
    3. **Treatment Compliance:** Follow doctor's prescriptions and attend all check-ups.  
    4. **Early Detection:** Regular health check-ups can help detect tumors early.  
    5. **Support System:** Emotional support from family and friends is crucial.  
    6. **Information:** Stay informed about your condition and treatment options.
    """)

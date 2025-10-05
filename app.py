import streamlit as st
import pandas as pd
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
            # Read dataset based on file type
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file, engine="openpyxl")  # specify engine for Excel
        except Exception as e:
            st.error(f"Error reading file: {e}")
        else:
            st.write("Dataset Preview:")
            st.dataframe(df.head())

            # Preprocessing
            st.write("Training model...")
            df_model = df.copy()

            # Encode categorical columns
            le = LabelEncoder()
            if 'Gender' in df_model.columns:
                df_model['Gender'] = le.fit_transform(df_model['Gender'])
            symptom_cols = [col for col in df_model.columns if 'Symptom' in col]
            for col in symptom_cols:
                df_model[col] = le.fit_transform(df_model[col].astype(str))
            df_model['Tumor_Type'] = le.fit_transform(df_model['Tumor_Type'].astype(str))

            # Train-test split
            X = df_model.drop(['Patient_ID', 'Tumor_Type'], axis=1, errors='ignore')
            y = df_model['Tumor_Type']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            st.success("✅ Model trained successfully!")

            # Save objects to session state
            st.session_state['model'] = model
            st.session_state['le'] = le
            st.session_state['symptom_cols'] = symptom_cols

# --- Predict Tumor ---
elif page == "Predict Tumor":
    if 'model' not in st.session_state:
        st.warning("Please upload dataset and train the model first!")
    else:
        st.subheader("Enter patient details")

        age = st.slider("Age", 1, 100)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        symptoms = st.multiselect(
            "Select Symptoms",
            st.session_state['symptom_cols']
        )

        if st.button("Predict"):
            # Prepare input
            input_data = {}
            input_data['Age'] = age
            input_data['Gender'] = st.session_state['le'].transform([gender])[0]
            for col in st.session_state['symptom_cols']:
                input_data[col] = 1 if col in symptoms else 0
            input_df = pd.DataFrame([input_data])

            # Predict
            model = st.session_state['model']
            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df).max() * 100
            tumor_label = st.session_state['le'].inverse_transform([pred])[0]

            st.success(f"Tumor Type: **{tumor_label}**")
            st.info(f"Probability: **{prob:.2f}%**")

            # Advice
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

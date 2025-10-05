import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import matplotlib.pyplot as plt

st.set_page_config(page_title="Brain Tumor Diagnostic Tool", page_icon="🧠", layout="centered")
st.title("🧠 Brain Tumor Diagnostic Tool")

page = st.sidebar.selectbox("Navigate", ["Home", "Upload Dataset & Train", "Predict Tumor", "Advice & Tips"])

# --- Home ---
if page == "Home":
    st.subheader("Welcome!")
    st.write("""
    Predict brain tumor type, stage, and location based on patient symptoms and important details.
    Only key information is required from the user.
    """)

# --- Upload & Train ---
elif page == "Upload Dataset & Train":
    st.subheader("Upload your dataset")
    uploaded_file = st.file_uploader("Upload Excel (.xlsx) or CSV (.csv)", type=["csv","xlsx"])

    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file, engine="openpyxl")
        except Exception as e:
            st.error(f"Error reading file: {e}")
        else:
            st.write("Dataset preview:")
            st.dataframe(df.head())

            # --- Preprocessing ---
            df_model = df.copy()

            # Identify important columns for input
            important_cols = ['Gender','Age'] + [col for col in df_model.columns if 'Symptom' in col]

            # Encode categorical features
            le_dict = {}
            for col in important_cols:
                if df_model[col].dtype == 'object':
                    le = LabelEncoder()
                    df_model[col] = le.fit_transform(df_model[col].astype(str))
                    le_dict[col] = le

            # Encode targets
            target_cols = ['Tumor_Type','Tumor_Stage','Tumor_Location']
            le_target = {}
            for col in target_cols:
                le = LabelEncoder()
                df_model[col] = le.fit_transform(df_model[col].astype(str))
                le_target[col] = le

            # Features and target
            X = df_model[important_cols].fillna(0)
            y = df_model[target_cols]

            # Train multi-output classifier
            model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
            model.fit(X, y)
            st.success("✅ Model trained successfully!")

            # Save to session
            st.session_state['model'] = model
            st.session_state['le_dict'] = le_dict
            st.session_state['feature_cols'] = important_cols
            st.session_state['target_cols'] = target_cols
            st.session_state['symptom_cols'] = [c for c in important_cols if c not in ['Gender','Age']]

# --- Predict Tumor ---
elif page == "Predict Tumor":
    if 'model' not in st.session_state:
        st.warning("Please upload dataset and train the model first!")
    else:
        st.subheader("Enter patient details")

        # Patient ID input
        patient_id = st.text_input("Patient ID")

        # User inputs
        gender = st.selectbox("Gender", ["Male","Female","Other"])
        age = st.slider("Age", 1, 100)
        selected_symptoms = st.multiselect("Select Symptoms:", st.session_state['symptom_cols'])

        if st.button("Predict"):
            if not patient_id:
                st.warning("Please enter Patient ID")
            else:
                input_data = {'Gender':gender, 'Age':age}
                for col in st.session_state['symptom_cols']:
                    input_data[col] = "Yes" if col in selected_symptoms else "No"

                input_df = pd.DataFrame([input_data])

                # Encode input
                for col, le in st.session_state['le_dict'].items():
                    input_df[col] = le.transform(input_df[col].astype(str))

                # Predict
                model = st.session_state['model']
                preds = model.predict(input_df)[0]
                probs_list = model.predict_proba(input_df)

                # Decode predictions
                tumor_type = st.session_state['le_target']['Tumor_Type'].inverse_transform([preds[0]])[0]
                tumor_stage = st.session_state['le_target']['Tumor_Stage'].inverse_transform([preds[1]])[0]
                tumor_location = st.session_state['le_target']['Tumor_Location'].inverse_transform([preds[2]])[0]

                # Display results
                st.subheader(f"Patient ID: {patient_id}")
                st.success(f"**Tumor Type:** {tumor_type}")
                st.info(f"**Tumor Stage:** {tumor_stage}")
                st.info(f"**Tumor Location:** {tumor_location}")

                # Probability charts for each target
                for i, col in enumerate(st.session_state['target_cols']):
                    probs = probs_list[i][0]*100
                    labels = st.session_state['le_target'][col].classes_
                    fig, ax = plt.subplots()
                    ax.bar(labels, probs, color='skyblue')
                    ax.set_ylabel("Probability (%)")
                    ax.set_title(f"{col} Probabilities")
                    st.pyplot(fig)

                # Stage-specific advice
                st.subheader("Advice & Tips")
                if tumor_stage in ["Stage I","Stage II"]:
                    st.write("- Early stage detected: Consult a doctor and follow treatment plan.")
                elif tumor_stage == "Stage III":
                    st.write("- Advanced stage: Immediate consultation and treatment required.")
                elif tumor_stage == "Stage IV":
                    st.write("- Critical stage: Urgent medical attention needed.")
                else:
                    st.write("- Follow standard medical advice and regular check-ups.")
                st.write("- Maintain healthy lifestyle, regular check-ups, and emotional support.")

# --- Advice & Tips ---
elif page == "Advice & Tips":
    st.subheader("General Brain Tumor Advice & Tips")
    st.write("""
    1. Immediate Consultation: See a neurologist for persistent symptoms.  
    2. Lifestyle: Eat healthy, exercise regularly, avoid smoking/alcohol.  
    3. Treatment Compliance: Follow doctor's instructions.  
    4. Early Detection: Regular check-ups help detect tumors early.  
    5. Support System: Emotional support is crucial.  
    6. Information: Stay informed about your condition and treatment options.
    """)

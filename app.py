import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import matplotlib.pyplot as plt

st.set_page_config(page_title="Brain Tumor Predictor", page_icon="🧠", layout="wide")
st.title("🧠 Brain Tumor Predictor App")

# --- Page Navigation ---
page = st.sidebar.selectbox("Navigate", [
    "Home", 
    "Brain Tumor Info", 
    "Symptoms & Prevention", 
    "Upload Dataset & Train", 
    "Predict Tumor", 
    "Advice & Tips"
])

# --- Home Page ---
if page == "Home":
    st.subheader("Welcome to the Brain Tumor Predictor App 🧠")
    st.image("https://share.google/images/d7mtQ7pvRJpRAZk0H.jpg", use_container_width=True)
    st.write("""
    Welcome! This app is designed to help patients and healthcare professionals
    understand brain tumors, predict tumor type, stage, and location using patient
    data, and provide advice based on the stage of the tumor.  

    Brain tumors are abnormal growths of cells in the brain that can be either benign
    (non-cancerous) or malignant (cancerous). Early detection is critical for
    effective treatment and better outcomes.  
    """)

# --- Brain Tumor Information Page ---
elif page == "Brain Tumor Info":
    st.subheader("What is a Brain Tumor?")
    st.image("https://share.google/images/f1km9RmVkRjKfYU3X.jpg", use_container_width=True)
    st.write("""
    A brain tumor is a mass or growth of abnormal cells in your brain. There are two
    main types: **benign** (non-cancerous) and **malignant** (cancerous).  

    Brain tumors can originate in the brain (**primary tumors**) or spread from other
    parts of the body (**secondary or metastatic tumors**).  

    **How brain tumors are harmful:**  
    - **Increased pressure:** Tumors can increase intracranial pressure, causing headaches, nausea, and vomiting.  
    - **Neurological deficits:** Depending on location, tumors can affect speech, vision, movement, or memory.  
    - **Seizures:** Tumors can trigger seizures, which may be severe or frequent.  
    - **Life-threatening:** Malignant brain tumors can grow rapidly and spread, making early detection and treatment critical.  
    """)

# --- Symptoms and Prevention Page ---
elif page == "Symptoms & Prevention":
    st.subheader("Basic Symptoms and Prevention of Brain Tumors")
    st.write("""
    **Common Symptoms:**  
    - Persistent headaches, often worse in the morning or at night  
    - Nausea and vomiting unrelated to other causes  
    - Vision or hearing problems  
    - Difficulty with balance or walking  
    - Changes in personality or cognitive functions  
    - Seizures or convulsions  
    - Weakness or numbness in arms or legs  

    **Prevention Tips:**  
    - Maintain a healthy lifestyle: eat a balanced diet, exercise regularly, and avoid smoking and alcohol  
    - Protect your head from injury  
    - Avoid exposure to radiation or harmful chemicals  
    - Regular health check-ups for early detection of any neurological changes  
    - Be aware of family history of brain tumors and report any unusual symptoms to a healthcare professional promptly  
    """)

# --- Upload & Train Section ---
elif page == "Upload Dataset & Train":
    st.subheader("Step 1: Upload Dataset")
    uploaded_file = st.file_uploader("Upload Excel (.xlsx) or CSV (.csv) with Patient ID, Age, Gender, Symptoms, Tumor Details", type=["csv","xlsx"])

    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file, engine="openpyxl")
        except Exception as e:
            st.error(f"Error reading file: {e}")
        else:
            st.success("Dataset loaded successfully!")
            st.write("Dataset Preview:")
            st.dataframe(df.head())

            if 'Patient_ID' not in df.columns:
                st.error("Dataset must contain a 'Patient_ID' column!")
            else:
                # Features & Targets
                feature_cols = ['Gender','Age'] + [col for col in df.columns if 'Symptom' in col]
                feature_cols = [col for col in feature_cols if col in df.columns]
                target_cols = ['Tumor_Type','Tumor_Stage','Tumor_Location']
                target_cols = [col for col in target_cols if col in df.columns]

                # Encode features
                le_dict = {}
                df_model = df.copy()
                for col in feature_cols:
                    if df_model[col].dtype == 'object':
                        le = LabelEncoder()
                        df_model[col] = le.fit_transform(df_model[col].astype(str))
                        le_dict[col] = le

                # Encode targets
                le_target = {}
                for col in target_cols:
                    le = LabelEncoder()
                    df_model[col] = le.fit_transform(df_model[col].astype(str))
                    le_target[col] = le

                # Train model
                X = df_model[feature_cols].fillna(0)
                y = df_model[target_cols]
                model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
                model.fit(X, y)
                st.success("✅ Model trained successfully!")

                # Save to session
                st.session_state['model'] = model
                st.session_state['df'] = df
                st.session_state['feature_cols'] = feature_cols
                st.session_state['le_dict'] = le_dict
                st.session_state['le_target'] = le_target
                st.session_state['target_cols'] = target_cols

# --- Prediction Section ---
elif page == "Predict Tumor":
    st.subheader("Step 2: Predict Tumor by Patient ID")
    st.image("https://upload.wikimedia.org/wikipedia/commons/2/2b/Brain_MRI.jpg", use_container_width=True)
    patient_id_input = st.text_input("Enter Patient ID")

    if st.button("Predict") and 'model' in st.session_state:
        df = st.session_state['df']
        if patient_id_input not in df['Patient_ID'].values:
            st.warning("Patient ID not found in dataset!")
        else:
            # Fetch patient details
            patient_row = df[df['Patient_ID'] == patient_id_input].iloc[0]
            st.write("**Patient Details:**")
            st.write(patient_row)

            # Prepare input
            input_data = {col: patient_row[col] for col in st.session_state['feature_cols']}
            input_df = pd.DataFrame([input_data])

            # Encode input
            for col, le in st.session_state['le_dict'].items():
                input_df[col] = le.transform(input_df[col].astype(str))

            # Predict
            model = st.session_state['model']
            preds = model.predict(input_df)[0]
            probs_list = model.predict_proba(input_df)

            # Display results side by side
            st.subheader("Predicted Tumor Details")
            cols = st.columns(3)
            for i, col_name in enumerate(st.session_state['target_cols']):
                value = st.session_state['le_target'][col_name].inverse_transform([preds[i]])[0]
                cols[i].success(f"**{col_name.replace('_',' ')}:** {value}")

                # Probability chart
                labels = st.session_state['le_target'][col_name].classes_
                probs = probs_list[i][0]*100
                colors = ['green' if lbl==value else 'skyblue' for lbl in labels]
                fig, ax = plt.subplots(figsize=(3,2))
                ax.bar(labels, probs, color=colors)
                ax.set_ylim([0,100])
                ax.set_ylabel("Probability %")
                ax.set_title(f"{col_name.replace('_',' ')} Probabilities", fontsize=10)
                for j, p in enumerate(probs):
                    ax.text(j, p+1, f"{p:.1f}%", ha='center', fontsize=8)
                cols[i].pyplot(fig)

            # Stage-specific advice
            if 'Tumor_Stage' in st.session_state['target_cols']:
                stage = st.session_state['le_target']['Tumor_Stage'].inverse_transform([preds[st.session_state['target_cols'].index('Tumor_Stage')]])[0]
                st.subheader("Stage-specific Advice")
                if stage in ["Stage I","Stage II"]:
                    st.write("- Early stage detected: Consult a doctor and follow treatment plan.")
                elif stage == "Stage III":
                    st.write("- Advanced stage: Immediate consultation and treatment required.")
                elif stage == "Stage IV":
                    st.write("- Critical stage: Urgent medical attention needed.")
                else:
                    st.write("- Follow standard medical advice and regular check-ups.")
                st.write("- Maintain healthy lifestyle, regular check-ups, and emotional support.")

# --- Advice & Tips Page ---
elif page == "Advice & Tips":
    st.subheader("General Brain Tumor Advice & Tips")
    st.image("https://upload.wikimedia.org/wikipedia/commons/1/1e/Doctor_advice.jpg", use_container_width=True)
    st.write("""
    1. Immediate Consultation: See a neurologist for persistent symptoms.  
    2. Lifestyle: Eat healthy, exercise regularly, avoid smoking/alcohol.  
    3. Treatment Compliance: Follow doctor's instructions.  
    4. Early Detection: Regular check-ups help detect tumors early.  
    5. Support System: Emotional support is crucial.  
    6. Information: Stay informed about your condition and treatment options.
    """)

import streamlit as st

# Set page config
st.set_page_config(
    page_title="Brain Tumor Diagnostic Tool",
    page_icon="🧠",
    layout="centered"
)

# --- Title ---
st.title("🧠 Brain Tumor Diagnostic Tool")
st.markdown("Enter patient details and symptoms to predict possible brain tumor type and get advice.")

# --- Sidebar for navigation ---
page = st.sidebar.selectbox("Go to", ["Home", "Predict Tumor", "Advice & Tips"])

# --- Home Page ---
if page == "Home":
    st.subheader("Welcome!")
    st.write("""
    This app helps to predict the type of brain tumor based on patient details and symptoms.
    It also provides guidance on what steps to take if a tumor is detected.
    """)
    st.image("https://images.unsplash.com/photo-1588776814546-6c0e93e923a2?auto=format&fit=crop&w=800&q=60", caption="Brain Health Matters")

# --- Predict Tumor Page ---
elif page == "Predict Tumor":
    st.subheader("Patient Details")

    # Input fields
    age = st.slider("Age", 1, 100)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    symptoms = st.multiselect(
        "Select Symptoms",
        ["Headache", "Nausea", "Dizziness", "Fatigue", "Vision Problem", "Weakness", "Vomiting", "Memory Loss", "Seizures"]
    )

    st.markdown("---")
    st.subheader("Prediction")

    # Predict button
    if st.button("Predict"):
        if len(symptoms) == 0:
            st.warning("Please select at least one symptom.")
        else:
            # Dummy prediction logic
            tumor_type = "Glioma"  # placeholder for ML model prediction
            probability = 87  # placeholder probability

            st.success(f"Tumor Type: **{tumor_type}**")
            st.info(f"Probability: **{probability}%**")

            # Advice section
            st.markdown("---")
            st.subheader("Next Steps & Advice")
            st.write("""
            - See a neurologist immediately  
            - Maintain a healthy lifestyle  
            - Follow prescribed treatment  
            - Regular check-ups  
            - Early diagnosis improves recovery chances
            """)
            st.image("https://images.unsplash.com/photo-1588776814546-6c0e93e923a2?auto=format&fit=crop&w=800&q=60", caption="Consult a Specialist")

# --- Advice & Tips Page ---
elif page == "Advice & Tips":
    st.subheader("Brain Tumor Advice & Tips")
    st.write("""
    1. **Immediate Consultation:** Always consult a neurologist if you notice symptoms like persistent headaches, seizures, or vision problems.  
    2. **Lifestyle:** Eat healthy, exercise regularly, and avoid smoking/alcohol.  
    3. **Treatment Compliance:** Follow doctor's prescriptions and attend all check-ups.  
    4. **Early Detection:** Regular health check-ups can help detect tumors early.  
    5. **Support System:** Emotional support from family and friends is crucial.  
    6. **Information:** Stay informed about your condition and treatment options.
    """)
    st.image("https://images.unsplash.com/photo-1588776814546-6c0e93e923a2?auto=format&fit=crop&w=800&q=60", caption="Stay Healthy, Stay Informed")

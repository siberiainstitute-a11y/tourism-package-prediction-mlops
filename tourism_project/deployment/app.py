"""Streamlit front end for the Wellness Tourism Package purchase predictor."""
import json
import os

import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

# The model repo lives under the same Hugging Face account as this Space.
# MODEL_REPO_ID is set as a Space variable by hosting.py; SPACE_AUTHOR_NAME is
# injected by Hugging Face automatically and is used as a fallback.
MODEL_REPO_ID = os.getenv(
    "MODEL_REPO_ID",
    f"{os.getenv('SPACE_AUTHOR_NAME', 'your-hf-username')}/tourism-package-prediction-model",
)

st.set_page_config(page_title="Wellness Package Predictor", page_icon="🧳", layout="centered")


@st.cache_resource(show_spinner="Loading the latest model from the Hugging Face model hub...")
def load_model():
    """Download the registered pipeline and its metadata (cached per container)."""
    model_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename="tourism_model.joblib")
    meta_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename="model_metadata.json")
    with open(meta_path) as fh:
        return joblib.load(model_path), json.load(fh)


model, metadata = load_model()
threshold = metadata["decision_threshold"]

st.title("🧳 Wellness Tourism Package - Purchase Predictor")
st.write(
    "Enter what is known about a customer **before contacting them**. The model "
    "estimates how likely they are to buy the Wellness Tourism Package."
)

# ------------------------------------------------------------------ input form
with st.form("customer_form"):
    st.subheader("Customer profile")
    left, right = st.columns(2)
    with left:
        age = st.number_input("Age", min_value=18, max_value=90, value=36)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital status", ["Married", "Single", "Divorced", "Unmarried"])
        occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
        designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
        monthly_income = st.number_input("Gross monthly income", min_value=0, max_value=200000, value=22000, step=500)
    with right:
        city_tier = st.selectbox("City tier", [1, 2, 3])
        passport = st.radio("Holds a valid passport?", ["Yes", "No"], horizontal=True)
        own_car = st.radio("Owns a car?", ["Yes", "No"], horizontal=True)
        n_trips = st.number_input("Trips per year", min_value=0, max_value=25, value=3)
        n_persons = st.number_input("People travelling with the customer", min_value=1, max_value=5, value=3)
        n_children = st.number_input("Children under 5 travelling", min_value=0, max_value=3, value=1)
        property_star = st.selectbox("Preferred hotel rating (stars)", [3, 4, 5])

    st.subheader("Sales interaction")
    left, right = st.columns(2)
    with left:
        type_of_contact = st.selectbox("How was the customer reached?", ["Self Enquiry", "Company Invited"])
        product_pitched = st.selectbox("Product pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
        pitch_duration = st.number_input("Pitch duration (minutes)", min_value=1, max_value=130, value=14)
    with right:
        n_followups = st.slider("Number of follow-ups", 1, 6, 4)
        pitch_score = st.slider("Pitch satisfaction score", 1, 5, 3)

    submitted = st.form_submit_button("Predict purchase likelihood", type="primary")

# ------------------------------------------------------------------ prediction
if submitted:
    # Column names and category labels mirror the training data exactly.
    customer = pd.DataFrame([{
        "Age": age,
        "TypeofContact": type_of_contact,
        "CityTier": city_tier,
        "DurationOfPitch": pitch_duration,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": n_persons,
        "NumberOfFollowups": n_followups,
        "ProductPitched": product_pitched,
        "PreferredPropertyStar": property_star,
        "MaritalStatus": marital_status,
        "NumberOfTrips": n_trips,
        "Passport": int(passport == "Yes"),
        "PitchSatisfactionScore": pitch_score,
        "OwnCar": int(own_car == "Yes"),
        "NumberOfChildrenVisiting": n_children,
        "Designation": designation,
        "MonthlyIncome": monthly_income,
    }])

    probability = float(model.predict_proba(customer)[0, 1])

    st.subheader("Result")
    st.metric("Estimated purchase probability", f"{probability:.1%}")
    st.progress(min(max(probability, 0.0), 1.0))
    if probability >= threshold:
        st.success("Likely buyer - prioritise this customer for the Wellness Package campaign.")
    else:
        st.info("Unlikely buyer - keep in the low-touch nurture list.")
    st.caption(f"Decision threshold: {threshold:.2f} (selected during training to maximise F1).")

    with st.expander("Input sent to the model"):
        # values are shown as text because one column cannot mix numbers and labels
        st.dataframe(customer.T.astype(str).rename(columns={0: "value"}))

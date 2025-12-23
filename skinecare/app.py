import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ---------- Load files ----------
model = joblib.load("skinecare/logistic_model.pkl")
vectorizer = joblib.load("skinecare/tfidf_vectorizer.pkl")
df = pd.read_csv("final_merged.csv")

# Create text column if not exists
if "text" not in df.columns:
    df["text"] = df["afterUse"] + " " + df["name"]

# ---------- Page Config ----------
st.set_page_config(page_title="Skincare NLP System")
st.title("🧴 Skincare NLP System")

# ---------- Concerns Dictionary ----------
concerns = {
    "acne": ["acne", "pimple", "breakout"],
    "redness": ["redness", "red", "irritation", "sensitive"],
    "dryness": ["dry", "hydrating", "moisturizing", "dehydrated"],
    "pigmentation": ["dark spot", "pigmentation", "brightening"]
}

# ---------- NLP: Extract skin type & concern ----------
def extract_user_info(user_text):
    text = user_text.lower()

    skin_types = {
        "dry": "Dry",
        "oily": "Oily",
        "combination": "Combination",
        "sensitive": "Sensitive",
        "normal": "Normal"
    }

    detected_skin = "Normal"
    detected_concern = None

    for k, v in skin_types.items():
        if k in text:
            detected_skin = v

    for concern, keywords in concerns.items():
        for kw in keywords:
            if kw in text:
                detected_concern = concern
                break

    return detected_skin, detected_concern

# ---------- NLP-based Skin Problem Recommendation ----------
def recommend_solution(user_text, df, vectorizer, top_n=5):

    skin_type, concern = extract_user_info(user_text)

    filtered_df = df[df[skin_type] == 1.0]

    if concern and concern in concerns:
        pattern = "|".join(concerns[concern])
        filtered_df = filtered_df[
            filtered_df['afterUse'].str.contains(pattern, case=False, na=False)
        ]

    if len(filtered_df) < 5:
        filtered_df = df[df[skin_type] == 1.0]

    X_products = vectorizer.transform(filtered_df["text"])
    X_user = vectorizer.transform([user_text])

    scores = (X_products @ X_user.T).toarray().ravel()
    filtered_df = filtered_df.copy()
    filtered_df["score"] = scores

    top_products = filtered_df.sort_values(
        "score", ascending=False
    ).head(top_n)

    return skin_type, concern, top_products[["brand", "name", "type"]]

# ---------- Simple Skin Routine ----------
def simple_skin_routine(skin_type, top_n=2):
    skin_type = skin_type.capitalize()

    if skin_type not in df.columns:
        return "Skin type not found in data"

    filtered_df = df[df[skin_type] == 1.0]

    if filtered_df.empty:
        return "No products found for this skin type"

    routine_order = ['cleanser', 'toner', 'serum', 'moisturizer']
    routine = {}

    for r_type in routine_order:
        products = filtered_df[
            filtered_df['type'].str.contains(r_type, case=False, na=False)
        ]
        if not products.empty:
            routine[r_type] = products[['brand', 'type']].head(top_n)

    return routine

# ---------- UI Choice ----------
choice = st.radio(
    "What do you want to do?",
    ["🔍 Fix skin problem (NLP)", "🧴 Get skin care routine"]
)

# ---------- NLP Skin Problem ----------
if choice == "🔍 Fix skin problem (NLP)":
    st.subheader("🔍 Describe your skin problem")

    text = st.text_area(
        "Example: I have dry skin and redness and irritation"
    )

    if st.button("Get Products"):
        if text.strip() == "":
            st.warning("Please enter your skin problem")
        else:
            skin, concern, results = recommend_solution(
                text, df, vectorizer
            )

            st.success(f"Detected Skin Type: {skin}")
            st.info(f"Detected Concern: {concern}")

            if results.empty:
                st.warning("No suitable products found")
            else:
                st.subheader("🧴 Recommended Products")
                for _, row in results.iterrows():
                    st.write(
                        f"- **{row['brand']}** | {row['name']} ({row['type']})"
                    )

# ---------- Routine Section ----------
if choice == "🧴 Get skin care routine":
    st.subheader("🧴 Skin Care Routine")

    skin = st.selectbox(
        "Choose your skin type",
        ["Oily", "Dry", "Normal", "Combination", "Sensitive"]
    )

    if st.button("Show Routine"):
        routine = simple_skin_routine(skin)

        if isinstance(routine, str):
            st.warning(routine)
        else:
            st.success(f"Recommended Routine for {skin} Skin")
            for r_type, items in routine.items():
                st.markdown(f"### {r_type.upper()}")
                for _, row in items.iterrows():
                    st.write(f"- {row['brand']} ({row['type']})")

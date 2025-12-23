import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ---------- Load files ----------
model = joblib.load("skinecare/logistic_model.pkl")
vectorizer = joblib.load("skinecare/tfidf_vectorizer.pkl")
df = pd.read_csv("final_merged.csv")

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

# ---------- Recommendation Function ----------
def recommend_solution(user_text, df, vectorizer, top_n=5):

    skin_type, concern = extract_user_info(user_text)

    # Filter by skin type
    filtered_df = df[df[skin_type] == 1.0]

    # Filter by concern keywords
    if concern and concern in concerns:
        pattern = "|".join(concerns[concern])
        filtered_df = filtered_df[
            filtered_df['afterUse'].str.contains(pattern, case=False, na=False)
        ]

    # Fallback if results are few
    if len(filtered_df) < 5:
        filtered_df = df[df[skin_type] == 1.0]

    # NLP similarity
    X_products = vectorizer.transform(
        filtered_df['afterUse'] + " " + filtered_df['name']
    )
    X_user = vectorizer.transform([user_text])

    scores = (X_products @ X_user.T).toarray().ravel()
    filtered_df = filtered_df.copy()
    filtered_df['score'] = scores

    top_products = filtered_df.sort_values(
        'score', ascending=False
    ).head(top_n)

    return skin_type, concern, top_products[['brand', 'name', 'type']]

# ---------- UI ----------
st.subheader("🔍 Describe your skin problem")

user_text = st.text_area(
    "Example: I have dry skin and redness and irritation"
)

if st.button("Get Recommendations"):
    if user_text.strip() == "":
        st.warning("Please enter your skin problem")
    else:
        skin, concern, results = recommend_solution(
            user_text, df, vectorizer
        )

        st.success(f"Detected Skin Type: {skin}")
        st.info(f"Detected Concern: {concern}")

        if results.empty:
            st.warning("No suitable products found")
        else:
            st.subheader("🧴 Recommended Products")
            for _, row in results.iterrows():
                st.write(f"- **{row['brand']}** | {row['name']} ({row['type']})")

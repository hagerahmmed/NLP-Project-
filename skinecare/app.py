import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ---------- Load files ----------
model = joblib.load("skinecare/logistic_model.pkl")
vectorizer = joblib.load("skinecare/tfidf_vectorizer.pkl")
df = pd.read_csv("final_merged.csv")

st.set_page_config(page_title="Skincare NLP System")
st.title("🧴 Skincare NLP System")

# --------- FUNCTION: NLP routine based on skin problem ----------
def nlp_skin_problem_routine(text, model, vectorizer, top_n=2):

    # 1) Vectorize input text
    vec = vectorizer.transform([text])

    # 2) Predict product type for the problem (cleanser/toner/serum/moisturizer)
    pred_type = model.predict(vec)[0]

    # 3) Filter products from dataset that match predicted type
    filtered = df[df["type"].str.contains(pred_type, case=False, na=False)]

    if filtered.empty:
        return {}

    # 4) Score products using NLP model again
    X_filtered = vectorizer.transform(filtered["text"])

    try:
        scores = model.decision_function(X_filtered)
    except:
        scores = model.predict_proba(X_filtered)
        scores = np.max(scores, axis=1)

    filtered["score"] = scores

    # 5) Sort and select top N
    filtered = filtered.sort_values("score", ascending=False)

    # Return top products
    return {
        pred_type: filtered[["brand", "name", "type"]].head(top_n).to_dict("records")
    }


# ---------- FUNCTION: Skin routine based on skin type ----------
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
    ["🔍 Fix skin problem (NLP)", "Get skin care routine"]
)

# ---------- NLP Skin Problem ----------
if choice == "🔍 Fix skin problem (NLP)":
    st.subheader("🔍 Enter your skin problem")

    text = st.text_area(
        "Describe your skin problem",
        placeholder="Example: I have acne and redness around my cheeks"
    )

    if st.button("Get Products"):
        if text.strip() == "":
            st.warning("Please enter your skin problem")
        else:
            results = nlp_skin_problem_routine(text, model, vectorizer)

            if not results:
                st.warning("No matching products found")
            else:
                st.success("Products recommended for your skin problem:")
                for step, prods in results.items():
                    st.markdown(f"### {step.upper()}")
                    for p in prods:
                        st.write(f"- {p['brand']} | {p['name']}")


# ---------- Routine Section ----------
if choice == "Get skin care routine":
    st.subheader("Skin Care Routine")

    skin = st.selectbox(
        "Choose your skin type",
        ["Oily", "Dry", "Normal", "Combination", "Sensitive"]
    )

    if st.button("Show Routine"):
        routine = simple_skin_routine(skin)

        if isinstance(routine, str):
            st.warning(routine)
        else:
            st.success(f" Recommended Routine for {skin} Skin")
            for r_type, items in routine.items():
                st.markdown(f"### {r_type.upper()}")
                for _, row in items.iterrows():
                    st.write(f"- {row['brand']} ({row['type']})")


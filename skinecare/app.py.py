import streamlit as st
import pandas as pd
import joblib

import streamlit as st

st.markdown("""
<style>
.stApp {
    background-color: #ffffff;
}

h1 {
    color: #f472b6;
    font-weight: 600;
}

textarea, input {
    border-radius: 12px !important;
    border: 1px solid #fbcfe8 !important;
}

[data-testid="stChatMessage"] {
    border-radius: 18px;
    padding: 14px;
    margin-bottom: 12px;
    max-width: 80%;
}

[data-testid="stChatMessage"][data-role="user"] {
    background-color: #f472b6;
    color: white;
    margin-left: auto;
}

[data-testid="stChatMessage"][data-role="assistant"] {
    background-color: #fce7f3;
    color: #831843;
    margin-right: auto;
}
</style>
""", unsafe_allow_html=True)

st.title("🌸 Skincare Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "أهلاً 💗 أنا مساعدك لاختيار منتجات العناية بالبشرة. قولي نوع بشرتك وإيه المشاكل اللي بتواجهيها 🌷"
        }
    ]



for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


        if user_input:
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.write(user_input)

    bot_reply = """
🌸 شكراً لمشاركتك!

بناءً على كلامك، أنصحك بـ:
- Cleanser لطيف
- Serum مناسب لنوع بشرتك
- Moisturizer خفيف
- Sunscreen صباحاً ☀️

لو حابة، قولي:
- ميزانيتك
- أو تحبي روتين صباح/ليل 💗
"""

    st.session_state.messages.append({
        "role": "assistant",
        "content": bot_reply
    })

    with st.chat_message("assistant"):
        st.write(bot_reply)






















# ---------- Load files ----------
model = joblib.load("skincare/svm_model.pkl")
vectorizer = joblib.load("skin_care/tfidf_vectorizer (2).pkl")
df = pd.read_csv("skincare/final_merged (2) (1).csv")

st.set_page_config(page_title="Skincare NLP System")
st.title(" Skincare NLP System")

# ---------- Helper Function ----------
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
    ["🔍 Predict product type", "Get skin care routine"]
)

# ---------- Predict Section ----------
if choice == "🔍 Predict product type":
    st.subheader("Predict Product Type")

    text = st.text_area(
        "Enter product description (after use / name / brand)",
        placeholder="Example: reduces acne and redness, Acne Serum, COSRX"
    )

    if st.button("Predict"):
        if text.strip() == "":
            st.warning("Please enter product description")
        else:
            vec = vectorizer.transform([text])
            pred = model.predict(vec)[0]
            st.success(f"Predicted Type: {pred}")

# ---------- Routine Section ----------
if choice == " Get skin care routine":
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

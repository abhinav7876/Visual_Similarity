import streamlit as st
import requests
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="Floor Similarity Search", layout="wide")
st.title("Floor Similarity Search App")

uploaded_file = st.file_uploader("Upload a room image:", type=["jpg", "png", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="Query Image")

    response = requests.post("http://127.0.0.1:8000/search", files={"file": uploaded_file})
    
    if response.status_code == 200:
        results = response.json()
        st.subheader("Top Matching Floor Products")

        cols = st.columns(5)
        for i, match in enumerate(results):
            img_path = f"data/Assess/sku/{match['name']}"
            img = Image.open(img_path)
            with cols[i % 5]:
                st.image(img, caption=f"{match['name']}\n Score: {match['score']}")

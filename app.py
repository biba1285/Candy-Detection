import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO


st.set_page_config(page_title="Candy 🍬")


st.header("Candy 🍬")


@st.cache_resource
def load_model():
    model = YOLO("best (1).pt")   
    return model

model = load_model()


uploaded_file = st.file_uploader("Upload a candy image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    
    results = model(img_array)

    
    results_img = results[0].plot()  

    
    st.image(results_img, caption="Detected Candy", use_column_width=True)

    
    detected = results[0].boxes  

    if len(detected) == 0:
        st.write("No candy detected.")
    else:
        st.write("Detected candy:")
        for box in detected:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            name = results[0].names[cls_id]
            st.write(f"- {name} ({conf:.2f} confidence)")



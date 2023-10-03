import streamlit as st
import numpy as np
import cv2
from keras.models import load_model


def get_data(data):
    img_size = 150
    resized_arr = cv2.resize(data, (img_size, img_size))
    return np.array(resized_arr)


def process(data):
    img_size = 150
    pr_data = data.reshape(-1, img_size, img_size, 1)
    return pr_data


st.set_page_config(
        page_title="My Page Title",
        layout="wide",
)


st.header("🩺 Auto-pneumonia detection app", divider="blue")


with st.sidebar:
    st.image("images/pneumonia.png", width=140)
    st.title("Pneumonia detection")
    choice = st.radio("Navigation",["Main page", "Training model info", "About"])


if choice == "Main page":
    st.text("You can upload the X-ray here to have a prediction of the pneumonia")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        d = get_data(opencv_image)
        d1 = process(d)
        new_model = load_model('xraymodel.h5')
        predictions = new_model.predict(d1)
        predictions = predictions.reshape(1,-1)[0]
        avg = np.mean(predictions)
        if np.min(predictions) == 0:
            st.error('Pneumonia test result: positive', icon="❌")
        else:
            st.success('Pneumonia test result: negative', icon="✅")
        view_button = st.button("Click here to see your image!")
        if view_button:
            st.image(opencv_image, channels="BGR", width=None)
elif choice == "Training model info":
    st.subheader("Here's the info about the model")
elif choice == "About":
    st.subheader("Developer info")
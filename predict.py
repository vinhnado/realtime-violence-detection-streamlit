from datetime import datetime
import cv2
from tensorflow.keras.models import load_model  # type: ignore
import numpy as np
import os
import streamlit as st
from discord_webhook import sendMsg
import os
from dotenv import load_dotenv

# Globals
WEBHOOK_SERVER_URL = 'https://discord.com/api/webhooks/1243503941120032862/GiOg3QSN0OsqB2cyzW8MIKZE1Me2IgJ_vsrdX0a-rzVPXRVAxWpZ2p2Idq_B8rY9aC7z'
WEBHOOK_ACCESS_TOKEN = 'GiOg3QSN0OsqB2cyzW8MIKZE1Me2IgJ_vsrdX0a-rzVPXRVAxWpZ2p2Idq_B8rY9aC7z'

IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 16
CLASSES_LIST = ["NonViolence", "Violence"]

# Load your trained model
model = load_model("models\BILSTM_RESNET_MODEL.h5")

def change_model(model_name):
    global model
    if model_name == "BILSTM_RESNET_MODEL":
        model = load_model("models\BILSTM_RESNET_MODEL.h5")
        print("Model changed to BILSTM_RESNET_MODEL")
    elif model_name == "BILSTM_VGG16T_MODEL":
        model = load_model("models\BILSTM_VGG16T_MODEL.h5")
        print("Model changed to BILSTM_VGG16T_MODEL")
    elif model_name == "MoBiLSTM_model":
        model = load_model("models\MoBiLSTM_model.h5")
        print("Model changed to MoBiLSTM_model")
    
def process_frames(
    frames_list
):
    predicted_labels_probabilities = model.predict(np.expand_dims(frames_list, axis=0))[0]
    predicted_label = np.argmax(predicted_labels_probabilities)
    predicted_class_name = CLASSES_LIST[predicted_label]
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"Time {current_time}: Predicted: {predicted_class_name}, Confidence: {predicted_labels_probabilities[predicted_label]:.4f}"
    )

    if predicted_class_name == CLASSES_LIST[1]:
        sendMsg(WEBHOOK_SERVER_URL, WEBHOOK_ACCESS_TOKEN, CLASSES_LIST[1] + " Detected")
    if predicted_class_name == "Violence":
        color = "red"
    else:
        color = "green"

    st.write(
        f"Time {current_time}: Predicted: <span style='color:{color};'>{predicted_class_name}</span>, Confidence: {predicted_labels_probabilities[predicted_label]:.4f}",
        unsafe_allow_html=True
    )

def predict_frames_from_folder(frames_folder_path):
    frame_paths = [
        os.path.join(frames_folder_path, f)
        for f in os.listdir(frames_folder_path)
        if f.endswith((".png", ".jpg", ".jpeg"))
    ]
    frame_paths.sort(reverse=True)

    if not frame_paths:
        st.write("No frames available for prediction.")
        return

    frames_list = []

    with st.sidebar:
        for frame_path in frame_paths:
            frame = cv2.imread(frame_path)
            resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
            normalized_frame = resized_frame / 255.0
            frames_list.append(normalized_frame)

            if len(frames_list) == SEQUENCE_LENGTH:
                process_frames(frames_list)
                frames_list = [] 

    if len(frames_list) > 0:
        process_frames(frames_list)
        

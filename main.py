import cv2
import streamlit as st
import os
from datetime import datetime 
from predict import change_model, predict_frames_from_folder
import shutil

def save_frame(frame):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = os.path.join("frames", f"frame_{timestamp}.jpg")
    cv2.imwrite(filename, frame)

if not os.path.exists("frames"):
    try:
        os.makedirs("frames")
    except FileExistsError:
        pass  

if "capture" not in st.session_state:
    st.session_state.capture = False
if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0  

prediction_text = False

st.set_page_config(
    page_title="Realtime Violence Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Realtime Violence Detection 🛡️")

if not st.session_state.capture:
    model_selection = st.selectbox(
        "Select Model",
        ("BILSTM_RESNET_MODEL", "BILSTM_VGG16T_MODEL", "MoBiLSTM_model"),
        index=0,
    )
else:
    st.text(f"Selected Model: {st.session_state.model_selection}")
    model_selection = st.session_state.model_selection

if not st.session_state.capture:
    st.session_state.model_selection = model_selection
    
MODEL_PATH = model_selection
if model_selection == "BILSTM_RESNET_MODEL":
    change_model("BILSTM_RESNET_MODEL")
elif model_selection == "BILSTM_VGG16T_MODEL":
    change_model("BILSTM_VGG16T_MODEL")
elif model_selection == "MoBiLSTM_model":
    change_model("MoBiLSTM_model")


message_placeholder = st.empty()
video_container = st.empty()

col1, col2, col3 = st.columns(
    3
)


with st.sidebar:
    st.title("Realtime Violence Detection")


with col1:
    if st.button("Start"):
        st.session_state.capture = True
        st.session_state.frame_count = 0 
        message_placeholder.empty()
        if not hasattr(st.session_state, "cap") or not st.session_state.cap.isOpened():
            st.session_state.cap = cv2.VideoCapture(1)
        prediction_placeholder = st.empty()

with col2:
    if st.session_state.capture:
        if st.button("Stop"):
            st.session_state.capture = False
            if hasattr(st.session_state, "cap"):
                st.session_state.cap.release()
            message_placeholder.success("Video capturing has been stopped.")

dir_path = "frames"
with col3:
    if os.path.exists(dir_path):
        if not st.session_state.capture:
            if st.button("Delete Directory"):
                try:
                    shutil.rmtree(dir_path)
                    st.success(f'All "{dir_path}" has been deleted successfully.')
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.text("Delete (Unavailable while Rec🔴)")
    else:
        st.write(f'Directory "{dir_path}" does not exist.')


if st.session_state.capture:
    while st.session_state.capture:
        ret, frame = st.session_state.cap.read()
        if not ret:
            message_placeholder.error("Failed to capture video.")
            st.session_state.capture = False
            if hasattr(st.session_state, "cap"):
                st.session_state.cap.release()
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_container.image(frame_rgb, channels="RGB", use_column_width=True)

        save_frame(frame)
        st.session_state.frame_count += 1

        if st.session_state.frame_count % 100 == 0:
            frames_folder_path = "frames"
            prediction = predict_frames_from_folder(frames_folder_path)
            # prediction_text = prediction
            # prediction_placeholder.markdown("### Prediction Result")
            # print("For Debugging",prediction_text)
            # prediction_placeholder.write(prediction)
            # st.session_state.last_prediction = prediction


with st.sidebar:
    if prediction_text:
        st.markdown("### Last Prediction Result")
        st.markdown(":blue["+prediction_text+"]")
        
    else:
        st.markdown("### Last Prediction Result")
        st.markdown(":green[No Prediction Right Now]")
import cv2
import altair as alt
import numpy as np
import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import tempfile
import os
import subprocess

hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
hub_module = hub.load(hub_handle)

st.set_option('deprecation.showPyplotGlobalUse', False)
# @functools.lru_cache(maxsize=None)
def load_image(uploaded_file, image_size=(384, 384),col=st):
    img = Image.open(uploaded_file)
    img = tf.convert_to_tensor(img)
    img = tf.image.resize(img, image_size)
    img = tf.reshape(img, [-1, image_size[0], image_size[1], 3]) / 255
    col.image(np.array(img[0]))
    return img

def preprocess_frame(frame):
    """Preprocesses a video frame for style transfer."""
    frame = tf.image.resize(frame, (img_width, img_height))
    frame = tf.reshape(frame, [-1, img_width, img_height, 3]) / 255
    return frame

def load_video(uploaded_file,frame_rate=1,col=st):
    """Loads and preprocesses a video into frames."""
    col.video(uploaded_file)
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap=cv2.VideoCapture(tfile.name)
    frames =[]
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(preprocess_frame(frame))
        
    cap.release()
    return frames

def combine_stylized_frames(stylized_frames, output_filename="stylized2_video.mp4"):
    """Combines stylized frames into a video."""

    height,width=384,384
    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*"AVC1") 
    video_writer = cv2.VideoWriter(output_filename, fourcc, 2, (width,height))
    for frame in stylized_frames:
        # Convert frame back to OpenCV format (if needed)
        frame = np.uint8(frame * 255.0)  # Assuming frame is between 0 and 1
        frame= np.concatenate(frame, axis=0)
        video_writer.write(frame)

    # Release video writer
    video_writer.release()
    st.success(f"Stylized video saved as: {output_filename}")
    return output_filename

def show_stylized_frames(stylized_frames):
  """Displays the generated stylized frames in Streamlit."""
  num_cols = 3  # Adjust the number of columns for displaying frames

  # Split frames into a grid for better layout
  for i in range(0, len(stylized_frames), num_cols):
    cols = st.columns(num_cols)
    for j, col in enumerate(cols):
      if i + j < len(stylized_frames):
        col.image(np.array(stylized_frames[i + j][0]))

## Basic setup and app layout
st.set_page_config(layout="wide")
st.title("Video Style Transfer")
alt.renderers.set_embed_options(scaleFactor=2)
hide_streamlit_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.sidebar.header('Go To')

if st.sidebar.button('Home'):
    subprocess.Popen(['streamlit', 'run', 'start.py'])

# Video button
if st.sidebar.button('image'):
    subprocess.Popen(['streamlit', 'run', 'image.py'])


if __name__ == "__main__":
    img_width, img_height = 384, 384
    img_width_style, img_height_style = 384, 384
    col1, col2 = st.columns(2)
    col1.markdown('# Add video on which style is required')
    uploaded_file = col1.file_uploader("Upload Video")
    if uploaded_file is not None:
        video_frames = load_video(uploaded_file,col=col1)
        # st.write(len(video_frames))

    col2.markdown('# Add image from which style will extracted')
    uploaded_file_style = col2.file_uploader("Upload Style Image")
    if uploaded_file_style is not None:
        style_image = load_image(uploaded_file_style, (img_width_style, img_height_style), col=col2)
        style_image = tf.nn.avg_pool(style_image, ksize=[3, 3], strides=[1, 1], padding='SAME')
        col3, col4, col5 = st.columns(3)
        

        # if video_frames and style_image:
        if col3.button("Apply Style to get video"):
            stylized_frames = []
            for frame in video_frames:
                outputs = hub_module(tf.constant(frame), tf.constant(style_image))
                stylized_frames.append(outputs[0])
            video_path=combine_stylized_frames(stylized_frames)
            col4.video(video_path)
           
        if col4.button("Apply style to get stylized images of a video"):
            stylized_frames = []
            for frame in video_frames:
                outputs = hub_module(tf.constant(frame), tf.constant(style_image))
                stylized_frames.append(outputs[0])
            show_stylized_frames(stylized_frames)
            



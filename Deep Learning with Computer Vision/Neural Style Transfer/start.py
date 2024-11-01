import streamlit as st
import subprocess

# Set page configuration to wide layout
st.set_page_config(layout="wide")

# Streamlit interface
st.title('Neural Style Transfer')

# Definition of Neural Style Transfer
st.markdown("""
<font size="+5">Neural Style Transfer (NST) is an algorithm that combines the content of one image with the style of another image
while retaining the content's features. It works by optimizing the input image to minimize both its content distance
with the content image and its style distance with the style image.

NST has been widely used in art generation and image processing. It allows artists and designers to apply various
artistic styles to their images, creating visually appealing artworks.</font>
""", unsafe_allow_html=True)

# Two-column layout for buttons
col1, col2 = st.columns(2)

# Button for Image Style Transfer
with col1:
    st.markdown('# Click here for Image Style Transfer')
    if st.button('Image Style Transfer'):
        # Navigate to another page for Image Style Transfer
        subprocess.Popen(['streamlit', 'run', 'image.py'])

# Button for Video Style Transfer
with col2:
    st.markdown('# Click here for Video Style Transfer')
    if st.button('Video Style Transfer'):
        subprocess.Popen(['streamlit', 'run', 'video.py'])


import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.applications import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing import image
import altair as alt
import subprocess

st.set_option('deprecation.showPyplotGlobalUse', False)

def crop_center(image):
  """Returns a cropped square image."""
  shape = image.shape
  new_shape = min(shape[0], shape[1])
  offset_y = max(shape[0] - shape[1], 0) // 2
  offset_x = max(shape[1] - shape[0], 0) // 2
  image = tf.image.crop_to_bounding_box(
      image, offset_y, offset_x, new_shape, new_shape)
  return image

# @functools.lru_cache(maxsize=None)
def load_image(uploaded_file, image_size=(256, 256), col = st):
  img = Image.open(uploaded_file)
  img = tf.convert_to_tensor(img)
  img = crop_center(img)
  img = tf.image.resize(img, image_size)
  if img.shape[-1] == 4:
        img = img[:, :, :3]
  img = tf.reshape(img, [-1, image_size[0], image_size[1], 3])/255
  col.image(np.array(img[0]))
  return img

#Function to preprocess the image
def preprocess_image(img):
    img = img.convert('RGB')
    img = img.resize((img_width, img_height))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Function to deprocess the image
def deprocess_image(x):
    x = x.reshape((img_height, img_width, 3))
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# Function to compute content loss
def content_loss(base, combination):
    return tf.reduce_sum(tf.square(combination - base))

# Function to compute style loss
def style_loss(style, combination):
    style_shape = tf.shape(style)
    combination_shape = tf.shape(combination)
    style_features = tf.reshape(style, (style_shape[0], -1))
    combination_features = tf.reshape(combination, (combination_shape[0], -1))
    style_matrix = tf.matmul(style_features, style_features, transpose_b=True)
    combination_matrix = tf.matmul(combination_features, combination_features, transpose_b=True)
    channels = style_shape[0]
    size = style_shape[1] * style_shape[2]
    return tf.reduce_sum(tf.square(style_matrix - combination_matrix)) / tf.cast(4 * (channels ** 2) * (size ** 2), tf.float32)



# Function to compute total variation loss for regularization
def total_variation_loss(x):
    a = tf.square(x[:, :img_height-1, :img_width-1, :] - x[:, 1:, :img_width-1, :])
    b = tf.square(x[:, :img_height-1, :img_width-1, :] - x[:, :img_height-1, 1:, :])
    return tf.reduce_sum(tf.pow(a + b, 1.25))

# Function to define the loss
def compute_loss(combination_image, base_image, style_reference_image, content_weight, style_weight, total_variation_weight):
    input_tensor = tf.concat([base_image, style_reference_image, combination_image], axis=0)
    features = feature_extractor(input_tensor)
    loss = tf.zeros(shape=())
    layer_features = features['block5_conv2']
    base_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    loss = loss + content_weight * content_loss(base_image_features, combination_features)

    feature_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    for layer_name in feature_layers:
        layer_features = features[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = style_loss(style_reference_features, combination_features)
        loss += (style_weight / len(feature_layers)) * sl
    loss += total_variation_weight * total_variation_loss(combination_image)
    return loss

# Function to compute the gradients of the loss with respect to the combination image
@tf.function
def compute_grads(combination_image, base_image, style_reference_image, content_weight, style_weight, total_variation_weight):
    with tf.GradientTape() as tape:
        loss = compute_loss(combination_image, base_image, style_reference_image, content_weight, style_weight, total_variation_weight)
    grads = tape.gradient(loss, combination_image)
    return grads


# Function to perform style transfer
def style_transfer(content_path, style_path, iterations=1000, content_weight=1e3, style_weight=1e-2, total_variation_weight=30):
    global img_height, img_width, feature_extractor
    
    content_image = Image.open(content_path)
    style_image = Image.open(style_path)
    img_height = 384
    img_width = int(content_image.width * img_height / content_image.height)

    base_image = preprocess_image(content_image)
    style_reference_image = preprocess_image(style_image)
    combination_image = tf.Variable(preprocess_image(content_image))

    # Load VGG19 model and set trainable to False
    vgg19 = VGG19(weights='imagenet', include_top=False)
    vgg19.trainable = False
    outputs_dict = dict([(layer.name, layer.output) for layer in vgg19.layers])
    feature_extractor = tf.keras.Model(inputs=vgg19.inputs, outputs=outputs_dict)
    optimizer = tf.keras.optimizers.Adam(learning_rate=5.0, beta_1=0.99, epsilon=1e-1)
    for i in range(iterations):
        grads = compute_grads(combination_image, base_image, style_reference_image, content_weight, style_weight, total_variation_weight)
        optimizer.apply_gradients([(grads, combination_image)])
        # if i % 100 == 0:
        #     with st.spinner(f'Iteration {i}'):
        #         st.write(f'Loss: {compute_loss(combination_image, base_image, style_reference_image, content_weight, style_weight, total_variation_weight)}')

    final_image = deprocess_image(combination_image.numpy())
    return final_image


def show_n(images, titles=('',), col = st):
  n = len(images)
  for i in range(n):
      col.image(np.array(images[i][0]))

# Streamlit app
# st.title('Neural Style Transfer')## Basic setup and app layout

# Set page configuration to wide layout

st.set_page_config(layout="wide")

alt.renderers.set_embed_options(scaleFactor=2)

hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.sidebar.header('Go To')

# Home button
if st.sidebar.button('Home'):
    subprocess.Popen(['streamlit', 'run', 'start.py'])

# Video button
if st.sidebar.button('Video'):
    subprocess.Popen(['streamlit', 'run', 'video.py'])

col1, col2 = st.columns(2)
col1.markdown('# Add image on which style is required')
col2.markdown('# Add image from which style will extracted')

uploaded_file = col1.file_uploader("Upload Content Image")
if uploaded_file is not None:
    content_image = load_image(uploaded_file, (384, 384), col = col1)
# style_image = col2.file_uploader('Upload Style Image', type=['jpg', 'jpeg', 'png'])
uploaded_file_style = col2.file_uploader("Upload Style Image")
if uploaded_file_style is not None:
    style_image = load_image(uploaded_file_style, (384, 384), col = col2)
iterations = 500
content_weight = 1e3
style_weight = 1e-2
total_variation_weight = 30
if st.button('Stylize'):
            stylized_image = style_transfer(content_image, style_image, iterations=iterations, content_weight=content_weight,
                                        style_weight=style_weight, total_variation_weight=total_variation_weight)
            col3, col4, col5 = st.columns(3)
            col4.markdown('# Style applied on the image')
            show_n([stylized_image], titles=['Stylized image'], col=col4)

    

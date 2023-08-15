import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Define content and style layers for VGG19/ResNet50
CONTENT_LAYERS_VGG = ['block3_conv4']
STYLE_LAYERS_VGG = ['block1_conv1', 'block2_conv2', 'block3_conv3', 'block4_conv4', 'block5_conv2']

CONTENT_LAYERS_RESNET = ['conv3_block4_out']
STYLE_LAYERS_RESNET = ['conv1_relu', 'conv2_block3_out', 'conv3_block4_out', 'conv4_block6_out', 'conv5_block3_out']

size = 360
# Predefined Hyperparameters
lr= 20
iterations = 200
style_wt = 0.001
content_wt = 0.9
style_layer_wts= [4,2,1,0.1,0.1]

def clip_and_normalize(image):
    # If image is of type float
    if image.dtype == np.float32 or image.dtype == np.float64:
        return np.clip(image, 0.0, 1.0)
    
    # If image is of type uint8
    elif image.dtype == np.uint8:
        return np.clip(image, 0, 255) 

    else:
        raise ValueError(f"Unexpected image data type: {image.dtype}")

def preprocess_image(pil_img, size=360):
    img = np.array(pil_img.resize((size, size)))
    img = np.expand_dims(img, axis=0)
    return img

# Define functions and classes used in the style transfer process
def gram_matrix(M):
    num_channels = tf.shape(M)[-1]
    M = tf.reshape(M, shape=(-1, num_channels))
    n = tf.shape(M)[0]
    G = tf.matmul(tf.transpose(M), M)
    return G

def content_cost(content_img, generated_img, model):
    C = model(content_img)
    G = model(generated_img)
    cost = tf.reduce_mean(tf.square(G - C))
    return cost

def style_cost(style_img, generated_img, model_list):
    total_cost = 0
    
    for i, style_model in enumerate(model_list):
        S = style_model(style_img)
        G = style_model(generated_img)
        GS = gram_matrix(S)
        GG = gram_matrix(G)
        current_cost = style_layer_wts[i] * tf.reduce_mean(tf.square(GS - GG))
        total_cost += current_cost

    return total_cost

def style_transfer(content_img, style_img, model_choice, content_layers, style_layers):
    if model_choice == "VGG19":
        model = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    elif model_choice == "ResNet50":
        model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet')

    content_model = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer(content_layers[0]).output)
    style_models = [tf.keras.models.Model(inputs=model.input, outputs=model.get_layer(style_layer).output) for style_layer in style_layers]

    generated_image = tf.Variable(preprocess_image(content_img), dtype=tf.float32)
    optimizer = tf.optimizers.Adam(learning_rate=lr)

    progress_bar = st.progress(0)  #

    for _ in range(iterations):
        with tf.GradientTape() as tape:
            J_content = content_cost(preprocess_image(content_img), generated_image, content_model)
            J_style = style_cost(preprocess_image(style_img), generated_image, style_models)
            J_total = content_wt * J_content + style_wt * J_style

        gradients = tape.gradient(J_total, generated_image)
        optimizer.apply_gradients([(gradients, generated_image)])

        progress_percentage = (_ + 1) / iterations
        progress_bar.progress(progress_percentage)

    return generated_image.numpy()[0]

def main():
    st.title("Neural Style Transfer")

    st.write('Please upload a content image')
    content_file = st.file_uploader("Choose a content image...", type=['jpg', 'png', 'jpeg'])
    if content_file:
        content_img = Image.open(content_file)
        st.image(content_img, caption='Uploaded Content Image.', use_column_width=True)

    st.write('Please upload a style image')
    style_file = st.file_uploader("Choose a style image...", type=['jpg', 'png', 'jpeg'])
    if style_file:
        style_img = Image.open(style_file)
        st.image(style_img, caption='Uploaded Style Image.', use_column_width=True)

    # Model choice
    model_choice = st.selectbox("Choose your base model", ("VGG19", "ResNet50"))

    if st.button("Transfer Style!"):
        if model_choice == "VGG19":
            content_layers = CONTENT_LAYERS_VGG
            style_layers = STYLE_LAYERS_VGG
        elif model_choice == "ResNet50":
            content_layers = CONTENT_LAYERS_RESNET
            style_layers = STYLE_LAYERS_RESNET

        result_img = style_transfer(content_img, style_img, model_choice, content_layers, style_layers)
        result_img = clip_and_normalize(result_img)
        st.image(result_img, caption="Styled Image", use_column_width=True)

if __name__ == "__main__":
    main()


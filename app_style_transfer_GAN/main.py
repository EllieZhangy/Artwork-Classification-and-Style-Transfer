# main.py

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import tensorflow_addons as tfa


# Function to load and preprocess the image
def load_and_preprocess_image(image_data):
    img = Image.open(image_data)
    img = img.resize((256, 256))  # Resize to the typical size for CycleGAN models
    img_array = np.array(img) / 255.0   # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Load the model based on user selection
@st.cache_data()
def load_model(model_name):
    custom_objects = {"InstanceNormalization": tfa.layers.InstanceNormalization}
    return tf.keras.models.load_model(model_name, custom_objects=custom_objects)

def main():
    st.title("CycleGAN Image Transformation")

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    model_choice = st.selectbox(
        'Choose a model',
        ('Monet Generator', 'Photo Generator', 'Monet Discriminator', 'Photo Discriminator')
    )

    model_mapping = {
        'Monet Generator': 'monet_generator.h5',
        'Photo Generator': 'photo_generator.h5',
        'Monet Discriminator': 'monet_discriminator.h5',
        'Photo Discriminator': 'photo_discriminator.h5'
    }

    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

        if st.button('Transform'):
            st.write(f"Transforming with {model_choice}...")

            processed_img = load_and_preprocess_image(uploaded_image)
            model = load_model(model_mapping[model_choice])

            if 'Discriminator' in model_choice:
                st.write("A discriminator does not transform images but I'll show its prediction score for your image.")
                predictions = model.predict(processed_img)
                st.write(f"Prediction Score: {predictions[0][0]}")
            else:
                transformed_img = model.predict(processed_img)
                if transformed_img.dtype == np.float32 or transformed_img.dtype == np.float64:
                    transformed_img = (transformed_img - transformed_img.min()) / (transformed_img.max() - transformed_img.min())

                st.image(transformed_img, caption=f"Transformed by {model_choice}", use_column_width=True)

if __name__ == "__main__":
    main()

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Load models when the application starts
model_cnn = load_model('model_cnn_tuned.h5')
model_resnet = load_model('model_resnet.h5')

models = {"custom cnn": model_cnn, "transfer learning - resnet": model_resnet}

class_labels = ['Titian', 'Marc_Chagall','Vincent_van_Gogh',
       'Pierre-Auguste_Renoir', 'Albrecht_Dürer', 'Paul_Gauguin',
       'Francisco_Goya', 'Rembrandt', 'Alfred_Sisley', 'Edgar_Degas', 'Pablo_Picasso']

st.title("Artworks Classifier")

# Upload the image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(224, 224))
    st.write('Please upload an image for classification')
    st.image(img, caption='Uploaded Image.', use_column_width=True)

    # Select the model
    model_name = st.selectbox("Select Model", ("custom cnn", "transfer learning - resnet"))

    if st.button('Predict'):
        st.markdown('**Predictions**')
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        model = models[model_name]  # select the model
        pred = model.predict(img)  # make prediction

        # Get the top 5 probabilities and their corresponding class labels
        top_3_indices = np.argsort(pred[0])[-3:][::-1]
        top_3_values = pred[0][top_3_indices]
        top_3_classes = [class_labels[i] for i in top_3_indices]

        # Create a DataFrame
        prediction = pd.DataFrame({
            'name': top_3_classes,
            'values': top_3_values
        })

        # Plot the results
        fig, ax = plt.subplots()
        ax = sns.barplot(y='name', x='values', data=prediction, order=prediction.sort_values('values', ascending=False).name)
        ax.set(xlabel='Confidence %', ylabel='Artists')
        st.pyplot(fig)

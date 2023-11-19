# Artwork-Classification-and-Style-Transfer
This project leverages a curated dataset from Kaggle's "Best Artworks of All Time," encompassing 50 most influential artists and 17,548 distinct images.
## Problem Overview
### 1. Image Classification
Conduct image classification to accurately categorize paintings of different artists.
1) Constructed a custom CNN and performed extensive hyperparameter tuning to optimize its performance.
2) Constructed a Transfer learning Model based on the ResNet50 architecture, capitalizing on pre-trained weights to boost accuracy.
### 2. Style Transfer
Conduct style transfer to mimic and transfer artistic styles.
1) Implemented style transfer models utilizing both VGG19 and ResNet50 architectures. The goal was to capture the artistic essence of one painting and project it onto another.
2) To delve deeper into artistic transformations, CycleGAN was employed. It was trained using a diverse dataset comprising natural landscapes and architectural imagery. The overarching ambition was to metamorphose these images to resonate with Claude Monet's iconic painting style.
### 3. Model Deployment
To bridge the gap between advanced computational models and end-users, we launched three web interfaces:
- An interface for classification based on the model we trained (custom CNN and transfer learning based on ResNet50)
- An interface for the VGG19 and ResNet50-based style transfer.
- An interface for the Monet-inspired CycleGAN style transformations.

## Result
### Image Classification
Here is a table summarizing the train and validation accuracy of the Custom CNN after hyper-parameter tuning and fine-tuned transfer learning model based on ResNet50:
<p align="center">
   <img width="500" alt="Screen Shot 2023-11-18 at 1 18 11 PM" src="https://github.com/EllieZhangy/Artwork-Classification-and-Style-Transfer/assets/133906690/e5d58fdd-4a92-4934-b7b7-e1f19b6a13df">
</p>

Furthermore, the confusion matrix is shown in the following figure: (The left one is the result of Custom CNN after Hyper-Parameter Tuning and the right one is the result of transfer learning model based on ResNet 50):
<img width="1046" alt="Screen Shot 2023-11-18 at 11 33 37 PM" src="https://github.com/EllieZhangy/Artwork-Classification-and-Style-Transfer/assets/133906690/ffd7cd51-0f9f-48f1-9b49-b0050f50d24d">

Considering the outstanding performance of the transfer learning model based on ResNet 50, here's its prediction result of 4 random images: 
<p align="center">
   <img width="800" alt="image" src="https://github.com/EllieZhangy/Artwork-Classification-and-Style-Transfer/assets/133906690/44f7750a-0b1a-4e37-93ea-41790ba3ed5e">
</p>

## Style Transfer
- Style Transfer Result of VGG19:
<p align="center">
   <img width="500" alt="Screen Shot 2023-11-18 at 1 20 02 PM" src="https://github.com/EllieZhangy/Artwork-Classification-and-Style-Transfer/assets/133906690/23875d23-2a67-48d9-9cb0-ac47f8e44033">
</p>

- Style Transfer Result of ResNet50:
<p align="center">
   <img width="500" alt="image" src="https://github.com/EllieZhangy/Artwork-Classification-and-Style-Transfer/assets/133906690/4c57e164-2f46-4f48-a7dd-1152ed179741">
</p>

- A specialized Cycle GAN model to transform photographs into Claude Monet-style paintings:
<p align="center">
   <img width="500" alt="Screen Shot 2023-11-18 at 11 46 07 PM" src="https://github.com/EllieZhangy/Artwork-Classification-and-Style-Transfer/assets/133906690/7a637947-99f3-4f79-a643-5b7f62c9f59d">
</p>
   
## Model Deployment
To bridge the gap between advanced computational models and end-users, we launched three web interfaces:
- An interface for classification based on the model we trained (custom CNN and transfer learning based on ResNet50),e.g.:
<p align="center">
  <img width="283" alt="Screen Shot 2023-11-18 at 11 49 07 PM" src="https://github.com/EllieZhangy/Artwork-Classification-and-Style-Transfer/assets/133906690/073a718a-655f-4e52-867e-4ef98a264086">
</p>

- An interface for the VGG19 and ResNet50-based style transfer.
- An interface for the Monet-inspired CycleGAN style transformations.
## Implications and Future Work:
Our comprehensive approach not only underscores the adaptability of deep learning in visual tasks but also paves the way for practical real-world applications. Future work might encompass model tuning or adjusting, and exploring other deep learning architectures for enhanced performance.
## Instructions
The code for each task is contained in individual .ipynb files, while the code and samples for each of the three web interfaces are organized in separate folders. Additionally, the project summary can be found in 'ProjectSlides.pdf', and the detailed project report is available in 'ProjectReport.pdf'.
## Data
- Main Dataset: Best Artworks of All Time (https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time). To be more specific, we subsampled the dataset to fit our computing capacity by only selecting the top artists who have 200 or more paintings. In this way, we get 11 artists(classes) in total, and here is the histogram of the count of paintings for these top artists.
<img width="600" alt="image" src="https://github.com/EllieZhangy/Artwork-Classification-and-Style-Transfer/assets/133906690/5d3b6287-665c-40d5-bc19-330944fd5a16">

- Additional data source for Cycle GAN Style Transfer (https://www.kaggle.com/datasets/ayaderaghul/gan-getting-started-2)






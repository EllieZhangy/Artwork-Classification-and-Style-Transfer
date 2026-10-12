# Artwork Classification and Style Transfer

**A visual AI project spanning artist classification, neural style transfer, and photo-to-Monet translation.**

`Computer Vision` · `CNN` · `ResNet50` · `VGG19` · `CycleGAN` · `Streamlit`

This project explores three related questions: can a model identify an artist from a painting, can it transfer visual style onto new content, and can a generative model learn a broader artistic domain transformation? The work combines model experiments with three lightweight Streamlit interfaces so the results can be explored outside a notebook.

## Project at a glance

| Track | Goal | Main approaches |
| --- | --- | --- |
| **Artist classification** | Predict the artist of an artwork | Custom CNN, transfer learning with ResNet50 |
| **Neural style transfer** | Apply one artwork's visual style to another image | VGG19- and ResNet50-based feature extraction |
| **Domain translation** | Transform photographs into Monet-like paintings | CycleGAN |
| **Interaction layer** | Make each task accessible through a UI | Three Streamlit apps |

## Data

The primary source is Kaggle's **Best Artworks of All Time** dataset, which contains 17,548 images associated with 50 influential artists. To fit the available compute budget and reduce extreme class imbalance, the classification experiments retain artists with at least 200 paintings, resulting in **11 artist classes**.

<p align="center">
  <img width="650" alt="Painting counts for the selected artist classes" src="https://github.com/EllieZhangy/Artwork-Classification-and-Style-Transfer/assets/133906690/5d3b6287-665c-40d5-bc19-330944fd5a16" />
</p>

The CycleGAN experiments use an additional photo and Monet-painting dataset from Kaggle's *GAN Getting Started* collection.

## 1. Artist classification

Two model families are compared:

- A **custom CNN**, tuned through multiple hyperparameter experiments.
- A **ResNet50 transfer-learning model**, initialized with pretrained visual features and fine-tuned for the selected artist classes.

### Performance summary

<p align="center">
  <img width="560" alt="Training and validation accuracy comparison" src="https://github.com/EllieZhangy/Artwork-Classification-and-Style-Transfer/assets/133906690/e5d58fdd-4a92-4934-b7b7-e1f19b6a13df" />
</p>

### Confusion matrices

The custom CNN is shown on the left and the fine-tuned ResNet50 model on the right.

<p align="center">
  <img width="1040" alt="Confusion matrices for the custom CNN and ResNet50 classifier" src="https://github.com/EllieZhangy/Artwork-Classification-and-Style-Transfer/assets/133906690/ffd7cd51-0f9f-48f1-9b49-b0050f50d24d" />
</p>

### Example predictions

<p align="center">
  <img width="820" alt="Example artist predictions from the ResNet50 model" src="https://github.com/EllieZhangy/Artwork-Classification-and-Style-Transfer/assets/133906690/44f7750a-0b1a-4e37-93ea-41790ba3ed5e" />
</p>

## 2. Neural style transfer

The style-transfer experiments compare feature representations from VGG19 and ResNet50. The objective is not to copy an artist exactly, but to examine how pretrained visual networks separate content structure from texture, color, and local style patterns.

### VGG19 result

<p align="center">
  <img width="540" alt="Neural style transfer result using VGG19" src="https://github.com/EllieZhangy/Artwork-Classification-and-Style-Transfer/assets/133906690/23875d23-2a67-48d9-9cb0-ac47f8e44033" />
</p>

### ResNet50 result

<p align="center">
  <img width="540" alt="Neural style transfer result using ResNet50" src="https://github.com/EllieZhangy/Artwork-Classification-and-Style-Transfer/assets/133906690/4c57e164-2f46-4f48-a7dd-1152ed179741" />
</p>

## 3. Photo-to-Monet domain translation

A CycleGAN model learns an unpaired mapping between ordinary photographs and Monet-like paintings. Unlike single-image neural style transfer, this task learns a broader transformation from two image domains.

<p align="center">
  <img width="560" alt="CycleGAN examples translating photographs into Monet-like paintings" src="https://github.com/EllieZhangy/Artwork-Classification-and-Style-Transfer/assets/133906690/7a637947-99f3-4f79-a643-5b7f62c9f59d" />
</p>

## Interactive demos

Three Streamlit interfaces were built for:

1. Artist classification with the custom CNN and ResNet50 models.
2. VGG19- and ResNet50-based neural style transfer.
3. CycleGAN photo-to-Monet transformation.

<p align="center">
  <img width="320" alt="Streamlit interface for artwork classification" src="https://github.com/EllieZhangy/Artwork-Classification-and-Style-Transfer/assets/133906690/073a718a-655f-4e52-867e-4ef98a264086" />
</p>

The interfaces are included as project artifacts; they are not currently maintained as public hosted services.

## Repository guide

- Task-specific experiments are stored in individual Jupyter notebooks.
- The three interface implementations and example assets are organized in separate app folders.
- `ProjectSlides.pdf` contains the presentation summary.
- `ProjectReport.pdf` contains the detailed methodology and analysis.

Because the experiments were created in 2023, package versions, pretrained-model URLs, and notebook runtimes may require updates before reproduction.

## Limitations and next steps

- The 11-class subset improves feasibility but does not represent the full 50-artist dataset.
- Artist labels can encourage a model to learn repeated subjects, palettes, or dataset artifacts rather than deeper artistic authorship.
- Accuracy alone does not explain where two artists are visually confusable; embedding analysis and error inspection would add useful context.
- Neural style transfer and CycleGAN outputs should be assessed visually as well as with automated metrics.
- Future work could add stronger augmentation, modern vision backbones, model cards, reproducible environment files, and a unified web experience.

## Selected work

- [AI Driving School](https://elliezhangy.github.io/ai-driving-school/) — a voice- and text-controlled Chinese parking game.
- [Daily Mood Screen](https://github.com/EllieZhangy/Daily-Mood-Screen) — a personalized macOS wallpaper pipeline using weather, photos, and mood.
- [Memory Studio](https://github.com/EllieZhangy/my-ai-poster-skills) — a photo-to-poster web app with reusable visual styles.
- [Radiology Impression Prediction](https://github.com/EllieZhangy/LLM-Based-Impression-Prediction-from-Radiology-Reports) — an LLM research project for radiology report impression generation.

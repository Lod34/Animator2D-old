# 🎨 Animator2D

Animator2D is an AI-powered model designed to generate pixel-art sprite animations from textual descriptions. This model leverages a BERT-based text encoder to extract textual features and a convolutional generative network to create animated sprites. The goal is to provide game developers and artists with a tool that can bring character concepts to life with minimal effort.

Link to Hugging Face account: https://huggingface.co/Lod34

## 🛠️ Model Overview

- **Name:** Animator2D
- **Input:**
  - Character description
  - Number of animation frames
  - Character action
  - Viewing direction
- **Output:** Animated sprite sheet in image format

## 📦 Dataset

The model was trained using the [spraix\_1024](https://huggingface.co/datasets/pawkanarek/spraix_1024) dataset, which contains animated sprites with detailed textual descriptions. This dataset serves as a foundation for training the model to generate high-quality, relevant sprites based on textual inputs.

## 🚀 Model Versions

Over time, several iterations of Animator2D have been developed, each improving on the previous version with different training strategies and hyperparameters. Below is a chronological overview of the versions created so far:

| Model Version        | Description |
|----------------------|-------------|
| **Animator2D-v1** | The first full version developed in this project, utilizing a structured training approach with BERT for text encoding and a convolutional generator for sprite creation. |
| **Animator2D-mini-10e** | A simplified version trained with only 10 epochs, batch size of 8, learning rate of 1e-4, and image size of 64x64. |
| **Animator2D-mini-100e** | An extension of the mini-10e version, trained for 100 epochs for improved performance. |
| **Animator2D-mini-250e** | A more refined version with 250 epochs, batch size increased to 16, learning rate of 2e-4, and image resolution of 128x128. |
| **Animator2D-v2 (In Development)** | A new version being built from scratch with an entirely redesigned training process, aiming for better animation quality and efficiency. |

## 🔮 Future Goals

This is just the first iteration of Animator2D. Future updates will focus on refining and expanding its capabilities:

- **Multiple Output Formats**: Currently, the model generates a single sprite sheet. Future updates will enable exporting animations in various formats, including folders with individual frames, GIFs, and videos.
- **Frame Input Optimization**: The number of frames is currently manually defined. Improvements will include a more intuitive system that considers FPS and actual animation duration.
- **Model Refinement**: The current model is in an early stage. Future improvements will enhance sprite generation consistency and quality by optimizing the architecture and training dataset.
- **Sprite Size Customization**: A new input will allow users to specify the character height in pixels, dynamically adjusting the sprite’s artistic style. This will ensure greater flexibility, allowing for different art styles (e.g., Pokémon vs. Metal Slug aesthetics).

---

Animator2D is an exciting step toward AI-assisted sprite animation generation, and future versions will continue to push the boundaries of what’s possible in pixel-art automation! 🚀🎮

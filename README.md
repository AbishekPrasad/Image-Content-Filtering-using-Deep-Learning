# Image Content Filtering using Deep Learning

This project implements an image classification pipeline using deep learning to automatically detect and filter images based on their content. 
It uses Convolutional Neural Networks (CNNs) to classify images into categories such as `drawings`, `hentai`, `neutral`, `porn`, and `sexy`. 
The system is suitable for applications like content moderation, parental control, and image search filtering.

---

## Project Objectives

- Classify images into predefined content categories using deep learning.
- Load and process images from URLs with robust error handling and augmentation.
- Provide a modular, customizable pipeline for scalable integration.
- Enable real-time classification for moderation and safe-search systems.

---

## Model Overview

- **Architecture**: Convolutional Neural Network (CNN)
- **Input Size**: 128x128 RGB images
- **Output**: 5-class `softmax` classifier
- **Optimizer**: Adam
- **Loss**: Categorical Crossentropy





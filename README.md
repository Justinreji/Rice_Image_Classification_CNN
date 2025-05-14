# Rice Image Classification using CNNs

This project implements a **Convolutional Neural Network (CNN)** to classify images of rice grains into four categories: **Arborio, Basmati, Ipsala, and Jasmine**. The model is trained from scratch using PyTorch and achieves high classification accuracy while remaining lightweight and efficient.

## Story

In the agricultural industry, ensuring the quality and correct classification of rice grains is vital for both domestic and international trade. Rice varieties like Arborio, Basmati, Ipsala, and Jasmine differ in aroma, shape, texture, and culinary use. Mislabeling or mixing these can lead to economic losses, export rejections, and reduced consumer trust.

Traditionally, rice classification has relied on manual inspection, which is time-consuming, subjective, and prone to human error. As global demand grows, the industry needs scalable, automated, and accurate solutions. This project leverages a Convolutional Neural Network (CNN) to automate the classification of rice varieties based on image data. By analyzing grain features visually, the model assists in quality control, packaging, and supply chain verification, providing a faster, more consistent alternative to manual sorting.

## The Dat

Dataset: https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset

- Each class contains a balanced number of images.
- Total dataset: ~60,000 images, 15,000 images for each rice type
- Image size resized to **64×64 pixels**

Rice types:
- Arborio
- Basmati
- Ipsala
- Jasmine

## Project Overview
- **Framework**: PyTorch
- **Model**: Custom CNN with 2 convolutional layers + 2 fully connected layers
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam
- **Normalization**: Dataset-specific mean and std:
  ```python
  mean = [0.1259, 0.1272, 0.1301]
  std = [0.2949, 0.2980, 0.3062]
  ```
- **Train/Val/Test Split**: 70% / 15% / 15%

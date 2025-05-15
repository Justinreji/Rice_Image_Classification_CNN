# Rice Image Classification using CNNs

This project implements a **Convolutional Neural Network (CNN)** to classify images of rice grains into four categories: **Arborio, Basmati, Ipsala, and Jasmine**. The model is trained from scratch using PyTorch and achieves high classification accuracy while remaining lightweight and efficient.

## Story

In the agricultural industry, ensuring the quality and correct classification of rice grains is vital for both domestic and international trade. Rice varieties like Arborio, Basmati, Ipsala, and Jasmine differ in aroma, shape, texture, and culinary use. Mislabeling or mixing these can lead to economic losses, export rejections, and reduced consumer trust.

Traditionally, rice classification has relied on manual inspection, which is time-consuming, subjective, and prone to human error. As global demand grows, the industry needs scalable, automated, and accurate solutions. This project leverages a Convolutional Neural Network (CNN) to automate the classification of rice varieties based on image data. By analyzing grain features visually, the model assists in quality control, packaging, and supply chain verification, providing a faster, more consistent alternative to manual sorting.

## The Data

Dataset: https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset

- Each class contains a balanced number of images.
- Total dataset: ~60,000 images, 15,000 images for each rice type
- Image size resized to **64×64 pixels**

Rice types:
- Arborio
- Basmati
- Ipsala
- Jasmine

![rice_images_ML](https://github.com/user-attachments/assets/146a2942-40e4-4e04-8763-c95917e94b01)


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

## CNN Architecture
```
Input: 3×64×64 RGB Image
↓ Conv2D (3→16, kernel=5, padding=2) + ReLU
↓ MaxPool2D (2×2)
↓ Conv2D (16→32, kernel=3, padding=1) + ReLU
↓ MaxPool2D (2×2)
↓ Flatten (32×16×16 → 8192)
↓ Fully Connected (8192 → 32) + ReLU
↓ Fully Connected (32 → 4) → Output Logits
```
- Total trainable parameters: **268,164**

## Training Performance

### Training Loss and Test Accuracy
Training Loss shows how well the model minimizes the classification error over training epochs. While Test Accuracy tracks model generalization across unseen test data after each epoch.

![training_loss](https://github.com/user-attachments/assets/fb2e7f5c-0241-4c5a-a663-7403a076800a)

Loss starts high (~155) and drops quickly within the first two epochs, while the test accuracy increases over the epochs. The model is learning effectively and minimizing classification error with each epoch.

### Confusion Matrix
Displays how often predictions match the actual rice class.

![rice_Image_Confusion_matrix](https://github.com/user-attachments/assets/8d04b76f-fde4-47ff-8772-8ea2547d0777)

Model performs extremely well across all classes, very minimal misclassifications, mostly between visually similar grains.

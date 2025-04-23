# Project - 2

## List of the Files to be Graded
### 🎥 Video Demo

[![Video Demo](https://img.shields.io/badge/Video%20Demo-MP4-red?logo=video&style=for-the-badge)](./Project2.mp4) 

### 📄 Research Paper

[![Paper](https://img.shields.io/badge/Research%20Paper-PDF-blue?logo=pdf&style=for-the-badge)](./Project2.pdf) 

### 📝 Training Notebook

[![Training Notebook](https://img.shields.io/badge/Training%20Notebook-.ipynb-orange?logo=jupyter&style=for-the-badge)](./training.ipynb)

This project focuses on generating descriptive captions for images using a hybrid CNN-Transformer model on the Flickr8k dataset. It explores the integration of a pre-trained ResNet-50 for image feature extraction and a Transformer decoder for caption generation, with an emphasis on evaluating model performance through BLEU scores and loss curves. The dataset is sourced from Flickr8k, processed using PyTorch, and analyzed with extensive Exploratory Data Analysis (EDA).

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Project Overview
The goal of this project is to develop and evaluate an image captioning system using a hybrid CNN-Transformer architecture on the Flickr8k dataset. The project investigates the following tasks:
1. **Data Preparation and EDA**: Loading and preprocessing the Flickr8k dataset, followed by comprehensive EDA to understand caption characteristics.
2. **Model Development**: Implementing a hybrid model with ResNet-50 as the encoder and a Transformer as the decoder for caption generation.
3. **Training and Evaluation**: Training the model for 20 and 100 epochs, evaluating performance using BLEU scores, and analyzing training/validation loss curves.

The dataset is preprocessed with standard image transformations (resize, normalization) and caption cleaning (lowercase, punctuation removal). The model is trained and evaluated using PyTorch on a GPU environment.

## Features
- **Dataset Handling**: Loads and processes Flickr8k images and captions, splitting into train, validation, and test sets.
- **Exploratory Data Analysis (EDA)**: Visualizes caption length distribution, word frequency, word cloud, and caption similarity.
- **Hybrid Model**: Combines ResNet-50 for image feature extraction and a Transformer decoder for caption generation.
- **Model Training**: Implements training loops with teacher forcing and evaluates using beam search.
- **Performance Evaluation**: Reports BLEU-1 and BLEU-4 scores, along with training and validation loss curves.

## Requirements
- Python 3.10+
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- PIL (Pillow)
- WordCloud
- Optional: Kaggle API (for dataset download)

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/UF-EGN6217-Spring25/project-2-Shreejit2401.git
   cd project-2-Shreejit2401
   ```
2. **Install Dependencies:**
   ```py
   pip install -r requirements.txt
   ```
3. **Download the Dataset:**
   ```bash
   !wget -q -P /kaggle/working/flickr8k https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip
   !wget -q -P /kaggle/working/flickr8k https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip
   !unzip -q -o /kaggle/working/flickr8k/Flickr8k_Dataset.zip -d /kaggle/working/flickr8k/
   !unzip -q -o /kaggle/working/flickr8k/Flickr8k_text.zip -d /kaggle/working/flickr8k/
   ```
4. **Pretrained Models & Transformer:**
   ```bash
   import torchvision.transforms as transforms
   from torchvision.models import resnet50
   ```
## Usage

1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook training.ipynb
   ```
2. Execute cells sequentially to load libraries, preprocess data, perform EDA, train the model, and evaluate results.

### Key Sections

- **Importing Libraries**: Include code to load necessary Python packages like PyTorch, torchvision, NumPy, Pandas, Matplotlib, Seaborn, and WordCloud.
- **Dataset Loading**: Include code to load and preprocess the Flickr8k dataset, splitting into train, validation, and test sets.
- **EDA**: Include code to generate and save plots for caption length distribution, top words, word cloud, and caption similarity.
- **Model Definition**: Include code to define the hybrid CNN-Transformer model (`EncoderCNN` and `TransformerDecoder` classes).
- **Training**: Include code to train the model for 20 and 100 epochs, saving loss curves.
- **Evaluation**: Include code to generate captions using beam search and compute BLEU-1 and BLEU-4 scores.

### Modify Parameters (if needed)
- Adjust hyperparameters like `BATCH_SIZE`, `EPOCHS`, `EMBED_SIZE`, or `MAX_LEN` in the notebook.
- Experiment with different image transforms in the `transforms.Compose` pipeline.

## Results

### EDA Insights
- **Total images**: 8,092
- **Total captions**: 40,458
- **Unique captions**: 40,018 (Diversity ratio: 0.9891)
- **Average caption length**: 9.79 words (std: 3.76)
- **Average caption similarity**: 0.0167
- **Vocabulary size**: 3,024

### Model Performance
#### BLEU Scores
- **BLEU-1**: 0.2716
- **BLEU-4**: 0.0549

#### Loss Curves
- Training loss decreases steadily over 20 and 100 epochs, but validation loss plateaus, indicating overfitting. Due to the lack of processing capacity, even though I knew the model was overfitting, I couldn’t do much about it (e.g., applying regularization or data augmentation).

The project demonstrates the feasibility of using a hybrid CNN-Transformer model for image captioning, though challenges like overfitting and limited caption diversity remain due to computational constraints.

## Contributing

Contributions are not allowed 😉! (Individual Assignment)

## Authors

- **Shreejit Cheela** - Solution Development - [Your GitHub](https://github.com/Shreejit2401)

## Acknowledgements

I would like to express my deepest gratitude to the individuals, organizations, and communities whose resources and support have contributed to the success of this project.

### Dataset
- **Flickr8k Dataset** - The dataset used for model training and evaluation.
   
   https://github.com/jbrownlee/Datasets/releases/download/Flickr8k

### Computing Resources
- **Kaggle** - For providing GPU resources essential for model training and experimentation.

### Frameworks and Libraries
- **PyTorch** - The primary deep learning framework used for building and training the model.
- **Torchvision** - For pre-trained models and image transformations.
- **NumPy** - For numerical operations and efficient data handling.
- **Pandas** - For data manipulation and analysis.
- **Matplotlib** - For plotting and visualization.
- **Seaborn** - For enhanced data visualization.
- **WordCloud** - For generating word cloud visualizations.

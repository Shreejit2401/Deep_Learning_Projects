# Project - 3

## List of the Files to be Graded
### 🎥 Video Demo

[![Video Demo](https://img.shields.io/badge/Video%20Demo-MP4-red?logo=video&style=for-the-badge)](./Project3.mp4) 

### 📄 Research Paper

[![Paper](https://img.shields.io/badge/Research%20Paper-PDF-blue?logo=pdf&style=for-the-badge)](./Project3.pdf) 

### 📝 Training Notebook

[![Training Notebook](https://img.shields.io/badge/Training%20Notebook-.ipynb-orange?logo=jupyter&style=for-the-badge)](./training.ipynb)

VocalVision focuses on generating descriptive captions for images using a hybrid DeiT-LSTM model with attention on the Flickr8k dataset. It extends the image captioning task by including multilingual translation and speech synthesis for assistive applications. The architecture integrates a fine-tuned DeiT transformer for image feature extraction and an attention-based LSTM decoder for caption generation. Translations and audio outputs in five languages add accessibility for visually impaired and multilingual users.

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
The goal of this project is to develop an assistive, multilingual image captioning system using a hybrid CNN-Transformer architecture. The project addresses the following key tasks:
1. **Data Preparation and EDA**: Loading and preprocessing the Flickr8k dataset with detailed analysis of captions.
2. **Model Development**: Implementing a hybrid architecture using DeiT as the encoder and an attention-enhanced LSTM decoder.
3. **Training and Evaluation**: Training for 100 epochs, evaluating using BLEU and ROUGE scores, and analyzing loss curves.
4. **Multilingual Output**: Translating captions into five languages and generating speech output for each, making the system useful for visually impaired users.

The dataset is processed using PyTorch pipelines, and inference outputs include both text and audio representations.

## Features
- **Dataset Handling**: Automates loading, transformation, and tokenization of the Flickr8k dataset.
- **EDA and Visualization**: Analyzes word frequency, caption length, and diversity.
- **Hybrid Model**: Combines DeiT for visual feature extraction with an LSTM and attention for decoding.
- **Multilingual Output**: Translates generated captions into five languages using `deep-translator`.
- **Text-to-Speech**: Converts translated text into audio using gTTS.
- **Performance Evaluation**: Tracks loss and evaluates model using BLEU and ROUGE metrics.

## Requirements
- Python 3.10+
- PyTorch
- Torchvision
- Timm
- NumPy
- Pandas
- Matplotlib
- Rouge-score
- NLTK
- gTTS
- Deep-Translator
- IPython

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/UF-EGN6217-Spring25/project-3-Shreejit2401.git
   cd project-3-Shreejit2401
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
    import timm
    deit = timm.create_model('deit_base_patch16_224', pretrained=True)
   ```
## Usage

1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook training.ipynb
   ```
2. Execute cells sequentially to load libraries, preprocess data, perform EDA, train the model, and evaluate results.

### Key Sections

- **Library Imports**: Load PyTorch, timm, NLTK, and other essential packages.
- **Dataset Pipeline**: Load and preprocess images and captions, tokenize, and build vocabulary.
- **EDA Visuals**: Generate caption distribution, frequency charts, and vocabulary insights.
- **Model Construction**: Define the DeiT encoder and LSTM decoder with attention.
- **Training**: Train the model for 100 epochs, observe loss/accuracy curves.
- **Evaluation**: Compute BLEU and ROUGE scores, visualize qualitative output.
- **Translation and Audio**: Translate captions using deep-translator and generate .mp3 files using gTTS.

### Modify Parameters (if needed)
- Adjust hyperparameters like `BATCH_SIZE`, `EPOCHS`, `EMBED_SIZE`, or embedded sizes in the notebook.
- Add or Remove languages from translation block for testing

## Results

### EDA Insights
- **Total images**: 8,092
- **Total captions**: 40,458
- **Unique captions**: 40,018 (Diversity ratio: 0.9891)
- **Vocabulary size**: 4,239
- **Avg Caption Length**: 9.79 words (std: 3.76)

### Model Performance
#### BLEU Scores
- **BLEU-4**: 0.215
- **ROUGE-L**: 0.718

#### Loss Curves
- Training loss drops below 1.5 over 100 epochs.
- Validation loss plateaus slightly, showing generalization but some room for regularization.

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
- **Timm** - For pre-trained DeiT encoder.
- **NumPy, Pandas** - For data manipulation and analysis.
- **Matplotlib** - For loss/score plotting.
- **NLTK, ROUGE** - For text evaluation metrics.
- **gTTS, deep-translator** - For multilingual audio generation.
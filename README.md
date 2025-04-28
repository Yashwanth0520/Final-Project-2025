# Analysing Spectrogram Parameters and Convolutional Neural Networks for Animal Emotion Recognition

## Overview
This project aims to teach a computer to recognize **animal emotions** (like happy, angry, or scared) using the sounds they make. We use **spectrograms** (pictures made from sounds) and smart computer models called **Convolutional Neural Networks (CNNs)** to do this.

---

## Key Highlights

- **Animal Sounds**: We listen to animal calls to understand how they feel.
- **Spectrograms**: We convert animal sounds into colorful pictures that computers can study.
- **CNN Model**: A deep learning model that looks at these pictures and learns patterns linked to different emotions.
- **Random Forest Model**: Another machine learning method used for comparison.
- **Parameter Tuning**: We tested different settings (like window size and hop size) to find the best way to make the spectrograms.

---

## Objectives

- Study how spectrogram settings (window size and hop length) affect the model's ability to recognize emotions.
- Compare CNN and Random Forest models.
- Find the best settings and model for accurate animal emotion recognition.

---

## Dataset
- Contains sounds from different animals with labels for their emotions.
- Categories: **Happy, Angry, Sad, Fearful**.
- Data was preprocessed to remove noisy or low-quality recordings.

---

## Methodology

1. **Data Collection**:
   - Animal sound recordings labeled with emotions.

2. **Preprocessing**:
   - Create spectrogram images from sound using different window sizes (1024, 2048, 4096).
   - Normalize and clean the data.

3. **Model Building**:
   - Build a simple 3-layer CNN.
   - Train and validate the model with spectrogram images.
   - Compare CNN results with Random Forest classifiers.

4. **Testing and Evaluation**:
   - Use accuracy, precision, recall, and F1-score to judge model performance.

---

## How to Run the Project

1. Install required packages:
```bash
pip install numpy pandas librosa matplotlib tensorflow scikit-learn
```

2. Prepare the dataset:
- Organize animal sounds into folders by emotion.
- Convert sounds into spectrogram images.

3. Train the CNN model:
```bash
python train_cnn.py
```

4. Evaluate the model:
```bash
python evaluate_model.py
```
---
## Tools and Technologies Used
- Python
- TensorFlow / Keras
- Librosa (for audio processing)
- Scikit-learn
- Matplotlib

---
## Results

- **Best Model**: CNN with window size 1024 performed the best.
- **Accuracy**: Over 91%.
- **Observations**:
  - Smaller window sizes captured animal emotions better.
  - CNN outperformed Random Forest significantly.
  - Fear emotion was the hardest to classify due to fewer examples.

---

## Challenges

- Imbalanced dataset (fewer samples for some emotions like fear).
- Small changes in spectrogram parameters made a big difference.
- CNN needed careful tuning to avoid overfitting.

---

## Future Work

- **Use LSTM models** to capture time-based features better.
- **Data Augmentation**: Create synthetic data to balance emotion classes.
- **Transfer Learning**: Use pre-trained models to save time and resources.
- **Self-Supervised Learning**: Train models without needing labeled data.

---

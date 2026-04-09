Baby Cry Classification System

Overview

This project focuses on building a deep learning system to classify different types of baby cries using audio signals. The goal is to identify the reason behind a baby’s cry (such as hunger, pain, discomfort, etc.) by analyzing sound patterns.

The system processes raw audio data, extracts meaningful features (MFCC), and uses an LSTM-based neural network to learn temporal patterns in the audio.

Problem Statement

Understanding why a baby is crying can be difficult, especially for new parents. This project aims to assist by automatically classifying baby cries into predefined categories using machine learning.

Dataset

The dataset consists of labeled audio files organized into multiple classes. Each class represents a different type of cry. The dataset was balanced to ensure equal representation (500 samples per class) to improve model performance and reduce bias.

Project Pipeline
1. Data Collection
    Audio files in .wav and .ogg format organized class-wise.
2. Preprocessing
    Audio files are loaded using librosa
    Noise handling and standard sampling rate applied
    Each audio is converted into MFCC (Mel Frequency Cepstral Coefficients)
3. Feature Engineering
    Extracted 40 MFCC features per audio
    Standardized all samples to fixed length using padding/truncation
    Saved features as .npy files for efficient reuse
4. Model Building (LSTM)
    Used LSTM layers to capture time-based patterns in audio
    Added dropout layers to reduce overfitting
    Dense layers used for final classification
5. Training and Validation
    Dataset split into training and validation sets
    Model trained over multiple epochs
    Performance evaluated using accuracy and loss
6. Evaluation
    Confusion matrix used to analyze class-wise performance
    Learning curves plotted for training vs validation
7. Prediction System
    New audio input is converted to MFCC
    Model predicts the class of baby cry


Technologies Used
    Python
    NumPy
    Pandas
    Librosa
    TensorFlow / Keras
    Scikit-learn

Model Architecture
    Input: MFCC features (time-series)
    LSTM Layer (128 units)
    Dropout Layer
    LSTM Layer (64 units)
    Dropout Layer
    Dense Layer (ReLU)
    Output Layer (Softmax for multi-class classification)


Results

The model was trained on a balanced dataset and showed improvement in classification performance compared to earlier versions. The use of MFCC and LSTM helped capture temporal audio patterns effectively.

Challenges Faced
    Handling different audio formats (.wav and .ogg)
    Fixing variable-length audio inputs
    Class imbalance in initial dataset
    Overfitting during training
    Managing file path and data pipeline errors


Future Improvements
    Apply advanced audio augmentation techniques
    Implement real-time prediction system
    Improve model generalization with larger datasets
    How to Run

Generate MFCC features:
    python generate_mfcc.py

Train the model:
    python train_lstm_mfcc.py

Run prediction:
    python predict.py


Conclusion

This project demonstrates how deep learning can be applied to audio classification problems. By combining signal processing techniques with LSTM networks, the system is able to learn meaningful patterns from baby cry sounds and classify them effectively.

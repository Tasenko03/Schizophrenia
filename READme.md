### Schizophrenia Detection from Speech using Custom CNN Architecture

This repository contains a Deep Learning pipeline designed to classify schizophrenia markers in human speech. The project converts raw audio into visual representations (Mel-spectrograms) and utilizes a custom-designed Convolutional Neural Network (CNN) for classification.

## Key Features
* Custom "Shico" Architecture: A specialized CNN with multi-scale feature extraction, residual identity blocks, and global feature merging.
* Advanced Audio Preprocessing: Automated silence trimming, sampling at 48kHz, and conversion to 3-channel input (Mel-spectrogram + Delta + Delta-Delta coefficients).
* Robust Data Augmentation: Implementation of SpecAugment (frequency and time masking) to prevent overfitting and improve generalization on small medical datasets.
* High Recall: Optimized for medical screening, achieving a 93.3% recall rate.

## Model Architecture: Shico_model
* The model is designed to capture both fine-grained acoustic details and global patterns:
* ConvBlocks: Progressive feature extraction with Batch Normalization and ReLU activation.
* Identity Blocks: Multi-scale skip-connections that resize and merge features from different layers (scale factors 8, 4, 2).
* Concatenation Layer: Merges local and global features before the final classification head.
* Classification Head: Sequential layers with PReLU activation, Dropout (0.2), and BatchNorm for stable convergence.

## Tech Stack
* Framework: PyTorch
* Audio Processing: Librosa
* Augmentation: Custom SpecAugment implementation
* Metrics: Scikit-learn (ROC AUC, F1, Recall)

## Performance Results
The model was evaluated on a validation split with the following results:
* Accuracy	82.26%
* Recall (Sensitivity)	93.33%
* F1-Score	83.58%
* ROC AUC	82.60%
* Precision	75.68%

## Project Structure
* read_as_melspectrogram: Advanced audio-to-image conversion.
* SpecAugment: Frequency and time domain masking for audio data.
* Shico_model: The core neural network architecture.
* NewDataset: Custom PyTorch Dataset class for handling wav-files and labels.

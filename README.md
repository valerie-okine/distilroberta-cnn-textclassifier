# Multi-Label Text Classification with CNN and DistilRoBERTa

This repository contains a multi-label text classification pipeline that combines deep learning and transformer-based tokenization. It leverages a Convolutional Neural Network (CNN) built in TensorFlow and uses the DistilRoBERTa tokenizer to process and classify short texts into multiple categories.

Features
	•	Performs multi-label classification with a custom CNN model.
	•	Utilizes DistilRoBERTa for tokenizing text inputs.
	•	Tracks performance using metrics such as F1 Score and AUC.
	•	Designed with a modular structure for training and prediction.

Project Summary

All required Python libraries are listed in the requirements.txt file and should be installed prior to running the project.

Training is performed on CSV datasets with a text column and multiple binary label columns. The model is trained using a CNN architecture optimized for short text sequences.

Predictions are generated using the saved model and applied to a new dataset. The output includes predicted labels and an evaluation score that reflects the model’s performance.

The project is designed to be flexible, supporting both training and prediction through a command-line interface.

Model Architecture
	•	Embedding Layer: Transforms token IDs into dense vector representations.
	•	Convolutional Layers: Extract n-gram features from sequences.
	•	Pooling Layers: Reduce dimensionality and retain key features.
	•	Dense Layers with Dropout: Perform classification while minimizing overfitting.
	•	Sigmoid Output: Supports multi-label classification by independently scoring each label.

Evaluation

Model performance is evaluated using the micro-averaged F1 score, a robust metric suitable for multi-label classification tasks, especially in cases with label imbalance.

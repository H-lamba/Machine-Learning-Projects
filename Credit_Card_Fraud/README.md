

# Credit Card Fraud Detection using 1D CNN

This project aims to build an efficient fraud detection system for credit card transactions using a deep learning approach, specifically a 1D Convolutional Neural Network (CNN). The model is trained, validated, and evaluated on a comprehensive synthetic dataset featuring various transaction, personal, and merchant details.

## Table of Contents

- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training & Results](#training--results)
- [Usage](#usage)
- [Requirements](#requirements)
- [License](#license)

## Dataset

The dataset was downloaded from Kaggle using the `kartik2112/fraud-detection` dataset available via `kagglehub`. It contains millions of credit card transactions, with labeled fraud and genuine records. Key features include timestamp, merchant, amount, category, user demographic, location, and a binary target (`is_fraud`).

**Number of Columns (Features):** 23  
**Target Variable:** `is_fraud` (0 - genuine, 1 - fraud)

## Project Structure

- **Credit_card.ipynb**: Main notebook containing complete code for data loading, preprocessing, model building, training, and evaluation.
- **Dataset**: Downloaded automatically via `kagglehub`.
- **Trained Model**: Model code and summary are shown; you can save the trained model as required.

## Data Preprocessing

- **Loading & Cleaning**: Data was loaded using Pandas, and irrelevant or high-cardinality columns (e.g., names, transaction IDs, addresses) were encoded or dropped as needed.
- **Encoding**: Categorical variables were transformed using label encoding.
- **Scaling**: Numerical features scaled using `StandardScaler` for normalization.
- **Splitting**: Dataset split into training and test sets using `sklearn`.

## Model Architecture

The model uses a 1D Convolutional Neural Network, optimized for tabular sequences:
- **Conv1D** layers for feature extraction.
- **BatchNormalization** for stable learning.
- **Dropout** layers (for regularization, preventing overfitting).
- **Dense (Fully Connected) Layers** for final classification.

Model summary:
```
Layer (type)              Output Shape          Param #
---------------------------------------------------------
conv1d                    (None, 21, 32)        96
batch_normalization       (None, 21, 32)        128
dropout                   (None, 21, 32)        0
conv1d                    (None, 20, 64)        4,160
batch_normalization       (None, 20, 64)        256
dropout                   (None, 20, 64)        0
flatten                   (None, 1280)          0
dense                     (None, 64)            81,984
dropout                   (None, 64)            0
dense                     (None, 1)             65
---------------------------------------------------------
Total params: ~86,700
```
- **Activation**: Binary output with sigmoid for fraud probability.

## Training & Results

- **Loss Function**: Binary Crossentropy (for binary classification)
- **Optimizer**: Adam, with a learning rate of 0.0001
- **Epochs**: 30
- **Metrics**: Accuracy, validation accuracy/loss observed
- **Validation Accuracy Achieved**: ~86.5%
- **Performance Visualization**: Learning curves plotted for training/validation accuracy and loss.

Example results:
```
Epoch 30/30
accuracy: 0.8684 - loss: 0.3242 - val_accuracy: 0.8650 - val_loss: 0.3453
```

## Usage

1. **Environment Setup**:
   - Install dependencies:
     ```
     pip install tensorflow pandas numpy sklearn seaborn matplotlib kagglehub
     ```

2. **Run Notebook**:
   - Download the dataset using kagglehub (shown in cell 1).
   - Ensure all preprocessing and model cells are run in order.

3. **Model Training/Evaluation**:
   - Adjust hyperparameters (CNN layers, dropout, learning rate, batch size) to optimize performance as needed.
   - Use the provided plot functions to visualize training curves.

4. **Prediction**:
   - Input transaction data (preprocessed) to the trained model for `is_fraud` prediction.

## Requirements

- Python 3.7+
- TensorFlow >= 2.x
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn
- kagglehub

## License

*This project is for educational and non-commercial research purposes. The dataset is for research only and must not be used in a production environment.*

**Credits:**  
- [Kaggle: kartik2112/fraud-detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
- TensorFlow, Keras and Scikit-learn Teams

For questions or improvements, please open an issue or a pull request.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/83411276/6dfa5dd6-e3f3-4594-8ad4-70c63b6e8a6c/Credit_card.ipynb

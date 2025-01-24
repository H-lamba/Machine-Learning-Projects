# Retinal OCT Disease Classification

## 📌 Project Overview
Retinal diseases such as Diabetic Retinopathy, Age-Related Macular Degeneration (AMD), and Glaucoma can lead to severe vision impairment if not detected early. This project implements a deep learning-based classification system to automate the detection of retinal diseases using Optical Coherence Tomography (OCT) images.

Leveraging Convolutional Neural Networks (CNNs) and transfer learning models like ResNet, MobileNet, and VGG16, this project achieves high accuracy in both binary and multi-class classification tasks. A Flask-based web application is also developed for real-time diagnostic predictions, improving accessibility and usability.

## 🚀 Features
- Deep Learning-based Retinal Disease Detection using CNNs and Transfer Learning
- Multi-class Classification (Normal, CNV, DME, Drusen)
- Flask Web Application for real-time image-based diagnosis
- Grad-CAM Visualizations for explainable AI-driven decision-making
- Performance Metrics Tracking (Accuracy, Precision, Recall, F1 Score)
- Optimized for Cloud & Edge Deployments

## 📂 Dataset
We use the Kermany2018 Retinal OCT Dataset, which consists of 84,495 high-resolution images categorized into four classes:
- Normal
- Choroidal Neovascularization (CNV)
- Diabetic Macular Edema (DME)
- Drusen

Preprocessing techniques such as image resizing, normalization, augmentation, segmentation, and contrast enhancement have been applied to improve model performance.

## 🖥️ Technologies Used
- Python (TensorFlow, PyTorch, OpenCV)
- Deep Learning (CNNs, ResNet, MobileNet, VGG16)
- Machine Learning (Random Forest Classifier for comparison)

## ⚡ Installation & Setup
1. Clone the repository
    ```sh
    git clone https://github.com/H-lamba/retinal-oct-classification.git  
    cd retinal-oct-classification
    ```
2. Install dependencies
    ```sh
    pip install -r requirements.txt
    ```

## 📊 Model Performance
| Model           | Accuracy | Precision | Recall  | F1 Score |
|-----------------|----------|-----------|---------|----------|
| Baseline CNN    | 96.88%   | 89.14%    | 97.52%  | 91.34%   |
| Random Forest   | 82.66%   | 75.23%    | 81.15%  | 78.01%   |

## 🎯 Future Enhancements
- Integrate Vision Transformers (ViTs) for improved accuracy
- Expand dataset to include additional retinal diseases
- Deploy on cloud and edge devices for real-world accessibility
- Enhance explainability using SHAP and LIME visualizations
- Telemedicine integration for remote diagnostics

## 📝 Contributors
- 👨‍💻 Aditya Gupta (23BAI10032)
- 👨‍💻 Himanshu (23BAI10041)
- 👩‍💻 Tejashri Choudhary (23BAI11279)
- 👨‍💻 Akshat Jha (23BAI10181)
- 👨‍💻 Chitransh Soral (23BAI10312)


## 📞 Contact
For queries, collaborations, or suggestions, please reach out via:
- 📧 Email: [Gmail](hljaat18@gmail.com)
- 🔗 GitHub: https://github.com/H-lamba

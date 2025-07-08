# 🧠 Brain Tumor Classifier

This project implements a convolutional neural network (CNN) with residual connections to classify brain MRI scans into four categories: **glioma**, **meningioma**, **pituitary**, and **no tumor**. It uses TensorFlow and is trained on a public MRI dataset available on Kaggle.

---

## 📁 Dataset

The dataset used is from [Kaggle - Brain Tumor Classification MRI](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri). It should be structured like:

brain-tumor-mri-dataset/
├── Training/
│ ├── glioma/
│ ├── meningioma/
│ ├── notumor/
│ └── pituitary/
└── Testing/
├── glioma/
├── meningioma/
├── notumor/
└── pituitary/


Place the dataset in `/kaggle/input/brain-tumor-mri-dataset` or adjust the `DATA_DIR` in the code accordingly.

---

## 🧠 Model Overview

- Input size: 224×224 grayscale MRI scans
- Data Augmentation: rotation, shift, shear, zoom, brightness, horizontal flip
- Residual CNN architecture
- Regularization: Dropout, L2
- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Metrics: Accuracy, Precision, Recall, AUC
- Learning Rate: Cosine annealing with warm-up

---

## 🚀 How to Run

To train and evaluate the model, run the script or notebook:

```bash
python brain_tumor_classifier.py
Or open and run the notebook:

bash

jupyter notebook brain_tumor_classifier.ipynb
Make sure the dataset path in the code matches your local/Kaggle directory.

✅ Evaluation Metrics
The model is evaluated using:

Accuracy

Precision

Recall

AUC Score

Confusion Matrix

Classification Report

📦 Requirements
Install dependencies with:

bash
pip install -r requirements.txt
Minimal requirements.txt:

bash
tensorflow>=2.10
matplotlib
numpy
scikit-learn
seaborn
ipykernel

📁 Project Structure

brain-tumor-classifier/
├── brain_tumor_classifier.ipynb      
├── brain_tumor_classifier.py         
├── README.md
├── requirements.txt
└── models/
    └── best_model.keras    

           
🖼️ Sample Outputs
After training, the notebook/script displays:

Training vs. Validation Accuracy/Loss curves

Confusion Matrix

Classification Report


🤝 Acknowledgments

Dataset: Kaggle - Brain Tumor Classification (MRI)
Frameworks: TensorFlow, Keras, scikit-learn, matplotlib, seaborn


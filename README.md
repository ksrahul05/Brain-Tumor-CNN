# Brain Tumor Detection using Custom CNN

This repository contains a Jupyter Notebook that implements a **Brain Tumor Detection** model using a **Custom Convolutional Neural Network (CNN)**.

## 📌 Features
- **Deep Learning Approach**: Uses a custom CNN for classification.
- **Dataset**: MRI scan images (tumor vs. non-tumor).
- **Visualization**: Data augmentation, training loss, and accuracy graphs.
- **Evaluation Metrics**: Accuracy, Precision, Recall, and Confusion Matrix.

## 📂 Files in this Repository
- `Brain_Tumor_Detection.ipynb` → The Jupyter Notebook containing model training and evaluation.
- `README.md` → Documentation.

## 🔧 Setup Instructions

1. **Clone the Repository**
   ```sh
   git clone https://github.com/ksrahul05/Brain-Tumor-CNN.git
   cd Brain-Tumor-CNN


Install Dependencies

sh
Copy
Edit
pip install -r requirements.txt
(If requirements.txt is missing, install manually: TensorFlow, Keras, OpenCV, NumPy, Matplotlib, Pandas.)

Open Jupyter Notebook

sh
Copy
Edit
jupyter notebook
Then, open Brain_Tumor_Detection.ipynb.

🧠 Model Overview
Custom CNN Architecture: Consists of convolutional, max-pooling, and fully connected layers.
Optimizer: Adam
Loss Function: Categorical Cross-Entropy
Activation Functions: ReLU and Softmax
🖼️ Dataset
Used an MRI dataset containing brain tumor images.
Dataset should be structured into Tumor and Non-Tumor folders.
🚀 Running the Notebook
Training: Execute all cells in Brain_Tumor_Detection.ipynb.
Testing: Load a trained model and classify new MRI images.

# 🧠 Brain Tumor Classifier

A deep learning-based classifier for identifying brain tumors (Glioma, Meningioma, Pituitary) from MRI scans using CNNs.

## 📌 Features
- Multi-class classification using CNN
- Works with MRI images in JPG format
- Uses BraTS dataset with 3 tumor types
- Includes preprocessing, model training, and evaluation
- Visual results (confusion matrix, prediction examples)

## 🛠️ Tech Stack
- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib

## 📂 Dataset
- Format: `.jpg`
- Classes: `glioma`, `meningioma`, `pituitary`
- Structure:

data/ ├── train/ │   ├── glioma/ │   ├── meningioma/ │   └── pituitary/ └── test/ ├── glioma/ ├── meningioma/ └── pituitary/

## 🚀 How to Run
1. Clone the repository:
 ```bash
 git clone https://github.com/yourusername/brain-tumor-classifier.git
 cd brain-tumor-classifier

2. Install dependencies:

pip install -r requirements.txt


3. Train the model:

python train.py


4. Test the model:

python predict.py --image test.jpg



🧠 Model Architecture

Custom CNN with Conv2D, MaxPooling, Dropout, Dense layers

Trained with categorical_crossentropy

Optimizer: Adam


📊 Results

Accuracy: ~92% on validation set

Metrics: Classification report + Confusion Matrix


📸 Sample Prediction



📌 Future Improvements

Add Grad-CAM for visual explanation

Deploy as a web app using Streamlit or Flask


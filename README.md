# ✋ Hand Gesture Recognition using CNN  
### SkillCraft Technology – Machine Learning Internship (Task 04)

This project is submitted as **Task 04** of my **Machine Learning Internship at SkillCraft Technology**.  
It implements a **Hand Gesture Recognition system** using a **Convolutional Neural Network (CNN)** trained on the **LeapGestRecog dataset**.

The system supports **image-based gesture prediction**, **batch inference (multiple images at once)**, and **confidence visualization in a single combined dashboard**.

---

## 📌 Project Highlights
- CNN-based hand gesture classification
- Trained on the LeapGestRecog dataset
- Supports selecting **multiple images simultaneously**
- Displays **all predictions in one combined graph**
- Confidence percentage shown directly on bars
- Clean inference-only pipeline (no retraining required)

---

## 📂 Dataset
- **Dataset Name:** LeapGestRecog
- **Source:** Kaggle  
- **Link:** https://www.kaggle.com/datasets/gti-upm/leapgestrecog
- The dataset contains grayscale hand gesture images organized by subjects and gesture classes.
- Due to size constraints, the dataset is **not uploaded** to this repository.

---

## 🧠 Model Architecture
- Convolutional layers for feature extraction  
- MaxPooling layers for dimensionality reduction  
- Fully connected Dense layers  
- Softmax output layer for multi-class classification  

---

## 🚀 Features Implemented
✔ Model training and saving  
✔ Accuracy and loss visualization  
✔ Image-based gesture prediction  
✔ Batch inference (multiple images at once)  
✔ Combined visualization dashboard:
- Input images
- Predicted gesture labels
- Confidence bars with percentage values

---

## 🖼️ Output Visualization
The prediction script generates a **single combined output window** where:
- Each row corresponds to one selected image
- Left: input hand gesture image
- Right: predicted gesture with confidence bar

Screenshots of outputs are available in the `output/` folder.

---

## 📁 Repository Structure
SCT-ML-4/
│
├── hand_gesture_recognition.py
├── image_gesture_prediction.py
├── hand_gesture_model.h5
├── gesture_labels.txt
├── requirements.txt
├── README.md
└── output/
├── training_accuracy.png
└── combined_prediction_output.png

---

## 🛠️ Tech Stack
- Python  
- TensorFlow / Keras  
- OpenCV  
- NumPy  
- Matplotlib  
- Tkinter  

---

## ▶️ How to Run

### 1️⃣ Train the model
python hand_gesture_recognition.py
--------
2️⃣ Predict gestures from images

python image_gesture_prediction.py

----------
📈 Results
---------------------

Achieved high accuracy on validation data

Predictions are displayed with interpretable confidence values

The system effectively classifies multiple gesture images in one run


# Breast Cancer Classification using LightGBM 🎗️

This project applies **LightGBM Classifier** on the popular **Breast Cancer dataset** from scikit-learn to predict whether a tumor is **malignant** or **benign**.  

---

## 📌 Project Overview
- Dataset: Breast Cancer Wisconsin dataset (`sklearn.datasets`)
- Task: **Binary Classification** (Malignant = 0, Benign = 1)
- Model: LightGBM Classifier (`LGBMClassifier`)
- Evaluation Metrics: Accuracy, Confusion Matrix, Precision, Recall, F1-score

---

## 📂 Dataset
The dataset contains **569 samples** with 30 features describing tumor cell nuclei properties, such as:
- Mean radius
- Texture
- Smoothness
- Compactness
- Symmetry, etc.  

Target classes:
- `0` → Malignant  
- `1` → Benign  

---

## ⚙️ Model
```python
from lightgbm import LGBMClassifier

model = LGBMClassifier(
    n_estimators=300,
    max_depth=6,
    num_leaves=3,
    objective='binary',
    random_state=42
)

model.fit(X_train, y_train)
📊 Results
Evaluation on the test set:

Accuracy: 97.36%

Confusion Matrix:

[[41  2]
 [ 1 70]]
Classification Report:

Class	Precision	Recall	F1-score	Support
0 (Malignant)	0.98	0.95	0.96	43
1 (Benign)	0.97	0.99	0.98	71

Macro Avg F1-score: 0.97

🔑 Key Learnings
LightGBM is highly effective for binary classification tasks with tabular medical data.

The model achieved high precision and recall, which is crucial for cancer detection.

Proper hyperparameter tuning (n_estimators, max_depth, num_leaves) impacts performance.

🚀 How to Run
Clone this repo:


git clone https://github.com/ishanegi5/breast_cancer_lightgbm_classifier.git
cd breast_cancer_lightgbm_classifier
Install requirements:


pip install -r requirements.txt
Run the Jupyter Notebook or Python script.

📦 Requirements
Python 3.x

scikit-learn

lightgbm

pandas

numpy

Install all with:


pip install lightgbm scikit-learn pandas numpy
👩‍💻 Author
Isha Negi

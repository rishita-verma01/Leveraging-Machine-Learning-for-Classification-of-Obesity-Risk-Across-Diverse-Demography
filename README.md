# 🧠 Leveraging Machine Learning for Classification of Obesity Risk

<p align="center">
  <img src="https://ieeeusa.org/wp-content/uploads/2020/06/XPDL-601.jpg" width="200" alt="IEEE Logo" />
</p>

🎓 **Published in:** IEEE InCACCT 2025 (Delhi Section)  
📚 **Library:** [IEEE Xplore](https://ieeexplore.ieee.org/document/11011446)  
📅 **Date:** March 2025  
📊 **Accuracy Achieved:** 98%   
📍 **Institution:** Thapar Institute of Engineering & Technology

---

## 📌 Abstract

Obesity is considered a global public health emergency due to its high risk of several chronic disorders, such as diabetes, hypertension, and cardiovascular disease. Due to the complexity of early risk prediction, obesity requires early interventions or classifications of individualized health management. This study demonstrates the implementation of several ML techniques to provide a comprehensive cross-frame study on the prediction of obesity risk. Application of multiple models, namely, LR (Logistic Regression), KNN (K-Nearest Neighbor), DTC (Decision Tree Classifier), GB (Gradient Boosting), MLP (Multiperceptron Network), and FNN (FeedForward Neural Network), are employed to check the performance on three benchmark datasets. The results showed that the accuracy of each model varied in the predictions, underlining both the benefits and the drawbacks of each approach in different scenarios. This study aims to develop more useful tools in clinical and preventive health, as it gives insight into comparing complex neural networks to conventional machine learning algorithms to predict the risk of obesity. The gradient boosting algorithm achieved the highest accuracy in all data sets, with a precision of 95% for Dataset 1 and 98% for both Dataset 2 and Dataset 3. This work underscores the potential of machine learning in public health and provides a foundation for policymakers and healthcare professionals to develop personalized and preventive strategies to combat obesity.

---

## 🧪 Methodology

### 1. 📥 Data Sourcing

Three benchmark datasets from Kaggle were used for obesity risk prediction:

- **Dataset 1** includes:
  - Gender, Age, Height, Weight
  - Family history of obesity
  - FAVC (Frequent consumption of high-calorie food)
  - FCVC (Frequency of vegetable intake)
  - NCP (Number of meals), CAEC (Snacking habits)
  - SMOKE, CH2O (Water intake), SCC (Calorie monitoring)
  - FAF (Physical activity), TUE (Technology usage)
  - CALC (Alcohol consumption), MTRANS (Transportation mode)

- **Dataset 2** contains:
  - ID, Age, Gender, Height, Weight, BMI, and Obesity Label

- **Dataset 3** includes:
  - Age, Gender, Height, Weight, BMI, Physical Activity Level, Obesity Category

All datasets were filtered and cleaned for consistency and usability during model training.

---

### 2. 🔧 Data Preprocessing

- Handled missing values using imputation or removal
- Standardized/normalized continuous variables
- Encoded categorical features using **Label Encoding** or **One-Hot Encoding**
- Split datasets into **80% training** and **20% testing**

---

### 3. 🧠 Cross-Framework Analysis

Multiple ML algorithms were applied across different frameworks to ensure robust and comparable results:

- **Models Used:**
  - Logistic Regression (LR)
  - K-Nearest Neighbors (KNN)
  - Decision Tree Classifier (DTC)
  - Gradient Boosting (GB)
  - Multilayer Perceptron (MLP)
  - Feedforward Neural Network (FNN)

- **Frameworks Used:**
  - `Scikit-learn`, `TensorFlow/Keras`, and `PyTorch`

- **Evaluation Metrics:**
  - Accuracy
  - Recall
  - F1-score

This cross-framework approach provided greater reproducibility and ensured fairness in model comparison.

---

### 4. 🧪 Model Training & Validation

- Used **Random Search** with **Cross-Validation** for hyperparameter tuning
- Neural networks were trained using:
  - Early stopping
  - Learning rate scheduling
  - Batch normalization
- Applied **K-Fold Cross-Validation** to evaluate model performance and stability




---

## 📈 Results

## 📊 Model Performance Results

### 📁 Dataset 1

| Model                          | Recall | F1-Score | Accuracy |
|-------------------------------|--------|----------|----------|
| Logistic Regression (LR)      | 0.84   | 0.80     | 0.87     |
| K-Nearest Neighbors (KNN)     | 0.76   | 0.75     | 0.82     |
| Decision Tree Classifier (DTC)| 0.96   | 0.96     | 0.93     |
| 🌟 Gradient Boosting (GB)     | **0.98** | **0.97** | **0.95** |
| Multilayer Perceptron (MLP)   | 0.92   | 0.91     | 0.92     |
| Feedforward Neural Network (FNN)| 0.92 | 0.91     | 0.92     |

---

### 📁 Dataset 2

| Model                          | Recall | F1-Score | Accuracy |
|-------------------------------|--------|----------|----------|
| Logistic Regression (LR)      | 0.86   | 0.86     | 0.86     |
| K-Nearest Neighbors (KNN)     | 0.86   | 0.86     | 0.86     |
| 🌟 Decision Tree Classifier (DTC) | **0.98** | **0.98** | **0.98** |
| 🌟 Gradient Boosting (GB)     | **0.98** | **0.98** | **0.98** |
| Multilayer Perceptron (MLP)   | 0.72   | 0.74     | 0.72     |
| Feedforward Neural Network (FNN)| 0.68 | 0.67     | 0.68     |

---

### 📁 Dataset 3

| Model                          | Recall | F1-Score | Accuracy |
|-------------------------------|--------|----------|----------|
| Logistic Regression (LR)      | 0.96   | 0.96     | 0.96     |
| K-Nearest Neighbors (KNN)     | 0.86   | 0.85     | 0.86     |
| 🌟 Decision Tree Classifier (DTC) | **0.98** | **0.98** | **0.98** |
| 🌟 Gradient Boosting (GB)     | **0.98** | **0.98** | **0.98** |
| Multilayer Perceptron (MLP)   | 0.95   | 0.95     | 0.95     |
| Feedforward Neural Network (FNN)| 0.80 | 0.78     | 0.79     |

---

### 🏆 Summary

> ✅ **Gradient Boosting** consistently outperformed other models across all datasets with the highest metrics.  
> 🧠 **MLP** and **FNN** neural models performed competitively but slightly under tree-based models.  
> 📉 **Logistic Regression** and **KNN** showed decent results but lacked consistency across datasets.



---

## 📄 Citation

If you use this work, please cite it as:

```bibtex
@INPROCEEDINGS{11011446,
  author={Verma, Rishita and Muhuri, Samya},
  booktitle={2025 3rd International Conference on Advancement in Computation & Computer Technologies (InCACCT)}, 
  title={Leveraging Machine Learning for Classification of Obesity Risk Across Diverse Demography}, 
  year={2025},
  volume={},
  number={},
  pages={284-289},
  keywords={Obesity;Adaptation models;Accuracy;Machine learning algorithms;Nearest neighbor methods;Predictive models;Boosting;Data models;Decision trees;Public healthcare;Obesity;Machine Learning Algorithms;Neural Networks;Predictive Modeling;Health Informatics},
  doi={10.1109/InCACCT65424.2025.11011446}}


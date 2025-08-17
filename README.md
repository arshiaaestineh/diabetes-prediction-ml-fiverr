# 🩺 Diabetes Prediction with Machine Learning

This project uses the **Pima Indians Diabetes Dataset** to build and compare multiple machine learning models for predicting whether a patient is likely to have diabetes.  
The goal is to evaluate different algorithms and identify the most effective one for this healthcare classification problem.

---

## 📊 Dataset
- **Source**: Pima Indians Diabetes Dataset (UCI / Kaggle)  
- **Features**: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age  
- **Target**: Outcome (1 = Diabetic, 0 = Non-diabetic)  
- **Preprocessing Steps**:
  - Handling invalid zeros by replacing them with median values  
  - Feature scaling with StandardScaler  
  - Train-test split for model evaluation  

---

## 🧠 Models Used
The following models were implemented and compared:
1. **Logistic Regression**  
2. **Decision Tree Classifier**  
3. **Random Forest Classifier**  
4. **Support Vector Machine (SVM)**  
5. **XGBoost Classifier**

---

## ⚡ Results
Performance metrics (Accuracy, Precision, Recall, F1-score) were calculated for all models.  

| Model                | Accuracy | Precision | Recall  | F1-score |
|-----------------------|----------|-----------|---------|----------|
| Random Forest         | 0.779    | 0.717     | 0.611   | 0.660    |
| XGBoost              | 0.759    | 0.674     | 0.611   | 0.641    |
| SVM                   | 0.740    | 0.653     | 0.556   | 0.600    |
| Logistic Regression   | 0.708    | 0.600     | 0.500   | 0.545    |
| Decision Tree         | 0.682    | 0.553     | 0.481   | 0.515    |

📌 **Best Model (Current):** **Random Forest Classifier** with the highest F1-score.

---

## 📈 Visualization
- Correlation Heatmap  
- Boxplots for feature distributions  
- Confusion Matrices for each model  
- Bar chart comparing model performance  

---

## 🚀 Future Work
- Hyperparameter tuning with GridSearchCV / RandomizedSearchCV  
- Feature engineering and selection  
- Handling class imbalance with SMOTE  
- Ensemble and stacking models  
- Trying Neural Networks (Deep Learning)  
- Model interpretability with SHAP / LIME  
- Deployment with **Streamlit / Flask**  

---

📝 Author
👤 Developed by Arshia
💡 Machine Learning Enthusiast | Kaggle Learner | Freelancer
email : arshiaestineh2005@icloud.com
GitHub : arshiaaestineh


⭐ If you like this project, don’t forget to star the repo! ⭐
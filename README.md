# Student Performance Prediction  

## 📌 Project Overview  
This project uses machine learning to predict students' **final grades (G3)** based on various factors such as previous grades, study time, and family background. We compare different models and apply **regularization** to prevent overfitting.

---

## 📊 Dataset  
The dataset is from the **UCI Machine Learning Repository** and contains student information from **Mathematics and Portuguese courses**. Features include:  
- **Academic Factors:** Previous grades (`G1`, `G2`), study time, failures.  
- **Personal Factors:** Age, address type, internet access.  
- **Family & Social Factors:** Parental education, school support, activities.  

The target variable is **G3 (Final Grade)**.

---

## 🏗️ Project Structure  

/student-performance-prediction

├── data/ # Raw and processed datasets  
├── notebooks/ # Jupyter notebooks for EDA and modeling  
├── src/ # Python scripts for preprocessing and model training  
├── results/ # Saved model performance metrics and visualizations  
├── README.md # Project documentation  


---

## ⚙️ Model Training & Evaluation  
Trained **Linear Regression** models with:  
1. **Subset of Features (Default SGDRegressor)**  
2. **Custom Gradient Descent Implementation**  
3. **All Features with Elastic Net Regularization** (Final Model)  

### **Model Comparison**  
| Model | Train MSE | Train RMSE | Test MSE | Test RMSE |
|--------|------------|------------|------------|------------|
| **SGDRegressor (All Features & Regularization)** | **0.1494** | **0.3865** | **0.2083** | **0.4564** |
| **SGDRegressor (Subset of Features)** | 0.1654 | 0.4067 | 0.2200 | 0.4691 |
| **Custom Gradient Descent (Subset, No Regularization)** | 0.1776 | 0.4214 | 0.2393 | 0.4892 |

### **Key Takeaways:**  
- The **regularized model (Elastic Net) achieved the lowest error** and generalizes best.  
- The **custom gradient descent model performed well** but was slightly outperformed by optimized SGDRegressor.  
- **Regularization helped reduce overfitting**, as shown by the smaller train-test error gap.  

---

## 🔧 How to Run  
1. **Clone the Repository**  
   ```bash
   git clone https://github.com/Abdulrasheed-Yusuf/student-performance-prediction.git
   cd student-performance-prediction
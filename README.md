# 🏦 Loan Default Risk Prediction

An end-to-end machine learning project and an explainable AI web app that predicts the likelihood of loan default using classification models (Logistic Regression and XGBoost) and explains decisions with SHAP. Built for credit risk assessment and SARB compliance

---

**Live Demo:** https://loan-default-risk-predictions-project-ncmzz9wdzpvxhrhxhnrpms.streamlit.app

## 📌 Project Overview
This project analyzes loan applicant data to predict the risk of default. It includes data preprocessing, EDA, feature engineering, model training, and a deployment-ready app.
This app helps loan officers assess credit risk in real-time. It takes applicant financial data, runs it through a calibrated XGBoost model, and returns:
1. **Probability of Default**
2. **Decision**: APPROVED / REVIEW / REJECTED based on a 0.32 threshold
3. **SHAP Waterfall Plot**: Explains the top factors driving the decision for audit/compliance

The goal is transparency. Every decline can be explained to the applicant and regulator.

---

### **Key Features**
- **XGBoost Model**: Trained on historical loan data with calibration for accurate probabilities
- **Explainable AI**: SHAP waterfall plots show exactly why an application was approved or rejected
- **Interactive UI**: Built with Streamlit for easy data entry
- **Download Report**: Export prediction + input data as CSV
- **SARB Compliance Focus**: Provides reasons for decline

---

## 🛠️ Tech Stack
- **Language**: Python 3.x
- **Libraries**: pandas, numpy, scikit-learn, matplotlib, seaborn, Logistic Regression, XGBoost and SHAP
- **App**: Streamlit
- **Notebooks**: Jupyter
---

### **How to Run Locally**
1. Clone the repo
```bash```
git clone https://github.com/luvhengork-design/Loan-Default-Risk-Predictions-Project.git
cd Loan-Default-Risk-Predictions-Project
---

### **How to use**
1. Enter applicant financial details in the sidebar
2. Click `Score` to get probability + decision
3. Review the SHAP plot to see top risk drivers
4. Download the result as CSV
---

### **Screenshots**
![App UI](assets/app_screenshot.png)
![App UI](assets/app2_screenshot.png)

---

### *Model Details*
- **Models Used**: Logistic Regression and XGBoost with Calibration
- **Decision Threshold**: 0.32
- **Explainability**: SHAP for regulatory transparency

---

### **Author**
Built by Rotshila Luvhengo  
LinkedIn: [Rotshila Luvhengo](https://www.linkedin.com/in/rotshilaluvhengo-438ab138a) | GitHub: [luvhengork-design](https://github.com/luvhengork-design)




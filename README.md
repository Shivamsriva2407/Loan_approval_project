Loan Approval Prediction Project

This project focuses on building a machine learning pipeline to predict loan approval outcomes using financial, demographic, and credit-history features.
The dataset used was imbalanced, requiring the application of SMOTE to achieve balanced class representation. The workflow includes end-to-end preprocessing,
feature engineering, statistical testing, model training, and model evaluation.

📌 Project Overview
This project implements a complete end‑to‑end machine learning pipeline for loan approval prediction, including preprocessing, feature engineering, imbalanced data handling, statistical testing, model training, and evaluation. The objective was to develop a classification model for predicting whether a loan application would be approved.
The project covers data analysis, cleaning, engineering, handling imbalance, and evaluating multiple machine learning models.

📂 Dataset Summary
The dataset contains: - 
Numerical Features: person_age, person_income, person_emp_exp, loan_amnt, loan_int_rate, loan_percent_income, cb_person_cred_hist_length. 
Categorical Features: Various demographic and financial categories. - Target Variable: loan_status (0 = Rejected, 1 = Approved).

🧹 Data Preprocessing
•	Loaded and inspected dataset (shape, info, duplicates).
•	Removed redundant columns such as person_Monthly_salary.
•	Treated skewness via transformations:
o	Log transform on loan_amnt.
o	Square-root transform on loan_percent_income.
o	Yeo-Johnson PowerTransformer on numerical features.
•	Scaled numerical features using StandardScaler.
•	Label-encoded categorical features.

📊 Exploratory Data Analysis
•	Conducted univariate analysis with KDE plots and boxplots.
•	Performed bivariate analysis comparing numerical features against loan_status.
•	Identified outliers and addressed skewed distributions.

📐 Feature Engineering & Selection
•	Applied transformations to normalize distributions.
•	Used Chi-Square Test to evaluate categorical feature significance.
•	Dropped features that showed low relevance or correlation.

⚖️ Handling Imbalanced Classes
The dataset had significant class imbalance. To address this: - Applied SMOTE (Synthetic Minority Oversampling Technique) on training data. - Improved recall and minority-class representation.

🤖 Model Training
Two models were trained: - Support Vector Machine (SVM) — performed well after scaling and transformations. - Logistic Regression — offered interpretability and stable baseline performance.

📈 Model Evaluation
Evaluated using: - Accuracy - Precision - Recall - F1-score - Confusion Matrix - ROC Curve and AUC score
Improvements were observed post-SMOTE, particularly in recall for the minority class.

📝 Key Outcomes
•	Successfully built a complete ML pipeline from preprocessing to evaluation.
•	Addressed real-world challenges like imbalanced data and skewed distributions.
•	Compared model outcomes and identified improvements due to SMOTE and transformations.
📎 Summary
This project demonstrates skills in: - Data preprocessing & cleaning - Handling imbalanced datasets - Feature engineering - Categorical analysis with statistical tests - 
Evaluating ML models with industry-standard metrics - Building interpretable and scalable ML pipelines


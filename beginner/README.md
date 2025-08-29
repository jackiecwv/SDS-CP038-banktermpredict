# 🟢 Beginner Track

## 🎯 Objectives

The Beginner Track focuses on building a **traditional ML workflow** to predict whether a client subscribes to a term deposit. You will:

* Perform exploratory data analysis (EDA)
* Clean and preprocess the dataset
* Engineer and encode features
* Train baseline machine learning models
* Evaluate results with standard metrics
* Deploy your final model with a **Streamlit app**

---

## 📅 Weekly Breakdown

### ✅ Week 1: Setup + EDA + Feature Engineering

* Set up your project environment and install dependencies
* Load the dataset and explore its structure
* Handle missing values, duplicates, and data types
* Analyze distributions of key features (age, balance, campaign, etc.)
* Check for target class imbalance (`y` variable)
* Summarize insights with visualizations (histograms, boxplots, correlation heatmaps)
* Engineer new features (e.g., campaign frequency, time since last contact)
* Handle class imbalance with techniques like SMOTE or class weights

### ✅ Week 2 + 3: Data Preprocessing + Model Development

* Encode categorical variables (label encoding, one-hot encoding)
* Scale numerical features with StandardScaler or MinMaxScaler
* Split dataset into training, validation, and test sets
* Train baseline models: Logistic Regression, Decision Tree, Random Forest
* Experiment with boosting methods (XGBoost, LightGBM)
* Track experiments with **MLflow**
* Evaluate models with Accuracy, Precision, Recall, F1-score, ROC-AUC

### ✅ Week 4: Model Tuning + Deployment

* Tune key hyperparameters using GridSearchCV or RandomizedSearchCV
* Validate models with cross-validation
* Select the best-performing model based on metrics
* Save final model pipeline with preprocessing steps included
* Build a **Streamlit app** that accepts client features as inputs
* Predict whether the client will subscribe to a term deposit
* Deploy the app on **Streamlit Community Cloud**

---

## 🛠️ Technical Requirements

* **Data Handling & Visualization**: `pandas`, `numpy`, `matplotlib`, `seaborn`
* **Machine Learning**: `scikit-learn`, `xgboost`, `lightgbm`, `mlflow`
* **Deployment**: `streamlit`

---

At the end of this track, you will have built and deployed a **machine learning model** capable of predicting client subscription likelihood with a simple web interface.

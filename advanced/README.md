# 🔴 Advanced Track

## 🎯 Objectives

The Advanced Track is designed for participants who want to apply **deep learning techniques** on tabular data and explore model explainability. You will:

* Perform EDA with attention to class imbalance and feature interactions
* Prepare data for deep learning (embeddings, normalization, batching)
* Build and train Feedforward Neural Networks (FFNNs)
* Apply hyperparameter tuning and regularization
* Integrate explainability methods like SHAP or LIME
* Deploy your model via Streamlit, Docker, or API-based services

---

## 📅 Weekly Breakdown

### ✅ Week 1: Setup + EDA + Preprocessing

* Perform EDA similar to Beginner Track, but focus more on skewed distributions and multicollinearity
* Encode categorical variables (integer encoding for embeddings)
* Normalize/scale numeric features
* Split dataset into train/val/test with stratification
* Convert into PyTorch `DataLoader` or TensorFlow `tf.data` pipelines

### ✅ Week 2 + 3: Neural Network Design + Model Development

* Build a baseline **Feedforward Neural Network (FFNN)** with hidden layers
* Include ReLU activations, Dropout, and Batch Normalization
* Train using Binary Cross-Entropy loss and Adam optimizer
* Evaluate with Accuracy, Precision, Recall, F1-score, and AUC
* Track all experiments with **MLflow**
* Apply regularization and early stopping
* Tune architecture: number of layers, neurons, dropout rates, learning rate, batch size
* Use schedulers for learning rate adjustments

### ✅ Week 4: Deployment

* Save and package best model

#### Deployment Options:

* 🟢 **Streamlit**: Build a UI similar to Beginner Track
* 🟡 **Docker + Hugging Face Spaces**: Containerize and deploy Streamlit app
* 🔴 **API-based Deployment**: Use Flask or FastAPI, containerize with Docker, and deploy on Railway/Render/Fly.io/GCP Cloud Run

---

## 🛠️ Technical Requirements

* **Data Handling & Visualization**: `pandas`, `numpy`, `matplotlib`, `seaborn`
* **Deep Learning**: `tensorflow` or `pytorch`, `mlflow`
* **Explainability**: `shap`, `lime`
* **Deployment**: `streamlit`, `docker`, `fastapi` or `flask`

---

## ❓ Key Questions to Answer

1. Can deep learning methods outperform traditional ML models on this dataset?
2. Which features contribute most to predictions according to explainability methods?
3. How does tuning network architecture affect performance?
4. What are the trade-offs between different deployment methods (Streamlit, Docker, API)?

---

At the end of this track, you will have built, explained, and deployed a **deep learning model** for predicting term deposit subscriptions with modern deployment practices.

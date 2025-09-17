# 🔴 Advanced Track

## ✅ Week 1: Exploratory Data Analysis (EDA)

---

### 📦 1. Data Integrity & Structure

Q: Did you find any missing, duplicate, or incorrectly formatted entries in the bank marketing dataset?  
A:  

Q: Are all data types appropriate for their features (e.g., numeric, categorical)?  
A:  

Q: Did you identify any constant, near-constant, or irrelevant features?  
A:  

---

### 🎯 2. Target Variable Assessment

Q: What is the distribution of the target variable (e.g., `deposit`)?  
A:  

Q: Is there a class imbalance? If so, how significant is it?  
A:  

Q: How could this imbalance affect your choice of evaluation metrics or modeling strategy?  
A:  

---

### 📊 3. Feature Distribution & Quality

Q: Which numerical features are skewed or contain outliers?  
A:  

Q: Did any features contain unrealistic or problematic values?  
A:  

Q: What transformation methods (if any) might improve these feature distributions?  
A:  

---

### 📈 4. Feature Relationships & Patterns

Q: Which categorical features (e.g., `job`, `marital`, `education`) show visible patterns in relation to the target variable?  
A:  

Q: Are there any strong pairwise relationships or multicollinearity between features?  
A:  

Q: What trends or correlations stood out during your analysis?  
A:  

---

### 🧰 5. EDA Summary & Preprocessing Plan

Q: What are your 3–5 biggest takeaways from EDA?  
A:  

Q: Which features will you scale, encode, or exclude in preprocessing?  
A:  

Q: What does your cleaned dataset look like (rows, columns, shape)?  
A:  

---

## ✅ Week 2: Feature Engineering & Deep Learning Prep

---

### 🏷️ 1. Categorical Feature Encoding

Q: Which categorical features in the dataset have more than two unique values?  
A:  

Q: Apply integer-encoding to these high-cardinality features. Why is this strategy suitable for a subsequent neural network with an embedding layer?  
A:  

Q: Display the first 5 rows of the transformed data to show the new integer labels.  
A:  

---

### ⚖️ 2. Numerical Feature Scaling

Q: Which numerical features did your EDA from Week 1 suggest would benefit from scaling?  
A:  

Q: Apply a scaling technique to these features. Justify your choice of `StandardScaler` vs. `MinMaxScaler` or another method.  
A:  

Q: Show the summary statistics of the scaled data to confirm the transformation was successful.  
A:  

---

### ✂️ 3. Stratified Data Splitting

Q: Split the data into training, validation, and testing sets (e.g., 70/15/15). What function and parameters did you use?  
A:  

Q: Why is it critical to use stratification for this specific dataset?  
A:  

Q: Verify the stratification by showing the class distribution of the target variable in each of the three resulting sets.  
A:  

---

### 📦 4. Deep Learning Dataset Preparation

Q: Convert your three data splits into PyTorch `DataLoader` or TensorFlow `tf.data.Dataset` objects. What batch size did you choose and why?  
A:  

Q: To confirm they are set up correctly, retrieve one batch from your training loader. What is the shape of the features (X) and labels (y) in this batch?  
A:  

Q: Explain the role of the `shuffle` parameter in your training loader. Why is this setting important for the training set but not for the validation or testing sets?  
A:  

---

## ✅ Week 3: Neural Network Experimentation & Explainability

---

### 🧪 1. Neural Network Architecture & Training

Q: What neural network architecture did you use (layers, activations, dropout, batch normalization)?  
A:  

Q: How did you select and tune hyperparameters such as learning rate, batch size, and number of layers?  
A:  

Q: What regularization techniques did you apply, and how did they affect model performance?  
A:  

Q: How did you monitor and prevent overfitting during training (e.g., early stopping, validation curves)?  
A:  

---

### 📊 2. Experiment Tracking

Q: How did you track your deep learning experiments and results?  
A:  

Q: What insights did you gain from comparing different model runs using MLflow or similar tools?  
A:  

---

### 🧠 3. Model Evaluation

Q: Which evaluation metrics did you use to assess your neural network, and why?  
A:  

Q: How did your neural network's performance compare to baseline models?  
A:  

Q: What steps did you take to ensure the reliability and reproducibility of your results?  
A:  

---

### 🕵️ 4. Model Explainability

Q: What explainability methods (e.g., SHAP, LIME) did you apply to interpret your model's predictions?  
A:  

Q: What were the most important features according to your explainability analysis?  
A:  

Q: How did model explainability influence your understanding or trust in the model?  
A:  
# 🎬 Netflix Recommendation System

A Machine Learning project focused on building a personalized movie recommendation engine using **Singular Value Decomposition (SVD)** on the Netflix dataset.

---

## 🚀 Project Overview

This project involves building a recommendation system to predict user ratings for unseen movies.  
It follows a structured data science pipeline including data cleaning, exploratory data analysis (EDA), data filtering, model training, and evaluation.

---

## 🛠 Tech Stack & Libraries

* **Language:** Python  
* **Data Analysis:** Pandas, NumPy  
* **Visualization:** Seaborn, Matplotlib  
* **Machine Learning:** Scikit-Surprise (SVD)  
* **Model Evaluation:** RMSE (Root Mean Squared Error)  

---

## 📊 Key Features & Methodology

To ensure strong recommendation accuracy and reliability, the following techniques were implemented:

* **Data Cleaning:**  
  Handled missing values, structured Movie IDs properly, and removed inconsistent entries.

* **Exploratory Data Analysis (EDA):**  
  Used rating distribution analysis and bar plots to understand user behavior and movie popularity.

* **Data Pre-Filtering:**  
  * Removed movies rated below the 60th percentile benchmark.  
  * Removed inactive users with low rating counts.  
  * Improved model performance by reducing noise.

* **Model Building:**  
  * Implemented **Singular Value Decomposition (SVD)** for collaborative filtering.  
  * Trained model using Surprise library.  
  * Applied 3-fold cross-validation.

* **Model Evaluation:**  
  * Used **RMSE** to evaluate prediction performance.  
  * Lower RMSE indicates better prediction accuracy.

* **Personalized Recommendation Engine:**  
  * Predicted estimated ratings for unseen movies.  
  * Ranked movies based on predicted scores.  
  * Generated Top-5 recommendations for selected users.

---

## 📈 Model Performance

The model was validated using cross-validation techniques and evaluated using RMSE to ensure prediction consistency and generalization capability.

---

## 📁 Dataset

* Netflix Prize Dataset  
* `combined_data_1.csv`  
* `movie_titles.csv`

---

## ▶ How to Run the Project

```bash
pip install -r requirements.txt
python Netflix_project_by_intellipaat.py

🔮 Future Improvements

Implement Hybrid Recommendation System

Deploy using Streamlit Web App

Use full dataset for improved accuracy

Integrate Deep Learning-based recommendation

👨‍💻 Author

Mohit Khatri
Machine Learning & AI Enthusiast

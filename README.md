🎬 Netflix Recommendation System

A Machine Learning project focused on building a personalized movie recommendation system using Collaborative Filtering (SVD) on the Netflix dataset.

🚀 Project Overview

This project develops a recommendation engine that predicts user ratings for unseen movies using Singular Value Decomposition (SVD).

The system follows a structured data science pipeline including:

Data Cleaning

Exploratory Data Analysis (EDA)

Data Filtering & Benchmarking

Model Training & Cross Validation

Personalized Movie Recommendations

The goal is to simulate how real-world streaming platforms recommend content to users.

🛠 Tech Stack & Libraries

Language: Python

Data Analysis: Pandas, NumPy

Visualization: Matplotlib, Seaborn

Machine Learning: Scikit-Surprise (SVD)

Model Evaluation: RMSE (Root Mean Squared Error)

📊 Key Features & Methodology

To ensure accuracy and recommendation quality, the following techniques were implemented:

🔹 Data Cleaning

Handled missing values

Removed invalid entries

Structured Movie IDs correctly

🔹 Exploratory Data Analysis (EDA)

Analyzed rating distribution

Counted total movies, customers, and ratings

Visualized star rating frequency

🔹 Data Pre-Filtering

Removed movies with low rating counts

Removed inactive users

Applied 60th percentile benchmark filtering

🔹 Model Building

Implemented Singular Value Decomposition (SVD)

Trained model using Surprise library

Performed 3-fold cross validation

🔹 Model Evaluation

Used RMSE as performance metric

Evaluated prediction accuracy

🔹 Personalized Recommendations

Predicted estimated ratings

Generated Top 5 movie recommendations for specific users

📈 Model Performance

The model was evaluated using cross-validation with RMSE to measure prediction accuracy.

Lower RMSE indicates better predictive performance.

📁 Dataset

Netflix Prize Dataset

combined_data_1.csv

movie_titles.csv

🔮 Future Improvements

Implement Hybrid Recommendation System

Deploy as Web App using Streamlit

Use full dataset for improved accuracy

Add content-based filtering

📌 How to Run
pip install -r requirements.txt
python Netflix_project_by_intellipaat.py

💡 Project Highlights

✔ Real-world large dataset handling
✔ Matrix factorization implementation
✔ Collaborative filtering approach
✔ Scalable recommendation logic

👨‍💻 Author

Mohit Khatri
Aspiring Machine Learning & AI Engineer

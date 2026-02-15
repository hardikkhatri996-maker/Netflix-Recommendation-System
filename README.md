
Netflix Movie Recommendation System
A Machine Learning project that leverages Singular Value Decomposition (SVD) and Collaborative Filtering to predict user movie ratings and provide personalized recommendations using the Netflix Prize dataset.

🚀 Project Overview
This project focuses on building a scalable recommendation engine. It handles a large-scale sparse dataset by implementing strategic data trimming and matrix factorization techniques to discover latent features in user-movie interactions.

🛠️ Tech Stack & Libraries
Language: Python

Data Analysis: Pandas, NumPy

Visualization: Matplotlib, Seaborn

Machine Learning: Scikit-Surprise (SVD)

📊 Key Features & Methodology
To ensure high-quality recommendations and efficient computation, the following techniques were implemented:

Data Cleaning: Transformed raw, semi-structured text data into a queryable format by mapping User IDs to their corresponding Movie IDs and handling null values.

Exploratory Data Analysis (EDA): Analyzed the total pool of movies, customers, and ratings to understand dataset sparsity and rating distributions.

Feature Engineering & Trimming:

Movie Benchmarking: Filtered out movies with ratings below the 60th percentile to focus on statistically significant content.

User Benchmarking: Removed inactive users who provided fewer ratings than the 60th percentile to reduce noise.

Advanced Optimization:

Matrix Factorization: Applied Singular Value Decomposition (SVD) to identify hidden connections between user preferences and item characteristics.

Cross-Validation: Evaluated model performance using 3-fold cross-validation with Root Mean Square Error (RMSE) metrics.

📈 Model Performance
The system was evaluated based on its predictive accuracy for user ratings:

Algorithm: SVD (Collaborative Filtering)

Metric: RMSE (Root Mean Square Error)

Final Output: Successfully generated personalized Top-5 movie recommendations for specific User IDs.

📁 Repository Structure
Netflix_project_by_intellipaat.py: Main Python script containing the data processing and model pipeline.

combined_data_1.csv: The raw dataset containing user ratings.

movie_titles.csv: Mapping of Movie IDs to Titles and Release Years.

README.md: Project documentation.

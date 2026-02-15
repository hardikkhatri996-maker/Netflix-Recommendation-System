🎬 Netflix Movie Recommendation System (SVD)
An end-to-end Collaborative Filtering recommendation engine that processes the massive Netflix Prize dataset to predict user ratings and suggest top-tier content.

📌 Project Overview
The goal of this project is to overcome the "Cold Start" and "Sparsity" challenges in recommendation systems. By utilizing Singular Value Decomposition (SVD), the system identifies latent factors (hidden patterns) between users and movies to estimate how a user would rate a movie they haven't seen yet.

Key Features:
Data Cleaning & Wrangling: Transforming raw Netflix text files into structured DataFrames.

Dynamic Filtering: Implements a "Benchmark" system to remove inactive users and unpopular movies, ensuring high-quality model training.

SVD Implementation: Uses the Matrix Factorization technique to predict ratings.

Personalized Top-N Recommendations: Generates a ranked list of the top 5 movie suggestions for any specific User ID.

🛠️ Technical Stack
Language: Python

Data Manipulation: pandas, numpy

Visualization: matplotlib, seaborn

ML Framework: scikit-surprise (SVD, Dataset, Reader)

🚀 Workflow
1. Data Pre-processing
The raw dataset contains movie IDs embedded as rows followed by user ratings. The script:

Extracts Movie_Id from the rows and populates a new column.

Removes null values and corrects data types for memory efficiency.

2. Data Trimming (The Benchmark)
To improve model accuracy and reduce computational load:

Movie Benchmark: Drops movies with ratings below the 60th percentile.

Customer Benchmark: Drops users who have rated fewer movies than the 60th percentile.

3. Collaborative Filtering (SVD)
The model decomposes the User-Item matrix into three smaller matrices. These matrices represent:

User Preferences (e.g., preference for Sci-Fi or Drama).

Item Characteristics (e.g., movie genre or director).

The strength of these latent features.

4. Evaluation
The model is validated using Cross-Validation (3-fold) with RMSE (Root Mean Square Error) as the primary metric to ensure prediction accuracy.

📊 Results & Usage
To generate recommendations for a specific user (e.g., User 1331154):

The model calculates an Estimate_Score for all movies in the database.

Movies already seen by the user are filtered out.

The system returns the top 5 highest-scored movies.

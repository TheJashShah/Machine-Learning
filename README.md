# Machine-Learning
 All of my Machine Learning Codes.

# Simple-Linear-Regression
 Contains Simple Linear Regression on a placement.csv file, using Sklearn, and from scratch in Python and C.

# Projects
 ## 1. Student-Score-Predictor:
    extremely simple project done only for confidence and to get a hang of using a somewhat real-life dataset.
    R2_score: 0.988

 ## 2. Taxi-Price-Predictor:
    Used Linear regression on a taxi-price dataset, which required cleaning, manually encoding categorical columns, and min-max scaling numerical columns.
    R2_score: ~0.83

 ## 3. Wine-Quality-Predictor:
    Used a wine dataset that contains chemical details of wine samples to predict whether a wine is 'good' or 'bad'. (Regression performed badly, so used random-forest classifier).
    [
    Accuracy -> 0.765,
    Precision -> 0.76,
    Recall -> 0.83,
    F1 -> 0.8,
    ]

 ## 4. Titanic-Survival-Predictor:
     Used Logistic Regression to predict whether a person lived or died on titanic, dataset from the Kaggle competition.
     [
      Model metrics:
      Accuracy -> 0.80,
      Precision -> 0.803,
      Recall -> 0.706,
      F1 -> 0.75,
     ]
     {Kaggle:
         Submissions: 1
         Score : 0.76555
     }

   ## 5. Car-Price-Predictor:
       Used Linear and Random Forest Regression to find the price of a car. 
       [
         Random Forest R2_Score -> 0.93
         Linear R2_Score -> 0.85
       ]

   ## 6. Movie-Recommendation-System:
       Used Content-based filtering, text vectorization and cosine-similarity
       [
         Similarity-scores: 0.26 << x >> 0.37
       ]
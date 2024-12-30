# IMPORTS #
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# IMPORTS #

# CLASS #
class LinearRegression:
    def __init__(self):
        self.m = None
        self.b = None

    # TRAINING METHOD #
    def fit(self, X_train, Y_train):
        
        numerator = denominator = 0

        for i in range(X_train.shape[0]):

            numerator = numerator + ((X_train[i] - X_train.mean()) * (Y_train[i] - Y_train.mean()))
            denominator = denominator + ((X_train[i] - X_train.mean()) * (X_train[i] - X_train.mean()))

        self.m = numerator / denominator
        self.b = Y_train.mean() - (self.m * X_train.mean())
    # TRAINING METHOD #

    # RETURN METHODS #
    def predict(self, X_test):
        
        return (self.m * X_test + self.b)
    
    def coef_(self):
        return self.m
    
    def intercept_(self):
        return self.b
    # RETURN METHODS #

    # METRICS #
    def mean_absolute_error(self, Y_test, Y_predict):

        numerator = 0

        for i in range(len(Y_test)):
            numerator += abs(Y_test[i] - Y_predict[i])

        return numerator/len(Y_test)
    
    def mean_squared_error(self, Y_test, Y_predict):

        numerator = 0

        for i in range(len(Y_test)):
            numerator += ((Y_test[i] - Y_predict[i]) * (Y_test[i] - Y_predict[i]))

        return numerator/len(Y_test)
    
    def r2_score(self, Y_test, Y_predict):

        numerator = 0

        for i in range(len(Y_test)):

            numerator += ((Y_test[i] - Y_test.mean()) * (Y_test[i] - Y_test.mean()))

        SS_mean = numerator/len(Y_test)

        return (1 - (self.mean_squared_error(Y_test, Y_predict) / SS_mean))
    
    # METRICS #
    
# CLASS #

data = pd.read_csv('Simple-Linear-Regression/placement.csv')

X = data.iloc[:,0].values
Y = data.iloc[:,1].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

LR  = LinearRegression()
LR.fit(X_train, Y_train)

print(LR.coef_(), LR.intercept_())

'''
RESULT FROM SKLEARN - [0.55795197] -0.8961119222429144
RESULT FROM SCRATCH - 0.5579519734250721 -0.8961119222429152
'''
# PLOT WITH BEST FIT LINE #
plt.scatter(data['cgpa'], data['package'])
plt.plot(X_train, LR.predict(X_train))
plt.xlabel("CGPA")
plt.ylabel("Package(in LPA)")
plt.show()
# PLOT WITH BEST FIT LINE #

Y_predict = LR.predict(X_test)

print("MAE", LR.mean_absolute_error(Y_test, Y_predict))
print("MSE", LR.mean_squared_error(Y_test, Y_predict))
print("RMSE", (LR.mean_squared_error(Y_test, Y_predict) ** 0.5))
print("R2", LR.r2_score(Y_test, Y_predict))

r2 = LR.r2_score(Y_test, Y_predict)
print("ADJ R2", 1 - ((1 - r2) * (X_test.shape[0] - 1)) / (X_test.shape[0] - 1 - 1))

'''
SCRATCH: 
MAE 0.28847109318781733
MSE 0.1212923531349552
RMSE 0.34827051717731605
R2 0.7807301475103842
ADJ R2 0.7749598882343417

SKLEARN:
MAE 0.2884710931878175
MSE 0.12129235313495527
RMSE 0.34827051717731616
R2 SCORE 0.780730147510384
ADJ R2 0.7749598882343415
'''

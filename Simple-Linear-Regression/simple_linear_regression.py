import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv('Simple-Linear-Regression\placement.csv')

# matplotlib plot
plt.scatter(data['cgpa'], data['package'])
plt.xlabel('CGPA')
plt.ylabel('Package(in LPA)')
#plt.show()

X = data.iloc[:, 0:1]
Y = data.iloc[:, -1]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

lr = LinearRegression()

# training model
lr.fit(X_train, Y_train)

# line plot
plt.scatter(data['cgpa'], data['package'])
plt.plot(X_train, lr.predict(X_train))
plt.xlabel('CGPA')
plt.ylabel('Package(in LPA)')
#plt.show()

# slope and intercept
m = lr.coef_
b = lr.intercept_ 

print(m, b)

'''
y = mx + b
'''

y_predict = lr.predict(X_test)
print("MAE", mean_absolute_error(Y_test, y_predict))
print("MSE", mean_squared_error(Y_test, y_predict))
print("RMSE", np.sqrt(mean_squared_error(Y_test, y_predict)))
print("R2 SCORE", r2_score(Y_test, y_predict))


r2 =  r2_score(Y_test, y_predict)
print("ADJ R2", 1 - ((1 -  r2)*(X_test.shape[0] - 1)/(X_test.shape[0] - 1 - X_test.shape[1])))


'''
new_df = data.copy()
new_df['random'] = np.random.random(200)

new_df = new_df[['cgpa', 'random', 'package']]
x = new_df.iloc[:, 0:2]
y = new_df.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

lr = LinearRegression()
lr.fit(x_train, y_train)

Y_predict = lr.predict(x_test)

print("R2 SCORE", r2_score(y_test, Y_predict))


r2_ =  r2_score(y_test, Y_predict)
print("ADJ R2", 1 - ((1 -  r2_)*(x_test.shape[0] - 1)/(x_test.shape[0] - 1 - x_test.shape[1])))
'''

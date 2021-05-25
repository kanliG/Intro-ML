# TODO: Add import statements
import pandas as pd
import sklearn.preprocessing as polynomian
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Assign the data to predictor and outcome variables
# TODO: Load the data
train_data = pd.read_csv('data01.csv')
X = train_data['Var_X'].values.reshape(-1, 1)
y = train_data['Var_Y'].values
print(X.shape)
plt.scatter(X, y)
# Create legend.
plt.xlabel('X')
plt.ylabel('Y')
# plt.show()

# Create polynomial features
# TODO: Create a PolynomialFeatures object, then fit and transform the
# predictor feature
degree = 4
poly_feat = polynomian.PolynomialFeatures(degree=degree)
X_poly = poly_feat.fit_transform(X)
print(X_poly.shape)


# Make and fit the polynomial regression model
# TODO: Create a LinearRegression object and fit it to the polynomial predictor
# features
poly_model = LinearRegression().fit(X_poly, y)
print(poly_model.score(X_poly, y))
print(poly_model.get_params(deep=True))

# find curve
maxP = max(X)
minP = min(X)
step = 100
stepP = (maxP - minP)/step
arrayP = np.arange(start=minP, stop=maxP, step=stepP)
arrayP = arrayP.reshape(-1, 1)
poly_feat = polynomian.PolynomialFeatures(degree=degree)
arrayP_poly = poly_feat.fit_transform(arrayP)
print(arrayP_poly.shape)
predict = poly_model.predict(arrayP_poly)
plt.plot(arrayP_poly[:, 1], predict, 'r')
# Create legend.
plt.xlabel('X')
plt.show()

# Once you've completed all of the steps, select Test Run to see your model
# predictions against the data, or select Submit Answer to check if the degree
# of the polynomial features is the same as ours!
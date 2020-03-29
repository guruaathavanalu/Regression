# Data Preprocessing Template

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2:3].values


from sklearn.linear_model import LinearRegression
linearregressor=LinearRegression()
linearregressor.fit(X,y)
plt.scatter(X,y,color='red')
plt.plot(X,linearregressor.predict(X))
plt.title('linear_model')
plt.show()

from sklearn.preprocessing import PolynomialFeatures
polynomialregressor=PolynomialFeatures(degree=4)
X_poly=polynomialregressor.fit_transform(X)

linearregressor1=LinearRegression()
linearregressor1.fit(X_poly,y)


X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color='red')
plt.plot(X_grid,linearregressor1.predict(polynomialregressor.fit_transform(X_grid )))
plt.title('poly_model')
plt.show()

linearregressor1.predict(polynomialregressor.fit_transform([[6.5]]))
linearregressor.predict([[6.5]])

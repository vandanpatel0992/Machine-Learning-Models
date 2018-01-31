# Polynomial Regression

# Importing the libraries
import numpy as np #to use mathematics in the code
import matplotlib.pyplot as plt #to do visualization
import pandas as pd #to import/export data set and manage it

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values  #needs to be matrix therfore 1:2 instead of 1
y = dataset.iloc[:,2].values 

## Splitting the dataset into the Training and Test set
#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fit linear regression model
from sklearn.linear_model import LinearRegression
lin_regr = LinearRegression()
lin_regr.fit(X,y)

# Fit polynomial regression model
#transform X into its higher powers
from sklearn.preprocessing import PolynomialFeatures
polytransform= PolynomialFeatures(degree = 8) #predective curve fits better with increase in degrees
X_poly = polytransform.fit_transform(X)
#develope regressors
poly_regr = LinearRegression()
poly_regr.fit(X_poly,y)

# Visualize linear regression model
plt.scatter(X,y)
plt.plot(X,lin_regr.predict(X))
plt.title('Linear regression plot')
plt.xlabel('Employment level')
plt.ylabel('Salary')

# Visualize polynomial regression model
plt.scatter(X,y)
plt.plot(X, poly_regr.predict(polytransform.fit_transform(X))) #shifts the curve to the left without X coordinate
#plt.plt(X_poly, color ='red') #doesnt work
plt.title('Linear regression plot')
plt.xlabel('Employment level')
plt.ylabel('Salary')

#predict linear regression model output
lin_regr.predict(6.5) #$330378

#predict poly regression model output
poly_regr.predict(polytransform.fit_transform(6.5)) #$171303

#conclusion
#plolynomial regression fits the data better but only with one variable and higher degrees

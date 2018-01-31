# Multiple Linear Regression

# Importing the libraries
import numpy as np #to use mathematics in the code
import matplotlib.pyplot as plt #to do visualization
import pandas as pd #to import/export data set and manage it

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values 
y = dataset.iloc[:,4].values 

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Remove dummy variable  trap : remove one column of dummy variable to get rid of redundancy
X=X[:,1:]

# Splitting the dataset into the Training and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fitting a multiple regression model
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(X_train,y_train)

# Predicting output: profit
y_pred= regressor.predict(X_test)

# Visulaizing the output: Training: cant plot multiple regression plot in 2D

#Backward elimination
#Current model have y= b1x1+b2x2+b3x3... but no intercept or b0x0 where x0=1
#therefore, we add a column of ones to matrix X at the begining.
# Also it is required by 'statsmodels' library to find significance of each independent variables
#appending coefficients array to matrix
X=np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)

import statsmodels.formula.api as sm
X_opt = X[:,[0,1,2,3,4,5]] #so that it becomes convinient to remove insignificant variables
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() #No train or test, use entire data set
regressor_OLS.summary() #give the output to see which variable has highest pvalue>0.05
#x2 has highest pvalue so remove it, index =2
X_opt = X[:,[0,1,3,4,5]] 
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() 
regressor_OLS.summary()
#x1 has highest pvalue so remove it, index =1
X_opt = X[:,[0,3,4,5]] 
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() 
regressor_OLS.summary()
#x2 has highest pvalue so remove it, index  = 2
X_opt = X[:,[0,3,5]] 
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() 
regressor_OLS.summary()
#x2 remove, index 2
X_opt = X[:,[0,3]] 
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() 
regressor_OLS.summary()

#all variable pvalues <0.05
#X[:,3] is the R&D spend, therefore profit depends upon R&D spend
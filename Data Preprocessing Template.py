"""
@author: Vandan
"""

# Data Preprocessing Template
#press ctrl+i on editor/source to see the definitions

# Importing the libraries
import numpy as np #to use mathematics in the code
import matplotlib.pyplot as plt #to do visualization
import pandas as pd #to import/export data set and manage it

# Importing the dataset
# To import any dataset first we have to specify working directory where the 
# requried data set is saved. Select the data folder and save the script in that data folder to 
# make it the working directory.
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values #Make input variable matrix
y = dataset.iloc[:,3].values #make output variable matrix

# Handling missing values
from sklearn.preprocessing import Imputer #from 'sklearn' package import 'preprocessing' library and 'Imputer' class
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0) #creates an object of Imputer class ('NaN' not 'nan')
imputer = imputer.fit(X[:,1:3]) #selects the columns needed to be transformed, apply 'fit' method 
X[:,1:3] = imputer.transform(X[:,1:3]) #updates the columns with the transformed values
a=imputer.fit_transform(X[:,1:3]) #fit_transform gives the same results as of doing seperately 

# Encoding the categorical variables
# LabelEncoder :encodes the categorical values in to number lables but in the same column
# OneHotEncoder: to use categorical variables in the analysis we need to create dummy variables ie multiple colums 
#out of single categorical column 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features= [0]) #[0] is the location of the categorical column in the X matrix
X = onehotencoder.fit_transform(X).toarray() 
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting the dataset into the Training and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) 
#we dont use 'fit_transform' because we want to use same scaling as that of training set and we already fitted trainig set

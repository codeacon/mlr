# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as sm

#Recursive Function that takes a significance Level, X matrix and y values:
#  1. Creates a regressor based on given X and y values
#  2. Gets pValues from regressor
#  3. If the largest pValue is bigger than the SL, it removes that column and returns getModel(sigLevel, newX, y)
#  4. If the largest pValue is smaller than the SL, it returns the X values given.
def getModel(sigLev, X, y):
  regressorOLS = sm.OLS(endog = y, exog = X).fit()
  pValues = list(regressorOLS.pvalues)
  largestPVal = max(pValues)
  if (largestPVal > sigLev):
    columnNum = pValues.index(largestPVal)
    return getModel(sigLev, np.delete(X, columnNum, axis=1), y)
  else:
    return X

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

#Building the optimal model using Backward Elimination
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
#Step 1: Set significance level
SL = 0.05
#Step 2: Fit model with all possible 
#regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#Step 3:
#print(max(regressor_OLS.pvalues))
X = getModel(SL, X_opt, y)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling, unnecessary for this test
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#fitting multiple linear regression to the trainng set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

#Displaying the results with error for each result
count = 0
for i in y_test:
  print("The real value is: ", i)
  print("The predicted value is: ", y_pred[count])
  print("The difference is: ", i - y_pred[count])
  count = count + 1
  input()

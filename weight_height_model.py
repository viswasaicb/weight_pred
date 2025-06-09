# Importing the Libraries
from pandas import read_csv
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error
import pickle

# Importing weight_height dataset
data = read_csv("weight-height.csv")

# Independent Variable - Height
# Dependent Variable - Weight
height = data.iloc[:, 1:2].values
weight = data.iloc[:, 2:3].values

# Splitting the data into train and test set
h_train, h_test, w_train, w_test = train_test_split(height, weight, test_size = 0.2, random_state = 42)

# Creating regressor model
regressor = LinearRegression()

# Fitting training set to the data
regressor.fit(h_train, w_train)

data['y_pred']=regressor.predict(height)

#Checking Accuracy

RMSE = root_mean_squared_error(weight,data['y_pred'])
print('RMSE value is ',np.round(RMSE,2))

RSquare = r2_score(weight,data['y_pred'])
print('R square value is ',np.round(RSquare,2))



# Saving the model to the pickle file
pickle.dump(regressor, open('regressor_model.pkl', 'wb'))

# Loading model to predict the results
# model = pickle.load(open('regressor_model.pkl', 'rb'))

# print(model.predict([[80]]))
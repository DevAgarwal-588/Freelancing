#Question 1
import math
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import tensorflow
tensorflow.random.set_seed(1)
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor

from numpy.random import seed
seed(1234)

#Data Loading
data = pd.read_csv("Ex9data.csv")
y1 = data[[data.columns[3]]].to_numpy()
x1 = data[[data.columns[0], data.columns[1],data.columns[2]]].to_numpy()

X_train, X_val, y_train, y_val = train_test_split(x1, y1, train_size=0.8, random_state=10, shuffle= True)

#Min-Max Scaler
y_train=np.reshape(y_train, (-1,1))
y_val=np.reshape(y_val, (-1,1))

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

scaler_x.fit(X_train)
xtrain_scale=scaler_x.transform(X_train)

scaler_x.fit(X_val)
xval_scale=scaler_x.transform(X_val)

scaler_y.fit(y_train)
ytrain_scale=scaler_y.transform(y_train)
ytrain_scale = ytrain_scale.reshape((ytrain_scale.shape[0],))

scaler_y.fit(y_val)
yval_scale=scaler_y.transform(y_val)
yval_scale = yval_scale.reshape((yval_scale.shape[0],))

# trying multiple hidden layers
df = pd.DataFrame(columns = ['MAPE', 'hidden_1', 'hidden_2'])
for hidden_1 in range(1,11):
  for hidden_2 in range(1,11):
    regr = MLPRegressor(hidden_layer_sizes=(hidden_1,hidden_2), activation = 'tanh',solver = 'lbfgs', random_state=10, max_iter=100000)
    model = regr.fit(xtrain_scale, ytrain_scale)
    y_pred = model.predict(xval_scale)
    mape = MAPE(yval_scale, y_pred)
    df = df.append({'MAPE' : mape, 'hidden_1' : hidden_1, 'hidden_2' : hidden_2}, ignore_index = True)

df = df.sort_values('MAPE')
print("Best NN architecture: {}, {}, {}".format(df.iloc[0][1], df.iloc[0][2], df.iloc[0][0]))

###########################################################################################################
#Question 2:

from scipy.optimize import minimize
from scipy import optimize

regr = MLPRegressor(hidden_layer_sizes=(int(df.iloc[0][1]),int(df.iloc[0][2])), activation = 'tanh',solver = 'lbfgs', random_state=10, max_iter=100000)
model = regr.fit(xtrain_scale, ytrain_scale)

bounds = [(0, 1), (0, 1), (0,1)]

def eggholder(x):
  a = np.array([x]).reshape(1,3)
  return (-1* model.predict(a))

results = dict()
results['shgo'] = optimize.shgo(eggholder, bounds)

print("Optimal objective function value from NN: {}".format(results['shgo']['fun']))
print("Maximum regression label value from data: {}".format(model.predict(results['shgo']['x'].reshape((-1,3)))[0]))
print("Optimal solution from NN optimization: {}, {}, {}".format(results['shgo']['x'][0],results['shgo']['x'][1],results['shgo']['x'][2]))
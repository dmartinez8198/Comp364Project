from datetime import datetime, timedelta
import time
from collections import namedtuple
import pandas as pd
import requests
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pickle
import statsmodels.api as sm
from sklearn.metrics import explained_variance_score, \
     mean_absolute_error, \
     median_absolute_error
from sklearn.model_selection import train_test_split

API_KEY = '7deb208750e1cc9558969cad0f80951b'
url = "http://api.openweathermap.org/data/2.5/forecast?id=4887398&appid={}"

features = ["temp", "feels_like", "temp_min", "temp_max", "humidity"]

Main = namedtuple("list", features)


def extract_weather_data(url, API_KEY, days):
    records = []
    for _ in range(days):
        request = url.format(API_KEY)
        response = requests.get(request)
        if response.status_code == 200:
            data = response.json()['list']['main'][0]
            records.append(DailySummary(
                hour = hour_count,
                temp = data['temp'],
                feelslike = data['feels_like'],
                tempmin = data['temp_min'],
                tempmax = data['temp_max'],
                humidity = data['humidity']))
        time.sleep(6)
        hour_count += timedelta(hours = 3)
    return records

records = extract_weather_data(url, API_KEY, 30)

df = pd.DataFrame(records, columns = features).set_index('hour')

#previous data
N =1

feature = ['temp', 'feels_like', 'temp_min', 'temp_max', 'humidity']


def derive_nth_day_feature(df, feature, N):
    # total number of rows
    rows = df.shape[0]
    #list representing Nth prior measurements, None maintains consistent rows
    nth_prior_measurements = [None]*N + [df[feature][i-N] for i in range(N, rows)]
    # new column name of Feature_N and add to DataFrame
    col_name = "{}_{}".format(feature, N)
    df[col_name] = nth_prior_measurements

    #loops over features
    for feature in features:
        if feature != 'hour':
            for N in range(1, 4):
                derive_nth_day_feature(df, feature, N)

#Converts all feature columns to floats                
df =df.apply(pd.to_numeric, errors='coerce')
#call describe on df and transpose it due to large number of columns
spread = df.describe().T
#precalculate range for ease of use
IQR = spread['75%'] - spread['25%']

#outlier column
spread['outliers'] = (spread['min']<(spread['25%']-(3*IQR)))|(spread['max'] > (spread['75%']+3*IQR))

#extreme outliers
spread.ix[spread.outliers,]


predictors = ['temp_1', 'temp_2' , 'temp_3' , 'temp_min_1', 'temp_min_2', 'temp_min_3', 'temp_max_1', 'temp_max_2', 'temp_max_3',]

df2 = df[['meantemp'] +predictors].set_index('hours')

df2 = df2.drop(['temp_min', 'temp_max'], axis = 1)

x = df[[col for col in df3.columns if col != 'meantemp']]

y = df['meantemp']

x_train, x_tmp, y_train, y_tmp = train_test_split(x, y, test_size = 0.2, random_state=23)

x_test, x_val, y_test, y_val = train_test_split(x_tmp, y_tmp, test_size =0.5, random_state=23)


x_train.shape, x.test.shape, x_val.shape

print("Training instances {}, Training features {}".format(x_train.shape[0], x_train.shape[1]))
print("Validation instances {}, Validation features {}".format(x_val.shape[0], x_val.shape[1]))
print("Testing instances {}, Testing features {}".format(x_test.shape[0], x_test.shape[1]))

feature_cols= [tf.feature_column.numeric_column(col) for col in x.columns]

regressor = tf.estimator.DNNRegressor(feature_columns = feature_cols, hidden_units[50, 50], model_dir = 'weather_model')

def input_function(x, y = None, num_epochs = None, shuffle = True, batch_size = 100):
    return tf.estimator.inputs.pandas_input_fn(x = x, y = y, num_epochs = num_epochs, shuffle = shuffle, batch_size = batch_size)


evaluations = []
steps = 200
for i in range(20):
    regressor.train(input_fn = input_function(x_train, y = y_train), steps = steps)
    evaluations.append(regressor.evaluate(input_fn = input_function(x_val, y_val, num_epochs = 1, shuffle, False)))

%matplotlib inline
plt.rcParams['figure.figsize'] = [14, 10]

loss_values = [ev['loss'] for ev in evaluations]
training_steps = [ev['step'] for ev in evaluations]

plt.scatter(x=training_steps, y = loss_values)
plt.xlabel('Training steps (Epochs = steps/2)')
plt.ylabel('Loss(SSE)')
plt.show()

pred = regressor.predict(input_fn = input_function(x_test, num_epochs = 1, shuffle = False))
predictions = np.array([p['predictions'][0] for p in pred])
print("Explained variance: %.2f" % explained_variance_score(y_test, predictions))
print("Mean absolute error: %.2f degrees Celcius" % mean_absolute error(y_test, predictions))
print("Median absolute error: %.2f degrees Celcius" % median_absolute_error(y_test, predictions))


    






    

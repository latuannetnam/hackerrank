import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
import sys


def score(y_true, y_pred):
    result = abs(y_true - y_pred) / y_true
    return result.mean()


def train_predict(train_data, test_data):
    n_features = test_data.shape[1]
    input_data = train_data[:, :n_features]
    # input_data = input_data.reshape(-1, 1)
    target = train_data[:, -1]
    target = target.ravel()
    # print("input:", input_data.shape, " target:", target.shape)
    # print(target[:5])
    # model = LinearRegression()
    # ElasticNet
    model = ExtraTreesRegressor(n_estimators=500, max_depth=10, n_jobs=-1, random_state=456)
    x_train, x_test, y_train, y_test = train_test_split(
        input_data, target, test_size=0.33, random_state=42)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print("Score:", score(y_test, y_pred))
    y_prediction = model.predict(test_data)
    y_prediction = y_prediction.ravel()
    for pred in y_prediction:
        print(str(pred))


state = 0
for line in sys.stdin:
    # print("state:", state, " line:", line)
    if state == 0:
        n_features, n_trains = line.split(' ')
        n_features = int(n_features)
        n_trains = int(n_trains)
        # print("n_features:", n_features, "n_train:", n_trains)
        state = 1
        row_count = 0
        train_data = np.zeros((n_trains, n_features + 1))
    elif state == 1:
        train_data[row_count] = line.split(' ')
        row_count = row_count + 1
        if (row_count == n_trains):
            # print(train_data[:5])
            state = 2
    elif state == 2:
        n_tests = int(line)
        test_data = np.zeros((n_tests, n_features))
        row_count = 0
        state = 3
    elif state == 3:
        test_data[row_count] = line.split(' ')
        row_count = row_count + 1
        if (row_count == n_tests):
            # print(test_data[:5])
            break

train_predict(train_data, test_data)

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.grid_search import GridSearchCV
import sys
import logging

POLY_DEGREE = 1
LOG_LEVEL = logging.DEBUG
# LOG_LEVEL = logging.INFO
# create logger
logger = logging.getLogger('hackerrank')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(LOG_LEVEL)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def search_best_params(model, param_grid, X, Y):
    model_name = model.__class__.__name__
    print('Searching best param for model:', model_name)
    grid_search = GridSearchCV(model, param_grid, n_jobs=1, cv=5)
    grid_search.fit(X, Y)
    print(grid_search.best_params_)
    quit()


def score(y_true, y_pred):
    result = abs(y_true - y_pred) / y_true
    return result.mean()


def train_predict(train_data, test_data):
    n_features = test_data.shape[1]
    input_data = train_data[:, :n_features]
    n_trains = input_data.shape[0]
    n_tests = test_data.shape[0]
    target = train_data[:, -1]
    target = target.ravel()
    # make polynominal features
    poly = PolynomialFeatures(POLY_DEGREE, True)
    combine_data = np.concatenate((input_data, test_data), axis=0)
    combine_data_tf = poly.fit_transform(combine_data)
    # print("Old features:", combine_data.shape, " New poly features:", combine_data_tf.shape)
    logger.debug("Old features:" + str(combine_data.shape) +
                 ", New poly features:" + str(combine_data_tf.shape))
    input_data = combine_data_tf[:n_trains]
    test_data = combine_data_tf[n_trains:]
    # print("input:", input_data.shape, " test:", test_data.shape)
    logger.debug("input:" + str(input_data.shape) +
                 ", test:" + str(test_data.shape))
    model = LinearRegression()
    # ElasticNet
    # model = ExtraTreesRegressor(
    #     n_estimators=500, max_depth=3, n_jobs=-1, random_state=456)
    param_grid = {"n_estimators": [50, 100, 200, 500],
                  "max_depth": [1, 3, 5, 10],
                  },
    # search_best_params(model, param_grid, input_data, target)
    x_train, x_test, y_train, y_test = train_test_split(
        input_data, target, test_size=0.33, random_state=42)

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    logger.debug("Score:" + str(score(y_test, y_pred)))
    y_prediction = model.predict(test_data)
    y_prediction = y_prediction.ravel()
    for pred in y_prediction:
        print(str(pred))


#  ----------------- Main --------------------
state = 0
logger.debug("House price prediction")
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

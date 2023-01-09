import pickle
import time
from pathlib import Path

import numpy as np

from learner.data_preprocessing import *
# import mean_squared_error from sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

import pandas as pd


# import leave one out cross validation
# Entry function for the missing values test.
def missing_values_main(X_train, y_train, X_test, y_test):
    # Merge X_train and y_train dataframes
    train = pd.concat([X_train, y_train], axis=1)

    remaining_X_test, remaining_y_test, removed_X_test, removed_Y_test \
        = missing_vt_combinations_test(train, 2)

    # Train model
    print('Training model with missing values')
    model_with_missing_values = train_tuned_random_forest(remaining_X_test, remaining_y_test)
    test_tuned_random_forst(model_with_missing_values , X_test, y_test)

    print('Training model without missing values')
    model_without_missing_values = train_tuned_random_forest(X_train, y_train)
    test_tuned_random_forst(model_without_missing_values, X_test, y_test)


def train_tuned_random_forest(X_train, y_train):
    print('------- Train tuned random forest --------')
    print(f'Number of training samples: {len(X_train)}')
    print(f'Number of testing samples: {len(X_test)}')
    print('-----')
    pipe = Pipeline(
        [("scaler", MinMaxScaler()), ("rfr", RandomForestRegressor(bootstrap=True, criterion='absolute_error',
                                                                   max_depth=30, min_samples_leaf=2,
                                                                   min_samples_split=4, n_estimators=10))])
    sample = X_test.iloc[0]
    start = time.time()
    pipe.fit(X_train, y_train)
    stop = time.time()
    s = stop - start
    print(f"Training time: {s} s")

    return pipe


def test_tuned_random_forst(model, X_test, y_test):
    print('------- Testing tuned random forest --------')
    sample = X_test.iloc[0]
    start = time.time()
    model.predict([sample])
    stop = time.time()
    ms = (stop - start) * 1000

    print(f"Prediction time (runtime): {ms} ms")

    save_trained_model(model, 'tuned_random_forest')

    y_pred = model.predict(X_test)
    print("MAE: {:.3f}".format(mean_absolute_error(y_test, y_pred)))
    print("MSE: {:.3f}".format(mean_squared_error(y_test, y_pred)))
    print("RMSE: {:.3f}".format(np.sqrt(mean_squared_error(y_test, y_pred))))
    print("R2 score: {:.3f}".format(r2_score(y_test, y_pred)))
    print('-----------------')


# Save trained model to disk
def save_trained_model(model, name):
    project_root = get_project_root()
    output_directory = project_root / 'learner' / 'rfr' / 'saved_models'

    with open(f'{output_directory}/{name}.model', 'wb') as f:
        pickle.dump(model, f)


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent


if __name__ == '__main__':
    # Constants
    train_split = 30
    project_root = get_project_root()
    input_directory = project_root / 'data' / 'dataset'

    # Load data
    df = pd.read_csv(input_directory / 'consolidated.csv', delimiter=',')

    X, y, X_train, y_train, X_test, y_test = non_random_split(df, train_split)
    missing_values_main(X_train, y_train, X_test, y_test)

    print('Done!')

import numpy as np
import pandas as pd
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from learner.data_preprocessing import *
from sklearn.model_selection import GridSearchCV
from reports.reports_main import create_reports
import learner.data_preprocessing as dp
from learner.visualizing_results.global_model_agnostic_methods import *


def svr(df, md):
    # Best parameters: {'svr__C': 4000, 'svr__epsilon': 0.001, 'svr__gamma': 0.1,
    # 'svr__kernel': 'rbf'}
    svr_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svr',
         SVR(kernel='rbf',
             C=5000,
             degree=1,
             epsilon=0.01,
             gamma=0.01,
             ))
    ])

    # Fit the models
    svr_pipe.fit(md.X_train, md.y_train)

    # Predict the test set
    y_pred = svr_pipe.predict(md.X_test)

    return svr_pipe, y_pred


def untrained_svr():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('svr',
         SVR(kernel='rbf',
             C=5000,
             degree=1,
             epsilon=0.01,
             gamma=0.01,
             ))
    ])


def print_results(model, y_test, y_pred):
    # Print the results
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # print(f'MAE: {mae}')
    # print(f'MSE: {mse}')
    # print(f'RMSE: {rmse}')
    # print(f'R2: {r2}')
    # print("Test score: {:.2f}".format(model.score(X_test, y_test)))
    #
    # print("Accuracy on training set: {:.2f}".format(model.score(X_train, y_train)))
    # print("Accuracy on test set: {:.2f}".format(model.score(X_test, y_test)))


# Perform grid search to find best hyper-parameters for SVR
def grid_search(pipe, X_train, y_train, X_test, y_test):
    print('------- Grid search --------')

    #  Best parameters:
    #  {'svr__C': 4000, 'svr__degree': 1, 'svr__epsilon': 0.001, 'svr__gamma': 0.1,
    #  'svr__kernel': 'rbf'}

    # Create range of candidate values
    param_grid = {'svr__C': [0.1, 1, 10, 100, 1000, 2000, 3000, 4000, 5000],
                  'svr__gamma': [0.1, 0.01, 0.001, 0.0001, 0.00001],
                  'svr__epsilon': [0.001, 0.01, 0.1, 1, 10, 100],
                  'svr__kernel': ['rbf', 'linear', 'poly'],
                  'svr__degree': [1, 2, 3, 4, 5],
                  }

    # Create grid search
    grid = GridSearchCV(pipe, param_grid, refit=True, verbose=3)
    grid.fit(X_train, y_train)

    # Print best hyper-parameters
    print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
    print("Test set score: {:.2f}".format(grid.score(X_test, y_test)))
    print("R2 score: {:.2f}".format(r2_score(y_test, grid.predict(X_test))))
    print("Best parameters: {}".format(grid.best_params_))


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent


if __name__ == '__main__':
    project_root = get_project_root()
    input_directory = project_root / 'data' / 'dataset'

    # Load data
    df = dp.get_data()

    # Split data
    md = non_random_split(df, 30)

    # Create model
    model, y_pred = svr(df, md)

    # Create reports
    reports = ['resource']

    name = "SVM"

    create_reports(name, reports, md, model, y_pred)
    # grid_search(model, md.X_train, md.y_train, md.X_test, md.y_test)

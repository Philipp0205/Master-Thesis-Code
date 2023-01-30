from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from learner.reports.reports_main import create_reports
from learner.rfr.random_forest import get_project_root

import misc.folders as folders
import learner.data_preprocessing as preprocessing


def create_linear_regression_model(model_data):
    lr_pipe = Pipeline([
        ('scaler', MinMaxScaler(feature_range=(0, 1))),
        ('svr', LinearRegression())
    ])

    lr_pipe.fit(model_data.X_train, model_data.y_train)

    y_pred = lr_pipe.predict(model_data.X_test)

    return lr_pipe, y_pred


def linear_regression_grid_search(pipe, X_train, y_train, X_test, y_test):
    print('------- Grid search --------')


if __name__ == '__main__':
    df = preprocessing.get_data()
    model_data = preprocessing.non_random_split(df, 30)
    model_data2 = preprocessing.random_split(df)

    # Create Linear Regression model
    model, y_pred = create_linear_regression_model(model_data)
    model2, y_pred2 = create_linear_regression_model(model_data2)

    reports = ['robustness']
    create_reports(reports, model_data, model, y_pred)
    # create_reports(reports, model_data2, model2, y_pred2)

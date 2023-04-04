from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import learner.data_preprocessing as preprocessing
from reports.reports_main import create_reports


def ada_boost(model_data):
    pipe = Pipeline(
        [("scaler", StandardScaler()),
         ("ada_boost", AdaBoostRegressor(learning_rate=0.2, loss='exponential', n_estimators=10))])

    pipe.fit(model_data.X_train, model_data.y_train)
    y_pred = pipe.predict(model_data.X_test)

    return pipe, y_pred


def grid_search(model_data, pipe):
    param_grid = {
        'ada_boost__n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'ada_boost__learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
        'ada_boost__loss': ['linear', 'square', 'exponential'],
    }

    gs = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1, verbose=1)
    gs.fit(model_data.X_train, model_data.y_train)

    print("Best parameters: {}".format(gs.best_params_))


if __name__ == '__main__':
    df = preprocessing.get_data()
    model_data = preprocessing.non_random_split(df, 30)
    model_data2 = preprocessing.random_split(df, 0.3)
    # model_data2 = preprocessing.non_random_split_with_validation(df, 30)

    model, y_pred = ada_boost(model_data)

    # grid_search(model_data, model)
    name = 'AB'
    reports = ['robustness']
    create_reports(name, reports, model_data, model, y_pred)

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import learner.data_preprocessing as preprocessing
from reports.reports_main import create_reports


def gradient_boosting(model_data):
    pipe = Pipeline(
        [("scaler", StandardScaler()),
         ("gradient_boosting",
          GradientBoostingRegressor(
             loss='huber',
             learning_rate=0.1,
             max_depth=4,
             min_samples_leaf=3,
             min_samples_split=3,
             n_estimators=200,
         ))])

    pipe2 = GradientBoostingRegressor(
        max_depth=5,
    )

    pipe2.fit(model_data.X_train, model_data.y_train)
    y_pred = pipe2.predict(model_data.X_test)

    return pipe2, y_pred


def grid_search(model, model_data):
    pipe = Pipeline(
        [("scaler", StandardScaler()),
         ("gradient_boosting", GradientBoostingRegressor(
             loss='huber',
             learning_rate=0.1,
             max_depth=4,
             min_samples_leaf=3,
             min_samples_split=3,
             n_estimators=200,
         ))])

    param_grid = {
        'gradient_boosting__loss': ['squared_error', 'absolute_error', 'huber',
                                    'quantile'],
        # 'gradient_boosting__learning_rate': [0.1, 0.05, 0.02, 0.01],
        # 'gradient_boosting__n_estimators': [100, 200, 300, ],
        # 'gradient_boosting__max_depth': [1, 2, 3, 4, 5, 6],
        # 'gradient_boosting__min_samples_split': [2, 3, 4, 9, 10],
        'gradient_boosting__min_samples_leaf': [1, 2, 3, 4, 5, 6, 9, 10],
        # 'gradient_boosting__min_weight_fraction_leaf': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        # 'gradient_boosting__subsample': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
        #                                  1.0],
        # 'gradient_boosting__max_features': ['auto', 'sqrt', 'log2', None],
        # 'gradient_boosting__max_leaf_nodes': [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        # 'gradient_boosting__min_impurity_decrease': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        # 'gradient_boosting__min_impurity_split': [None, 0.1, 0.2, 0.3, 0.4, 0.5],
        # 'gradient_boosting__alpha': [0.9, 0.75, 0.5, 0.25, 0.1],
    }

    gs = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1, verbose=1)
    gs.fit(model_data.X_train, model_data.y_train)

    print("Best parameters: {}".format(gs.best_params_))


if __name__ == '__main__':
    df = preprocessing.get_data()
    model_data = preprocessing.non_random_split(df, 30)
    model_data2 = preprocessing.random_split(df, 0.3)

    model, y_pred = gradient_boosting(model_data)
    # model2, y_pred2 = gradient_boosting(model_data2)

    # grid_search(model, model_data)

    name = 'GBT'
    reports = ['robustness']
    create_reports(name, reports, model_data, model, y_pred)
    # create_reports(name, reports, model_data2, model2, y_pred2)

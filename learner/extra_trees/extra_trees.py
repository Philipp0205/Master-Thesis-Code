from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import learner.data_preprocessing as preprocessing
from reports.reports_main import create_reports


def extra_trees(model_data):
    pipe = ExtraTreesRegressor(
                            bootstrap=True,
                               criterion='absolute_error',
                               max_depth=6,
                               min_samples_split=4,
                               # n_estimators=60,
                               n_estimators=10,
                               # min_impurity_decrease=0.0,
                               # min_weight_fraction_leaf=0.0,
                               min_samples_leaf=1,
                               )

    # pipe = Pipeline(
    #     [("scaler", MinMaxScaler()),
    #      ("et", ExtraTreesRegressor(bootstrap=False,
    #                                 criterion='absolute_error',
    #                                 max_depth=6,
    #                                 min_samples_split=4,
    #                                 n_estimators=60,
    #                                 min_impurity_decrease=0.0,
    #                                 min_weight_fraction_leaf=0.0,
    #                                 min_samples_leaf=1,
    #                                 ))])

    pipe.fit(model_data.X_train, model_data.y_train)

    y_pred = pipe.predict(model_data.X_test)

    return pipe, y_pred


def random_forest(model_data):
    # Best parameters: {'random_forest__max_depth': 7,
    # 'random_forest__min_samples_leaf': 2, 'random_forest__min_samples_split': 5,
    # 'random_forest__n_estimators': 10}

    pipe = Pipeline(
        [("scaler", StandardScaler()),
         ("random_forest", RandomForestRegressor(bootstrap=True,
                                                 criterion='absolute_error',
                                                 min_samples_split=5,
                                                 min_samples_leaf=2,
                                                 n_estimators=10,
                                                 max_depth=7,
                                                 ))])

    pipe.fit(model_data.X_train, model_data.y_train)

    y_pred = pipe.predict(model_data.X_test)

    return pipe, y_pred


def grid_search(pipe):
    param_grid = {
        # 'et__criterion': ['squared_error', 'absolute_error', 'friedman_mse'],
        'et__criterion': ['squared_error', 'absolute_error'],
        # 'et__n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'et__n_estimators': [60],
        # 'et__max_depth': [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'et__max_depth': [6],
        # 'et__min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10],
        'et__min_samples_split': [4],
        # 'et__bootstrap': [True, False],
        'et__bootstrap': [False],
        'et__min_samples_leaf': [1, 2, 4],
        'et__min_weight_fraction_leaf': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        'et__max_leaf_nodes': [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'et__min_impurity_decrease': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    }

    grid = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1, verbose=1)
    grid.fit(model_data.X_train, model_data.y_train)

    # Print best hyper-parameters
    print('Best parameters: {}'.format(grid.best_params_))


if __name__ == '__main__':
    print('------- Extra Trees --------')
    df = preprocessing.get_data()
    model_data = preprocessing.non_random_split(df, 30)
    model_data2 = preprocessing.random_split(df, 0.3)
    # model_data3 = preprocessing.non_random_split_with_validation(df, 30)

    # Create Linear Regression model
    model, y_pred = extra_trees(model_data)
    model2, y_pred2 = extra_trees(model_data2)
    # model3, y_pred3 = extra_trees(model_data2)

    # grid_search(model)

    name = 'ET'
    reports = ['robustness']
    create_reports(name, reports, model_data, model, y_pred)
    # create_reports(name, reports, model_data2, model2, y_pred2)
    # create_reports(name, reports, model_data3, model3, y_pred3)

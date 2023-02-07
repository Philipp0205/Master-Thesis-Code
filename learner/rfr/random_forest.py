import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV

import scienceplots

from learner.data_preprocessing import *
# import mean_squared_error from sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor

from learner.reports.reports_main import create_reports

import learner.data_preprocessing as preprocessing


def random_forest(df, X_train, y_train, X_test, y_test):
    pipe = Pipeline(
        [("scaler", StandardScaler()),
         ("rfr", RandomForestRegressor(bootstrap=True,
                                       criterion='absolute_error',
                                       min_samples_split=2,
                                       n_estimators=40,
                                       ))])

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    return pipe, y_pred


def grid_search(pipe):
    param_grid = {
        'rfr__n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'rfr__criterion': ['squared_error', 'absolute_error'],
        # 'rfr__max_depth': [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        # 'rfr__min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10],
        'rfr__min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        # 'rfr__min_weight_fraction_leaf': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    }

    grid = GridSearchCV(pipe, param_grid, cv=5, verbose=1)
    grid.fit(model_data.X_train, model_data.y_train)

    print("Best parameters: {}".format(grid.best_params_))


def feature_importances(rf_model, df):
    feature_names = df.columns
    importances = rf_model.feature_importances_
    # Remove 'springback"
    feature_names = feature_names.drop('springback')

    forest_importances = pd.Series(importances, index=feature_names)

    indices = np.argsort(importances)

    plt.style.use(['science', 'ieee'])

    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.savefig('rf_feature_importances.png', dpi=600)


def calculate_feature_importances(regressor, X_train):
    # Importances
    importances = regressor.feature_importances_

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_train.columns[i] for i in indices]

    # Use scientific style
    plt.style.use(['science', 'bright'])

    # Create plot
    plt.figure(dpi=600)
    plt.title("Feature Importance")

    # Add bars
    plt.bar(range(X_train.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_train.shape[1]), names, rotation=90)

    # Save figure
    plt.savefig(f'results/rfr_default/Random_Forest_Regression_importances.png')
    plt.clf()


if __name__ == '__main__':
    df = preprocessing.get_data()
    model_data = preprocessing.non_random_split(df, 30)
    model_data2 = preprocessing.random_split(df)

    # Create Linear Regression model
    model, y_pred = random_forest(model_data, model_data.X_train, model_data.y_train,
                                  model_data.X_test, model_data.y_test)
    model2, y_pred2 = random_forest(model_data2, model_data2.X_train, model_data2.y_train,
                                    model_data2.X_test, model_data2.y_test)

    # grid_search(model)
    # calculate_feature_importances()feature_importances(model.steps[1][1], df)
    calculate_feature_importances(model.steps[1][1], model_data.X_train)

    name = 'RF'
    reports = ['correctness']
    create_reports(name, reports, model_data, model, y_pred)
    create_reports(name, reports, model_data2, model2, y_pred2)

import numpy as np
from sklearn.model_selection import GridSearchCV

from learner.data_preprocessing import *
# import mean_squared_error from sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from reports.reports_main import create_reports
import learner.data_preprocessing as preprocessing
import learner.visualizing_results.lime as lime


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


def feature_importances1(rf_model, df):
    plt.style.use(['science'])
    importances = rf_model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf_model.estimators_], axis=0)

    feature_names = df.columns
    # Remove 'springback"
    feature_names = feature_names.drop('springback')

    forest_importances = pd.Series(importances, index=feature_names)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    # fig.tight_layout()
    plt.savefig('rf_feature_importances.png', transparent=True, dpi=600)




def grid_search(model, model_data):
    pipe = Pipeline(
        [("scaler", StandardScaler()),
         ("random_forest", RandomForestRegressor(bootstrap=True,
                                                 criterion='absolute_error',
                                                 min_samples_split=5,
                                                 min_samples_leaf=2,
                                                 n_estimators=10,
                                                 max_depth=7,
                                                 ))])

    param_grid = {
        'random_forest__n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'random_forest__criterion': ['squared_error', 'absolute_error'],
        'random_forest__max_depth': [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'random_forest__min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10],
        'random_forest__min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'random_forest__min_weight_fraction_leaf': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    }

    gs = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1, verbose=1)
    gs.fit(model_data.X_train, model_data.y_train)

    print("Best parameters: {}".format(gs.best_params_))


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
    plt.title("Feature Importance1")

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
    model_data2 = preprocessing.random_split(df, 0.3)
    # model_data2 = preprocessing.non_random_split_with_validation(df, 30)

    model, y_pred = random_forest(model_data)
    model2, y_pred2 = random_forest(model_data2)

    # grid_search(model, model_data)

    # grid_search(model)
    # calculate_feature_importances()feature_importances(model.steps[1][1], df)
    # calculate_feature_importances(model.steps[1][1], model_data.X_train)

    # feature_importances1(model.steps[1][1], df)

    name = 'RF'
    # lime.create_lime_explanation(name, model, df, model_data)
    lime.create_lime_subplot(model_data)
    # reports = ['stability']
    # create_reports(name, reports, model_data, model, y_pred)
    # create_reports(name, reports, model_data2, model2, y_pred2)

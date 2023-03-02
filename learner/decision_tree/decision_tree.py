import dtreeviz
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeRegressor

import matplotlib.pyplot as plt

import learner.data_preprocessing as preprocessing
from reports.reports_main import create_reports


def decision_tree(model_data):
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        (
            'decision tree',
            DecisionTreeRegressor(max_depth=7,
                                  min_samples_leaf=5,
                                  min_samples_split=6,
                                  min_weight_fraction_leaf=0.0,
                                  ccp_alpha=0.0,
                                  splitter='random'))
    ])

    pipe.fit(model_data.X_train, model_data.y_train)

    y_pred = pipe.predict(model_data.X_test)

    return pipe, y_pred


def grid_search(pipe):
    param_grid = {
        'decision tree__criterion': ['squared_error'],
        'decision tree__splitter': ['random'],
        # 'decision tree__max_depth': [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'decision tree__max_depth': [7],
        'decision tree__min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'decision tree__min_weight_fraction_leaf': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        # 'decision tree__min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10],
        'decision tree__min_samples_split': [6],
        # 'decision tree__min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'decision tree__min_weight_fraction_leaf': [0.0, 0.1, 0.2, 0.3, 0.4,
                                                    0.5],
        'decision tree__min_samples_leaf': [1, 2, 3],
        # 'decision tree__max_leaf_nodes': [None, 1, 2, 3, 4, 5, 6,G 7, 8, 9, 10],
        # 'decision tree__min_impurity_decrease': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        'decision tree__ccp_alpha': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

        # 'decision tree__max_features': [None, 'auto', 'sqrt', 'log2'],
        # 'decision tree__min_impurity_split': [None, 0.1, 0.2, 0.3, 0.4, 0.5],
    }

    grid = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1, verbose=1)
    grid.fit(model_data.X_train, model_data.y_train)

    # Print best hyper-parameters
    print("Best parameters: {}".format(grid.best_params_))


def get_feature_names(df):
    # Get feature names as array
    df = df.drop('springback', axis=1)
    feature_names = list(df.columns)

    return feature_names


def visualize_tree(model, X, y, feature_names):
    dtr = DecisionTreeRegressor(max_depth=7,
                                min_samples_leaf=5,
                                min_samples_split=6,
                                min_weight_fraction_leaf=0.0,
                                ccp_alpha=0.0,
                                splitter='random')

    dtr = dtr.fit(X, y)

    viz_rmodel = dtreeviz.model(dtr, X, y,
                                feature_names=feature_names,
                                target_name='springback')

    viz_rmodel.rtree_feature_space(features=['thickness', 'die_opening'])

    viz_rmodel.rtree_feature_space3D(features=['die_opening', 'thickness'],
                                     fontsize=10,
                                     elev=30, azim=20,
                                     show={'splits', 'title'},
                                     colors={'tessellation_alpha': .5})

    viz3 = dtreeviz.model(model=dtr,
                          X_train=X,
                          y_train=y,
                          feature_names=feature_names,

                          target_name="springback")

    viz_rmodel = dtreeviz.model(model=dtr,
                                X_train=X,
                                y_train=y,
                                feature_names=feature_names,
                                target_name="springback")


def visualization_temp(df):
    x = df['distance']
    y = df['springback']

    plt.scatter(x, y, color='red')
    plt.title('Distance vs Springback')
    plt.xlabel('Distance')
    plt.ylabel('Springback')
    plt.savefig('distance_vs_springback.png')


if __name__ == '__main__':
    df = preprocessing.get_data()
    model_data = preprocessing.non_random_split(df, 30)
    model_data2 = preprocessing.random_split(df, 0.3)

    # Create Linear Regression model
    model, y_pred = decision_tree(model_data)
    model2, y_pred2 = decision_tree(model_data2)

    # grid_search(model)
    feature_names = get_feature_names(df)

    name = 'DT'
    reports = ['resource']
    create_reports(name, reports, model_data, model, y_pred)
    create_reports(name, reports, model_data2, model2, y_pred2)

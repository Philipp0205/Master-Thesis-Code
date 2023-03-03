from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.inspection import PartialDependenceDisplay

import learner.data_preprocessing as preprocessing
from reports.reports_main import create_reports


def mlp(X_train, y_train, X_test, y_test):
    # Best parameters: {'mlp__activation': 'tanh', 'mlp__alpha': 0.05,
    # 'mlp__hidden_layer_sizes':
    # (100, 100), 'mlp__learning_rate': 'constant', 'mlp__max_iter': 500,
    # 'mlp__solver': 'lbfgs'}
    pipe = Pipeline([
        ('scaler', MinMaxScaler(feature_range=(0, 1))),
        ('mlp', MLPRegressor(random_state=1,
                             solver='lbfgs',
                             # solver='sgd',
                             max_iter=5000,
                             # activation='tanh',
                             alpha=0.05,
                             hidden_layer_sizes=(100, 100),
                             learning_rate='constant',
                             ))])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    return pipe, y_pred


def grid_search(pipe):
    param_grid = {
        'mlp__hidden_layer_sizes': [(100, 100, 100), (100, 100), (100,)],
        'mlp__activation': ['relu', 'tanh', 'logistic'],
        'mlp__solver': ['lbfgs', 'sgd', 'adam'],
        'mlp__alpha': [0.0001, 0.05],
        'mlp__learning_rate': ['constant', 'invscaling', 'adaptive'],
        'mlp__max_iter': [500, 1000, 2000, 5000],
    }

    grid = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1, verbose=1)
    grid.fit(model_data.X_train, model_data.y_train)

    # Print best hyper-parameters
    print('Best parameters: {}'.format(grid.best_params_))

def partial_dependence(model, md):
    features = ['distance', 'thickness', 'die_opening']
    display = PartialDependenceDisplay.from_estimator(model, md.X, features)

    # Save partial dependence plot
    display.plot()
    display.figure_.savefig('partial_dependence_mlp.png')


if __name__ == '__main__':
    df = preprocessing.get_data()

    model_data = preprocessing.non_random_split(df, 30)
    model_data2 = preprocessing.random_split(df, 0.3)

    model, y_pred = mlp(model_data.X_train,
                        model_data.y_train,
                        model_data.X_test,
                        model_data.y_test)

    model2, y_pred2 = mlp(model_data2.X_train,
                          model_data2.y_train,
                          model_data2.X_test,
                          model_data2.y_test)

    # grid_search(model)

    partial_dependence(model, model_data)

    # name = "MLP"
    # reports = ['stability']
    # create_reports(name, reports, model_data, model, y_pred)

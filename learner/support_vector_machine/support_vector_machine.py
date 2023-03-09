from sklearn.compose import ColumnTransformer
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from learner.data_preprocessing import *
# import GridSearchCV from sklearn
from sklearn.model_selection import GridSearchCV
from reports.reports_main import create_reports
import learner.data_preprocessing as dp
from learner.visualizing_results.global_model_agnostic_methods import *


def svr(df, X_train, y_train, X_test, y_test):
    # pipe = Pipeline([
    #     ('scaler', StandardScaler()),
    #     ('support_vector_machine',
    #      SVR(kernel='rbf', C=4000, gamma=0.1, epsilon=0.001, degree=1))
    # ])
    #
    pipe = Pipeline([
        ('scaler', MinMaxScaler(feature_range=(0, 1))),
        ('mlp', SVR(kernel='rbf', C=4000, gamma=0.1, epsilon=0.001, degree=1
                    ))])

    feature_names = X_train.columns

    model = SVR(kernel='rbf', C=4000, gamma=0.1, epsilon=0.001, degree=1)
    pipe.fit(X_train, y_train)

    model.fit(X_train, y_train)
    # Predict the test set
    y_pred = pipe.predict(X_test.values)
    # y_pred = model.predict(X_test)

    return pipe, y_pred


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
                  'svr__gamma': [0.1],
                  'svr__epsilon': [0.001],
                  'svr__kernel': ['rbf']}

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
    md2 = random_split(df, 0.3)
    md3 = non_random_split_with_validation(df, 30)

    # Create model
    model, y_pred = svr(df, md.X_train, md.y_train, md.X_test, md.y_test)
    # model2, y_pred2 = svr(df, md2.X_train, md2.y_train, md2.X_test, md2.y_test)
    # model3, y_pred3 = svr(df, md3.X_train, md3.y_train, md3.X_test, md3.y_test)

    # Create reports
    reports = ['stability']

    name = "SVM"
    feature_names = df.columns
    # remove target variable
    feature_names = feature_names.drop('springback').to_numpy()

    # partial_dependence_plot(model, md, feature_names, name)
    # model = model.named_steps['support_vector_machine']
    # feature_importance_plot(model, df, name)
    permutation_feature_importance(model, md, df)

    # create_reports(name, reports, md, model, y_pred)

from sklearn.neural_network import MLPRegressor
import learner.data_preprocessing as preprocessing
from learner.reports.reports_main import create_reports


def mlp(X_train, y_train, X_test, y_test):
    model = MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return model, y_pred


if __name__ == '__main__':
    df = preprocessing.get_data()

    model_data = preprocessing.non_random_split(df, 30)
    model_data2 = preprocessing.random_split(df)

    model, y_pred = mlp(model_data.X_train,
                        model_data.y_train,
                        model_data.X_test,
                        model_data.y_test)

    model2, y_pred2 = mlp(model_data2.X_train,
                          model_data2.y_train,
                          model_data2.X_test,
                          model_data2.y_test)

    name = "MLP"

    reports = ['robustness']
    create_reports(name, reports, model_data, model, y_pred)
    # create_reports(name, reports, model_data2, model2, y_pred2)

    print('Hello World!')

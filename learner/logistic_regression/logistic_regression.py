from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


import learner.data_preprocessing as preprocessing

from reports.reports_main import create_reports


def logistic_regression(df, X_train, y_train, X_test, y_test):
    pipe = Pipeline(
        [("scaler", MinMaxScaler()),
         ("lr", LogisticRegressionCV(cv=5, random_state=0))])

    # convert y values to categorical values
    lab = LabelEncoder()
    X_train_transformed = lab.fit_transform(X_train)
    y_train_transformed = lab.fit_transform(y_train)
    X_test_transformed = lab.fit_transform(X_test)

    pipe.fit(X_train_transformed, y_train_transformed)
    y_pred = pipe.predict(X_test_transformed)

    return pipe, y_pred


if __name__ == '__main__':
    df = preprocessing.get_data()

    model_data = preprocessing.non_random_split(df, 30)
    model_data2 = preprocessing.random_split(df, 0.2)

    model, y_pred = logistic_regression(df, model_data.X_train,
                                        model_data.y_train,
                                        model_data.X_test, model_data.y_test)

    model2, y_pred2 = logistic_regression(df, model_data2.X_train,
                                          model_data2.y_train,
                                          model_data2.X_test,
                                          model_data2.y_test)

    name = "LogR"

    reports = ['correctness']
    create_reports(name, reports, model_data, model, y_pred)
    create_reports(reports, model_data2, model2, y_pred2)

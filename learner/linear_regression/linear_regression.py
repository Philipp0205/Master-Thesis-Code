import pandas as pd
from matplotlib import pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler
import statsmodels.api as sm
import scienceplots

from learner.reports.reports_main import create_reports

import misc.folders as folders
import learner.data_preprocessing as preprocessing


def linear_regression(model_data):
    lr_pipe = Pipeline([
        ('scaler', MinMaxScaler(feature_range=(0, 1))),
        ('support vector machine', LinearRegression())
    ])

    lr_pipe.fit(model_data.X_train, model_data.y_train)

    y_pred = lr_pipe.predict(model_data.X_test)

    return lr_pipe, y_pred


def linear_regression_grid_search(pipe, X_train, y_train, X_test, y_test):
    print('------- Grid search --------')


def weigh_plot(df):
    # Get distance thickness and die_opening from df
    data = distance = df[['distance', 'thickness', 'die_opening']]
    target = df['springback']

    # linear_regression = make_pipeline(StandardScaler(),
    #                                   LinearRegression())

    linear_regression = make_pipeline(PolynomialFeatures(degree=2),
                                      LinearRegression())
    cv_results = cross_validate(linear_regression, data, target,
                                cv=10, scoring="neg_mean_squared_error",
                                return_train_score=True,
                                return_estimator=True)

    train_error = -cv_results["train_score"]
    print(f"Mean squared error of linear regression model on the train set:\n"
          f"{train_error.mean():.3f} ± {train_error.std():.3f}")

    model_first_fold = cv_results["estimator"][0]

    feature_names = model_first_fold[0].get_feature_names_out(
        input_features=data.columns)

    print(feature_names)

    coefs = [est[-1].coef_ for est in cv_results["estimator"]]
    weights_linear_regression = pd.DataFrame(coefs, columns=feature_names)

    color = {"whiskers": "black", "medians": "black", "caps": "black"}
    weights_linear_regression.plot.box(color=color, vert=False, figsize=(6, 16))
    _ = plt.title("Linear regression coefficients")
    plt.savefig('linear_regression_coefficients.png')


def feature_importance(X_train, y_train):
    plt.style.use(['science', 'ieee', 'grid'])
    model = make_pipeline(StandardScaler(),
                          LinearRegression())

    model.fit(X_train, y_train)

    coefs = pd.DataFrame(
        model[1].coef_,
        columns=['Coefficients'], index=X_train.columns
    )
    # coefs.plot(kind='barh')
    X_train.std(axis=0).plot(kind='barh')
    plt.title('Ridge model, small regularization')
    # plt.axvline(x=0, color='.5')
    # plt.xlim((0, 0.5))
    plt.subplots_adjust(left=.3)
    plt.savefig('linear_regression_coefficients.png', dpi=600)

    # X2 = sm.add_constant(X_test)
    # est = sm.OLS(y_test, X2)
    # est2 = est.fit()
    # print(est2.summary())


def pdp(X_train, y_train):
    # plt.style.use(['science', 'ieee', 'grid'])
    model = make_pipeline(LinearRegression())

    # model = LinearRegression()

    model.fit(X_train, y_train)
    features = ['distance', 'thickness', 'die_opening']

    display = PartialDependenceDisplay.from_estimator(model, X_train, features,
                                                      kind='both')
    plt.savefig('linear_regression_pdp.png', dpi=600)


if __name__ == '__main__':
    df = preprocessing.get_data()
    model_data = preprocessing.non_random_split(df, 30)
    model_data2 = preprocessing.random_split(df)

    # Create Linear Regression model
    model, y_pred = linear_regression(model_data)
    model2, y_pred2 = linear_regression(model_data2)

    # weigh_plot(df)
    # feature_importance(model_data.X_train, model_data.y_train)
    pdp(model_data.X_train, model_data.y_train)

    # reports = ['robustness']
    # create_reports(reports, model_data, model, y_pred)

    # create_reports(reports, model_data2, model2, y_pred2)


import numpy as np
from sklearn.model_selection import cross_val_score


def variance_of_cv(X, y, model):
    scores = cross_val_score(model, X, y, cv=5, scoring='r2')

    print(f'Cross-validation scores: {scores}')

    variance = np.var(scores, ddof=1)

    return variance


def calculate_r2(model, X_test, y_test):
    r2 = model.score(X_test, y_test)

    return r2


def relevance_report(model_data, model, y_pred):
    variance = variance_of_cv(model_data.X, model_data.y, model)
    print(f'Variance: {round(variance, 3)}')

    r2 = calculate_r2(model, model_data.X_test, model_data.y_test)
    print(f'R2: {round(r2, 3)}')

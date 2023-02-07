# import LOOCV
import numpy as np
import sklearn.metrics
from sklearn.model_selection import LeaveOneOut, cross_val_score


# Perform a leave one out cross validation
def loo_cv(X, y, model):
    loo = LeaveOneOut()

    scores = cross_val_score(model, X, y, cv=loo, scoring='neg_mean_squared_error')

    # Calculate average prediction error
    avg_error = -np.mean(scores)

    return avg_error


def stability_report(md, model, y_pred):
    print('------- STABILITY REPORT --------')

    avg_error = loo_cv(md.X, md.y, model)
    print(f'LOOCV: {avg_error}')

    print('\n------- END STABILITY REPORT --------\n')

# import LOOCV
import numpy as np
import pandas as pd
import sklearn.metrics
from sklearn.model_selection import LeaveOneOut, cross_val_score, KFold
from sklearn.model_selection import StratifiedKFold


# Perform a leave one out cross validation
def loo_cv(X, y, model):
    # Concat X and y to dataframe

    n_splits = 5

    # Evaluate the stability of the model using random k-fold cross-validation
    cv_rand = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_loocv = LeaveOneOut()

    scores_rand = cross_val_score(model, X, y, cv=cv_rand)
    scores_loocv = cross_val_score(model, X, y, cv=cv_loocv, scoring='neg_mean_squared_error')

    # loo = LeaveOneOut()
    # scores = cross_val_score(model, X, y, cv=loo, scoring='neg_mean_squared_error')

    # Calculate average prediction error
    # avg_error = -np.mean(scores)

    # Print the average and standard deviation of the scores
    print('Random K-Fold CV: Accuracy = {:.3f} ± {:.3f}'.format(scores_rand.mean(),
                                                                scores_rand.std()))
    # print('LOOCV K-Fold CV: Accuracy = {:.3f} ± {:.3f}'.format(scores_loocv.mean(),
    #                                                            scores_loocv.std()))

    return 2


def stability_report(md, model, y_pred):
    print('------- STABILITY REPORT --------')

    avg_error = loo_cv(md.X, md.y, model)
    print(f'LOOCV: {round(avg_error, 3)}')

    print('\n------- END STABILITY REPORT --------\n')

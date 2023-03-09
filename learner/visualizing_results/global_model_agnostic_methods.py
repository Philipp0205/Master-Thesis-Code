import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from yellowbrick.features import Rank1D

import learner.data_preprocessing as dp
import scienceplots
from yellowbrick.datasets import load_concrete
from yellowbrick.model_selection import FeatureImportances


def partial_dependence_plot(model, md, feature_names, name):
    plt.style.use(['science', 'ieee'])

    X = md.X
    y = md.y
    # Make two-way partial dependence plot for features die_opening and springback
    # 0 = distance, 1 = thickness,
    features = [1, 2, (1, 2)]
    display = PartialDependenceDisplay.from_estimator(model, X, features,
                                                        feature_names=feature_names,
                                                        n_cols=2)

    # display = PartialDependenceDisplay.from_estimator(model, feature_names)


    root = dp.get_root_directory()
    output_path = os.path.join(root, 'learner', 'visualizing_results', 'results',
                               'partial_dependence')

    # Save partial dependence plot
    display.plot()
    display.figure_.savefig(f'{output_path}/partial_dependence_die_opening{name}.png',
                            transparent=True,
                            dpi=600)


def feature_importance_plot(model, df, name):
    plt.style.use(['science'])

    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)

    feature_names = df.columns
    # Remove 'springback"
    feature_names = feature_names.drop('springback')

    forest_importances = pd.Series(importances, index=feature_names)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")

    root = dp.get_root_directory()
    output_path = os.path.join(root, 'learner', 'visualizing_results', 'results',
                               'partial_dependence')

    # fig.tight_layout()
    plt.savefig(f'{output_path}/feature_importances_{name}.png', transparent=True, dpi=600)

def permutation_feature_importance(model, md, df):
    # scoring = ['r2', 'neg_mean_absolute_percentage_error', 'neg_mean_squared_error']
    scoring = ['r2',  'neg_mean_squared_error']

    r_multi = permutation_importance(model, md.X_test, md.y_test,
                                     n_repeats=30,
                                     random_state=0,
                                     scoring=scoring)

    feature_names = df.columns.to_numpy()
    feature_names = feature_names[feature_names != 'springback']

    # Iterate over each metric r_multi and store the feature importance scores in a list
    importances = []
    for metric in r_multi:
        r = r_multi['r2']
        importances.append(r.importances_mean)

    plt.style.use(['science', 'ieee'])

    # Plot the feature importances using a bar chart
    plt.figure(figsize=(10,8))
    plt.bar(feature_names, importances[0])
    plt.xticks(rotation=90)
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.title('Permutation Feature Importance')
    plt.tight_layout()
    plt.savefig('permutation_feature_importance.png', transparent=True, dpi=600)

    # Iterate over each metric r_multi and print the feature name and its importance
    for metric in r_multi:
        print(f"{metric}")
        r = r_multi[metric]
        for i in r.importances_mean.argsort()[::-1]:
            print(f" {feature_names[i]:}"
                  f" {r.importances_mean[i]:.3f}"
                  f" +/-  {r.importances_std[i]:.6f}")
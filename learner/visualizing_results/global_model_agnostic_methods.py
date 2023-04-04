import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

import learner.data_preprocessing as dp
import scienceplots


def partial_dependence_plot(model, md, feature_names, name):
    plt.style.use(['science', 'ieee'])
    features = ['distance', 'thickness', 'die_opening']
    display = PartialDependenceDisplay.from_estimator(model, md.X, feature_names)

    root = dp.get_root_directory()
    output_path = os.path.join(root, 'learner', 'visualizing_results', 'results',
                               'partial_dependence')

    # Save partial dependence plot
    display.plot()
    display.figure_.savefig(f'{output_path}/partial_dependence_{name}.png', transparent=True,
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




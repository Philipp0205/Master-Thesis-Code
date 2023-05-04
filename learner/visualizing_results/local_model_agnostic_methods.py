import numpy as np
from lime import lime_tabular
from matplotlib import pyplot as plt
import learner.data_preprocessing as preprocessing


def create_lime_explanation(name, model, df, md):
    feature_names = df.columns
    feature_names = np.array(feature_names)

    X = md.X_train.values

    explainer = lime_tabular.LimeTabularExplainer(X, feature_names=md.X_train.columns,
                                                  class_names=['springback'],
                                                  kernel_width=5,
                                                  mode='regression',
                                                  )

    # Choose instance to explain
    instance = md.X_test.iloc[1, :]

    X_test = md.X_test

    # Get instances from md.X_test:
    # 1. die_opening 20, thickness 1.0, distance 10
    # 2. die_opening 20, thickness 3.0, distance 10

    instances = [
        X_test[(X_test['die_opening'] == 20) & (X_test['thickness'] == 2.0) & (X_test['distance'] == 10)].iloc[0, :],
        X_test[(X_test['die_opening'] == 20) & (X_test['thickness'] == 3.0) & (X_test['distance'] == 5)].iloc[0, :],

        X_test[(X_test['die_opening'] == 20) & (X_test['thickness'] == 1.5) & (X_test['distance'] == 10)].iloc[0, :],
        X_test[(X_test['die_opening'] == 30) & (X_test['thickness'] == 1.5) & (X_test['distance'] == 10)].iloc[0, :],

        X_test[(X_test['die_opening'] == 40) & (X_test['thickness'] == 0.5) & (X_test['distance'] == 10)].iloc[0, :],
        X_test[(X_test['die_opening'] == 50) & (X_test['thickness'] == 0.5) & (X_test['distance'] == 5)].iloc[0, :],
    ]

    plt.style.use(['science', 'ieee'])

    figs = []

    # Create lime explanations for all instances as sublplit
    for i, instance in enumerate(instances):
        exp = explainer.explain_instance(instance, model.predict, num_features=10)
        fig = exp.as_pyplot_figure()

        figs.append(fig)

        die_opening = instance['die_opening']
        thickness = instance['thickness']

        root = preprocessing.get_root_directory()
        fig.savefig(f'{root}/learner/visualizing_results/results/lime/{i}_lime_{name}_{die_opening}_{thickness}.png',
                    transparent=True, dpi=400)


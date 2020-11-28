from scikitplot.metrics import plot_roc, plot_confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None


def rf_results(model, X_test, y_pred_test, y_test, y_pred_train, y_train, verbose=True, plot=True):
    """Basic results to random forest classifier

    Args:
        model (sklearn): ML model
        X_test (pd.df): testing features
        y_pred_test (pd.df): predictions
        y_test (list): testing dependent variable
        y_pred_train (list): prediction dependent variable training
        y_train (list): dependent variable training
        verbose (bool, optional): printing options. Defaults to True.
        plot (bool, optional): ploting options. Defaults to True.

    Returns:
        dict: classification results
    """

    results = {
        'Test Accuracy': metrics.accuracy_score(y_test, y_pred_test),
        'OOB Score': model.oob_score_,
        'Train Accuracy': metrics.accuracy_score(y_train, y_pred_train),
    }

    if verbose:
        print('Random Forest Model Results:')
        print(results)

    if plot:
        y_true = y_test
        y_probas = model.predict_proba(X_test)
        plot_roc(y_true, y_probas, title='ROC Curves')
        plt.show()
        plot_confusion_matrix(y_pred_test, y_test)
        plt.show()

    return results


def plot_importance(model, features):
    """plots feature importance chart for random forest

    Args:
        model (sklearn.RandomForestX): ML model
        features (list): list of features in order to model
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)
    plt.title('Feature Importances')
    plt.barh(range(len(indices)),
             importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()

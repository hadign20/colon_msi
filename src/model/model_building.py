import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix, classification_report,
                             precision_score, recall_score, f1_score, roc_curve)
from scipy import stats
import seaborn as sns
from openpyxl import load_workbook


def get_classifiers():
    """
    Returns a dictionary of classifiers with their hyperparameter grids to be evaluated.
    """
    return {
        'RandomForest': (RandomForestClassifier(), {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth': [4, 6, 8, 10, 12],
            'criterion': ['gini', 'entropy']
        }),
        'SVM': (SVC(probability=True), {
            'C': [0.1, 1, 10, 100, 1000],
            'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
            'kernel': ['rbf', 'poly', 'sigmoid']
        }),
        'LogisticRegression': (LogisticRegression(), {
            'penalty': ['l1', 'l2'],
            'C': [0.01, 0.1, 1, 10, 100]
        }),
        'NaiveBayes': (GaussianNB(), {})
    }


def compute_metrics(y_true, y_pred, y_pred_prob):
    """
    Compute evaluation metrics and their confidence intervals.

    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.
    y_pred_prob (array-like): Predicted probabilities.

    Returns:
    dict: Evaluation metrics and their confidence intervals.
    """

    accuracy = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred_prob) if y_pred_prob is not None else None
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0
    sensitivity = recall_score(y_true, y_pred)
    ppv = precision_score(y_true, y_pred)
    npv = tn / (tn + fn) if (tn + fn) else 0
    f1 = f1_score(y_true, y_pred)

    metrics = {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'ppv': ppv,
        'npv': npv,
        'f1_score': f1
    }

    ci = {}
    for metric, value in metrics.items():
        if value is not None:
            ci[metric] = compute_confidence_interval(value, y_true.size)

    return metrics, ci


def compute_confidence_interval(metric, n, alpha=0.95):
    """
    Compute confidence interval for a metric.

    Parameters:
    metric (float): Metric value.
    n (int): Sample size.
    alpha (float): Confidence level.

    Returns:
    tuple: Lower and upper bounds of the confidence interval.
    """
    se = np.sqrt((metric * (1 - metric)) / n)
    h = se * stats.norm.ppf((1 + alpha) / 2)
    return metric - h, metric + h


def hyperparameter_tuning(clf, param_grid, X_train, y_train, name):
    """
    Perform hyperparameter tuning using GridSearchCV.

    Parameters:
    clf: The classifier to tune.
    param_grid (dict): The parameter grid to search over.
    X_train (pd.DataFrame): The training feature matrix.
    y_train (pd.Series): The training target vector.
    name (str): The name of the classifier.

    Returns:
    The best estimator found by GridSearchCV.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=skf, n_jobs=-1, scoring='roc_auc')
    grid_search.fit(X_train, y_train)

    # Save grid search results
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df.to_csv(f'{name}_grid_search_results.csv', index=False)

    print(f"Best parameters for {name}: {grid_search.best_params_}")
    print(f"Best cross-validation score for {name}: {grid_search.best_score_}")

    return grid_search.best_estimator_


def train_test_split_evaluation(X, y,
                      test_size=0.3,
                      random_state=42,
                      tuning=True):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    classifiers = get_classifiers()
    results = {}
    train_results = {}
    test_results = {}

    for classifier_name, (clf, param_grid) in classifiers.items():
        tuned_clf = hyperparameter_tuning(clf, param_grid, X_train, y_train, classifier_name)
        tuned_clf.fit(X_train, y_train)

        y_pred_train = tuned_clf.predict(X_train)
        y_pred_prob_train = tuned_clf.predict_proba(X_train)[:, 1] if hasattr(tuned_clf, "predict_proba") else None
        metrics_train, ci_train = compute_metrics(y_train, y_pred_train, y_pred_prob_train)
        train_results[classifier_name] = {
            'metrics': metrics_train,
            'confidence_intervals': ci_train
        }

        y_pred_test = tuned_clf.predict(X_test)
        y_pred_prob_test = tuned_clf.predict_proba(X_test)[:, 1] if hasattr(tuned_clf, "predict_proba") else None
        metrics_test, ci_test = compute_metrics(y_test, y_pred_test, y_pred_prob_test)
        test_results[classifier_name] = {
            'metrics': metrics_test,
            'confidence_intervals': ci_test
        }

        # Plot ROC curve for the test set
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob_test)
        roc_auc = roc_auc_score(y_test, y_pred_prob_test)
        plot_roc_curve(fpr, tpr, roc_auc, f'{classifier_name} ROC Curve', f'{classifier_name}_roc_curve.png')

        # Plot feature importance for tree-based models
        if classifier_name == 'RandomForest':
            plot_feature_importance(tuned_clf.feature_importances_, X.columns, 'Feature Importance',
                                    'feature_importance.png')

    results['train'] = train_results
    results['test'] = test_results

    return results



def cross_validation_evaluation(X, y, cv_folds=5, tuning=True):
    """
    Perform cross-validation and evaluate models.

    Parameters:
    X (pd.DataFrame): Feature matrix.
    y (pd.Series): Target vector.
    cv (int): Number of cross-validation folds.

    Returns:
    dict: Results for each classifier.
    """
    classifiers = get_classifiers()
    results = {}

    for name, (clf, param_grid) in classifiers.items():
        metrics_list = []
        best_estimators = []

        tuned_clf = hyperparameter_tuning(clf, param_grid, X, y, name)
        tuned_clf.fit(X, y)
        best_estimators.append(tuned_clf)

        print(f"best parameters for {name}", tuned_clf.best_params_)

        y_pred = tuned_clf.predict(X_test)
        y_pred_prob = tuned_clf.predict_proba(X_test)[:, 1] if hasattr(tuned_clf, "predict_proba") else None
        metrics, _ = compute_metrics(y_test, y_pred, y_pred_prob)
        metrics_list.append(metrics)

        # Average metrics and confidence intervals across folds
        averaged_metrics = {metric: np.mean([m[metric] for m in metrics_list if m[metric] is not None]) for metric in metrics_list[0]}
        ci = {metric: compute_confidence_interval(averaged_metrics[metric], y.size) for metric in averaged_metrics}

        results[name] = {
            'metrics': averaged_metrics,
            'confidence_intervals': ci
        }


    return results







def plot_roc_curve(fpr, tpr, auc, title, filename):
    """
    Plot ROC curve and save the plot.

    Parameters:
    fpr (array-like): False positive rates.
    tpr (array-like): True positive rates.
    auc (float): Area Under the Curve.
    title (str): Title of the plot.
    filename (str): Filename to save the plot.
    """
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(filename)
    plt.close()

def plot_feature_importance(importances, features, title, filename):
    """
    Plot feature importance and save the plot.

    Parameters:
    importances (array-like): Feature importances.
    features (array-like): Feature names.
    title (str): Title of the plot.
    filename (str): Filename to save the plot.
    """
    indices = np.argsort(importances)[::-1]
    plt.figure()
    plt.title(title)
    plt.bar(range(len(importances)), importances[indices], color="r", align="center")
    plt.xticks(range(len(importances)), [features[i] for i in indices], rotation=90)
    plt.xlim([-1, len(importances)])
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def save_classification_results(results, output_file, num_features, method='train_test_split'):
    """
    Save evaluation results to an Excel file.

    Parameters:
    results (dict): Evaluation results for each classifier.
    output_file (str): Path to save the Excel file.
    num_features (int): Number of features used in the classification.
    method (str): Method used for evaluation ('train_test_split' or 'cross_validation').
    """
    print(f"Saving evaluation results to {output_file} using method '{method}' with {num_features} features.")

    if method == 'train_test_split':
        rows = []
        for dataset, classification_results in results.items():
            for classifier, data in classification_results.items():
                metrics = data.get('metrics', {})
                ci = data.get('confidence_intervals', {})
                row = [
                    dataset.capitalize(),
                    classifier,
                    f"{metrics.get('roc_auc', 'N/A'):.2f} ({ci.get('roc_auc', ['N/A', 'N/A'])[0]:.2f}, {ci.get('roc_auc', ['N/A', 'N/A'])[1]:.2f})",
                    f"{metrics.get('sensitivity', 'N/A'):.2f} ({ci.get('sensitivity', ['N/A', 'N/A'])[0]:.2f}, {ci.get('sensitivity', ['N/A', 'N/A'])[1]:.2f})",
                    f"{metrics.get('specificity', 'N/A'):.2f} ({ci.get('specificity', ['N/A', 'N/A'])[0]:.2f}, {ci.get('specificity', ['N/A', 'N/A'])[1]:.2f})",
                    f"{metrics.get('ppv', 'N/A'):.2f} ({ci.get('ppv', ['N/A', 'N/A'])[0]:.2f}, {ci.get('ppv', ['N/A', 'N/A'])[1]:.2f})",
                    f"{metrics.get('npv', 'N/A'):.2f} ({ci.get('npv', ['N/A', 'N/A'])[0]:.2f}, {ci.get('npv', ['N/A', 'N/A'])[1]:.2f})",
                ]
                rows.append(row)

        df = pd.DataFrame(rows, columns=['Dataset', 'Classifier', 'AUC (95% CI)', 'Sensitivity (95% CI)',
                                         'Specificity (95% CI)',
                                         'PPV (95% CI)', 'NPV (95% CI)'])

    elif method == 'cross_validation':
        rows = []
        for classifier, data in results.items():
            metrics = data.get('metrics', {})
            ci = data.get('confidence_intervals', {})
            row = [
                classifier,
                f"{metrics.get('roc_auc', 'N/A'):.2f} ({ci.get('roc_auc', ['N/A', 'N/A'])[0]:.2f}, {ci.get('roc_auc', ['N/A', 'N/A'])[1]:.2f})",
                f"{metrics.get('sensitivity', 'N/A'):.2f} ({ci.get('sensitivity', ['N/A', 'N/A'])[0]:.2f}, {ci.get('sensitivity', ['N/A', 'N/A'])[1]:.2f})",
                f"{metrics.get('specificity', 'N/A'):.2f} ({ci.get('specificity', ['N/A', 'N/A'])[0]:.2f}, {ci.get('specificity', ['N/A', 'N/A'])[1]:.2f})",
                f"{metrics.get('ppv', 'N/A'):.2f} ({ci.get('ppv', ['N/A', 'N/A'])[0]:.2f}, {ci.get('ppv', ['N/A', 'N/A'])[1]:.2f})",
                f"{metrics.get('npv', 'N/A'):.2f} ({ci.get('npv', ['N/A', 'N/A'])[0]:.2f}, {ci.get('npv', ['N/A', 'N/A'])[1]:.2f})",
            ]
            rows.append(row)

        df = pd.DataFrame(rows, columns=['Classifier', 'AUC (95% CI)', 'Sensitivity (95% CI)', 'Specificity (95% CI)',
                                         'PPV (95% CI)', 'NPV (95% CI)'])

    else:
        raise ValueError("Invalid method. Choose 'train_test_split' or 'cross_validation'.")

    try:
        # Try to open an existing workbook
        book = load_workbook(output_file)
        writer = pd.ExcelWriter(output_file, engine='openpyxl')
        writer.book = book
    except FileNotFoundError:
        # If the file does not exist, create a new one
        writer = pd.ExcelWriter(output_file, engine='openpyxl')

    df.to_excel(writer, index=False, sheet_name=str(num_features))
    writer.save()
    writer.close()
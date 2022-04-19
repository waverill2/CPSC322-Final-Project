"""myevaluation.py 
@author Joshua Seward

Functions for evaluating classifiers in the mysklearn package.
"""

from math import ceil
import copy
import numpy as np

from mysklearn import myutils

def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
            Use random_state to seed your random number generator
                you can use the math module or use numpy for your generator
                choose one and consistently use that generator throughout your code
        shuffle(bool): whether or not to randomize the order of the instances before splitting
            Shuffle the rows in X and y before splitting and be sure to maintain the parallel order of X and y!!

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)

    Note:
        Loosely based on sklearn's train_test_split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    X_values = copy.deepcopy(X)
    y_values = copy.deepcopy(y)
    X_train, X_test, y_train, y_test = [], [], [], []
    if shuffle:
        myutils.randomize_in_place(a_list=X_values, parallel_list=y_values, random_state=random_state)
    if type(test_size) == float:
        num_test_instances = ceil(test_size*len(X_values))
        X_train = X_values[:-num_test_instances]
        X_test = X_values[-num_test_instances:]
        y_train = y_values[:-num_test_instances]
        y_test = y_values[-num_test_instances:]
        return X_train, X_test, y_train, y_test
    elif type(test_size) == int:
        X_train = X_values[:-test_size]
        X_test = X_values[-test_size:]
        y_train = y_values[:-test_size]
        y_test = y_values[-test_size:]
        return X_train, X_test, y_train, y_test

def kfold_cross_validation(X, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold
        X_test_folds(list of list of int): The list of testing set indices for each fold

    Notes:
        The first n_samples % n_splits folds have size n_samples // n_splits + 1,
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    X_indices = list(range(len(X)))
    n_samples = len(X_indices)
    if shuffle:
        myutils.randomize_in_place(a_list=X_indices, random_state=random_state)
    # create folds
    folds = []
    index = 0
    for folds_index in range(n_splits):
        current_fold = []
        if folds_index < (n_samples % n_splits):
            for _ in range(n_samples // n_splits + 1):
                current_fold.append(X_indices[index])
                index += 1
        else:
            for _ in range(n_samples // n_splits):
                current_fold.append(X_indices[index])
                index += 1
        folds.append(current_fold)
    # create train and test folds
    X_train_folds, X_test_folds = [], []
    for fold in folds:
        X_test_folds.append(fold)
        train_fold = []
        for index in range(n_samples):
            if index not in fold:
                train_fold.append(index)
        X_train_folds.append(train_fold)
    return X_train_folds, X_test_folds

def stratified_kfold_cross_validation(X, y, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into stratified cross validation folds.

    Args:
        X(list of list of obj): The list of instances (samples).
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X).
            The shape of y is n_samples
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold.
        X_test_folds(list of list of int): The list of testing set indices for each fold.

    Notes:
        Loosely based on sklearn's StratifiedKFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    X_indices = list(range(len(X)))
    y_values = copy.deepcopy(y)
    n_samples = len(X_indices)
    if shuffle:
        myutils.randomize_in_place(a_list=X_indices, parallel_list=y_values, random_state=random_state)
    # create stratified folds
    groups = myutils.group_by_result(instance_indices=X_indices, results=y_values)
    folds = [[] for _ in range(n_splits)]
    index = 0
    for group in groups:
        for instance_index in group:
            folds[index].append(instance_index)
            if index + 1 >= len(folds):
                index = 0
            else:
                index += 1
    # create train and test folds
    X_train_folds, X_test_folds = [], []
    for fold in folds:
        X_test_folds.append(fold)
        train_fold = []
        for index in range(n_samples):
            if index not in fold:
                train_fold.append(index)
        X_train_folds.append(train_fold)
    return X_train_folds, X_test_folds

def bootstrap_sample(X, y=None, n_samples=None, random_state=None):
    """Split dataset into bootstrapped training set and out of bag test set.

    Args:
        X(list of list of obj): The list of samples
        y(list of obj): The target y values (parallel to X)
            Default is None (in this case, the calling code only wants to sample X)
        n_samples(int): Number of samples to generate. If left to None (default) this is automatically
            set to the first dimension of X.
        random_state(int): integer used for seeding a random number generator for reproducible results
    Returns:
        X_sample(list of list of obj): The list of samples
        X_out_of_bag(list of list of obj): The list of "out of bag" samples (e.g. left-over samples)
        y_sample(list of obj): The list of target y values sampled (parallel to X_sample)
            None if y is None
        y_out_of_bag(list of obj): The list of target y values "out of bag" (parallel to X_out_of_bag)
            None if y is None
    Notes:
        Loosely based on sklearn's resample():
            https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html
    """
    if n_samples is None:
        n_samples = len(X)
    if random_state is not None:
        np.random.seed(random_state)
    X_sample = []
    X_out_of_bag = copy.deepcopy(X)
    if y is not None:
        y_sample = []
        y_out_of_bag = copy.deepcopy(y)
    else:
        y_sample, y_out_of_bag = None, None
    for _ in range(n_samples):
        sample_index = np.random.randint(0, len(X))
        random_instance = copy.deepcopy(X[sample_index])
        X_sample.append(random_instance)
        if y is not None:
            random_result = copy.deepcopy(y[sample_index])
            y_sample.append(random_result)
        if random_instance in X_out_of_bag:
            X_out_of_bag.remove(random_instance)
            if y is not None:
                y_out_of_bag.remove(random_result)
    return X_sample, X_out_of_bag, y_sample, y_out_of_bag

def confusion_matrix(y_true:list, y_pred:list, labels:list):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry
            indicates the number of samples with true label being i-th class
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    # create shape of the confusion matrix
    confusion_matrix = [[0 for _ in labels] for _ in labels]
    # fill confusion matrix
    for index in range(len(y_pred)):
        predicted_value = y_pred[index]
        actual_value = y_true[index]
        confusion_matrix[labels.index(actual_value)][labels.index(predicted_value)] += 1
    return confusion_matrix

def accuracy_score(y_true, y_pred, normalize=True):
    """Compute the classification prediction accuracy score.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        normalize(bool): If False, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.

    Returns:
        score(float): If normalize == True, return the fraction of correctly classified samples (float),
            else returns the number of correctly classified samples (int).

    Notes:
        Loosely based on sklearn's accuracy_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    """
    num_correctly_predicted = 0
    for index in range(len(y_pred)):
        if y_pred[index] == y_true[index]:
            num_correctly_predicted += 1
    if normalize:
        return num_correctly_predicted / len(y_pred)
    else:
        return num_correctly_predicted

def binary_precision_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the precision (for binary classification). The precision is the ratio tp / (tp + fp)
        where tp is the number of true positives and fp the number of false positives.
        The precision is intuitively the ability of the classifier not to label as
        positive a sample that is negative. The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        precision(float): Precision of the positive class

    Notes:
        Loosely based on sklearn's precision_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    """
    if labels is None:
        labels = list(set(y_true))
    if pos_label is None:
        pos_label = labels[0]
    false_positives = 0
    true_positives = 0
    for index in range(len(y_pred)):
        if y_pred[index] == pos_label:
            if y_pred[index] == y_true[index]:
                true_positives += 1
            else:
                false_positives += 1
    precision_score = 0.0 if true_positives + false_positives == 0 else true_positives / (true_positives + false_positives)
    return precision_score

def binary_recall_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the recall (for binary classification). The recall is the ratio tp / (tp + fn) where tp is
        the number of true positives and fn the number of false negatives.
        The recall is intuitively the ability of the classifier to find all the positive samples.
        The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        recall(float): Recall of the positive class

    Notes:
        Loosely based on sklearn's recall_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    """
    if labels is None:
        labels = list(set(y_true))
    if pos_label is None:
        pos_label = labels[0]
    true_positives = 0
    false_negatives = 0
    for index in range(len(y_pred)):
        if y_true[index] == pos_label:
            if y_pred[index] == y_true[index]:
                true_positives += 1
            else:
                false_negatives += 1
    recall_score = 0.0 if (true_positives + false_negatives) == 0 else true_positives / (true_positives + false_negatives)
    return recall_score

def binary_f1_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the F1 score (for binary classification), also known as balanced F-score or F-measure.
        The F1 score can be interpreted as a harmonic mean of the precision and recall,
        where an F1 score reaches its best value at 1 and worst score at 0.
        The relative contribution of precision and recall to the F1 score are equal.
        The formula for the F1 score is: F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        f1(float): F1 score of the positive class

    Notes:
        Loosely based on sklearn's f1_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """
    if labels is None:
        labels = list(set(y_true))
    if pos_label is None:
        pos_label = labels[0]
    precision = binary_precision_score(y_true, y_pred, labels, pos_label)
    recall = binary_recall_score(y_true, y_pred, labels, pos_label)
    f1_score = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)
    return f1_score

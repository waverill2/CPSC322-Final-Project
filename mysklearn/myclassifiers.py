"""myclassifiers.py
@author Joshua Seward

Classifier classes for the mysklearn package.
"""
import numpy as np
from scipy import rand
# from sqlalchemy import all_

from mysklearn import myutils
from mysklearn import myevaluation
from mysklearn.mysimplelinearregressor import MySimpleLinearRegressor

class MySimpleLinearRegressionClassifier:
    """Represents a simple linear regression classifier that discretizes
        predictions from a simple linear regressor (see MySimpleLinearRegressor).

    Attributes:
        discretizer(function): a function that discretizes a numeric value into
            a string label. The function's signature is func(obj) -> obj
        regressor(MySimpleLinearRegressor): the underlying regression model that
            fits a line to x and y data

    Notes:
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self, discretizer, regressor:MySimpleLinearRegressor = None):
        """Initializer for MySimpleLinearClassifier.

        Args:
            discretizer(function): a function that discretizes a numeric value into
                a string label. The function's signature is func(obj) -> obj
            regressor(MySimpleLinearRegressor): the underlying regression model that
                fits a line to x and y data (None if to be created in fit())
        """
        self.discretizer = discretizer
        if regressor == None:
            self.regressor = MySimpleLinearRegressor()
        else: 
            self.regressor = regressor

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """
        self.regressor.fit(X_train=X_train, y_train=y_train)

    def predict(self, X_test):
        """Makes predictions for test samples in X_test by applying discretizer
            to the numeric predictions from regressor.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        predicted_values = self.regressor.predict(X_test=X_test)
        return [self.discretizer(value) for value in predicted_values]

class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train
        self.y_train = y_train

    def kneighbors(self, X_test):
        """Determines the k closest neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        distances = []
        indices = []
        for test_instance in X_test:
            test_distances = []
            test_indices = []
            for index, train_instance in enumerate(self.X_train):
                test_distances.append(myutils.euclidean_distance(train_instance, test_instance))
                test_indices.append(index)
            distance_index = zip(test_distances, test_indices)
            sorted_distance_index = sorted(distance_index)
            test_distances, test_indices = zip(*sorted_distance_index)
            distances.append(list(test_distances))
            indices.append(list(test_indices))
        top_k_distances = [row[:self.n_neighbors] for row in distances]
        top_k_indices = [row[:self.n_neighbors] for row in indices]
        return top_k_distances, top_k_indices

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        predicted_values_list = []
        _, k_closest_indices_list = self.kneighbors(X_test=X_test)
        for k_closest_indices in k_closest_indices_list:
            predicted_values = dict()
            for index in k_closest_indices:
                predicted_value = self.y_train[index]
                if predicted_value in predicted_values:
                    predicted_values[predicted_value] += 1
                else:
                    predicted_values[predicted_value] = 1
            predicted_values_list.append(max(predicted_values, key=predicted_values.get))
        return predicted_values_list

class MyDummyClassifier:
    """Represents a "dummy" classifier using the "most_frequent" strategy.
        The most_frequent strategy is a Zero-R classifier, meaning it ignores
        X_train and produces zero "rules" from it. Instead, it only uses
        y_train to see what the most frequent class label is. That is
        always the dummy classifier's prediction, regardless of X_test.

    Attributes:
        most_common_label(obj): whatever the most frequent class label in the
            y_train passed into fit()

    Notes:
        Loosely based on sklearn's DummyClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
    """
    def __init__(self):
        """Initializer for DummyClassifier.

        """
        self.most_common_label = None

    def fit(self, X_train, y_train):
        """Fits a dummy classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Zero-R only predicts the most frequent class label, this method
                only saves the most frequent class label.
        """
        class_labels = dict()
        for class_label in y_train:
            if class_label in class_labels:
                class_labels[class_label] += 1
            else:
                class_labels[class_label] = 1
        self.most_common_label = max(class_labels, key=class_labels.get)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        return [self.most_common_label for _ in X_test]


class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        posteriors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The posterior probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.priors = None
        self.posteriors = None

    def fit(self, X_train: list, y_train: list):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        # get priors
        self.priors = dict()
        priors_count = dict()
        for classification in y_train:
            if priors_count.get(classification) is None:
                priors_count[classification] = 1
            else:
                priors_count[classification] += 1
        self.priors = dict(sorted(priors_count.items(), key=lambda item: item[0]))
        number_of_classifications = len(y_train)
        for prior in list(self.priors.keys()):
            self.priors[prior] = priors_count[prior]/number_of_classifications
        # get posteriors
        self.posteriors = dict()
        # create a dict for each attribute
        for attribute_number in range(len(X_train[0])):
            attribute_key = "att" + str(attribute_number+1)
            self.posteriors[attribute_key] = dict()
        # fill the posteriors list
        for instance_index in range(len(X_train)):
            instance_classification = y_train[instance_index]
            for attribute_number in range(len(X_train[instance_index])):
                attribute_name = "att" + str(attribute_number+1)
                attribute_value = X_train[instance_index][attribute_number]
                # create the dictionaries for each attribute value's classification
                if self.posteriors[attribute_name].get(attribute_value) is None:
                    self.posteriors[attribute_name][attribute_value] = dict()
                    # fill the dictionary with 0 for each classification
                    for classification in list(self.priors.keys()):
                        self.posteriors[attribute_name][attribute_value][classification] = 0
                self.posteriors[attribute_name][attribute_value][instance_classification] += 1
        # divide the counted values of the posteriors by the counted number for the instance's classification
        for attribute_name, attribute_values in self.posteriors.items():
            for attribute_value, attribute_classicications in attribute_values.items():
                for classification, classification_count in attribute_classicications.items():
                    self.posteriors[attribute_name][attribute_value][classification] = classification_count/priors_count[classification]

    def predict(self, X_test:list):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        predictions = list()
        for test_instance in X_test:
            classification_predictions = dict()
            for classification, prior in self.priors.items():
                # multiply the prior by the posterior for each attribute of the test instance
                classification_prediction = prior
                for attribute_index in range(len(test_instance)):
                    attribute_name = "att" + str(attribute_index+1)
                    attribute_value = test_instance[attribute_index]
                    classification_prediction = classification_prediction * self.posteriors[attribute_name][attribute_value][classification]
                classification_predictions[classification] = classification_prediction
            # get the maximum product and get the class it is associated with
            classes = list(classification_predictions.keys())
            values = list(classification_predictions.values())
            predictions.append(classes[values.index(max(values))])
        return predictions


class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.tree = None

    def fit(self, X_train: list, y_train: list) -> None:
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            On a majority vote tie, choose first attribute value based on attribute domain ordering.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
        header = list()
        attribute_domains = dict()
        for index in range(len(X_train[0])):
            attribute_name = 'att' + str(index)
            # programatically extract the header (e.g. att0, att1, ...)
            header.append(attribute_name)
            attribute_domains[attribute_name] = list()
        # programatically extract the attribute domains
        for instance in X_train:
            for index in range(len(instance)):
                value = instance[index]
                attribute_name = 'att' + str(index)
                attribute_domains[attribute_name].append(value)
        for attribute in list(attribute_domains.keys()):
            domain = attribute_domains[attribute]
            attribute_domains[attribute] = sorted(list(set(domain)))
        # stitch X_train and y_train together so the class label stays with the instance
        train = [X_train[index] + [y_train[index]] for index in range(len(X_train))]
        # next, make a copy of your header - tdidt is going to modify the list
        available_attributes = header.copy()
        # RECALL - Python is pass by object reference
        self.tree = myutils.tdidt(current_instances=train, header=header, attribute_domains=attribute_domains,
            available_attributes=available_attributes)

    def predict(self, X_test: list) -> list:
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        predictions = list()
        header = ["att" + str(index) for index in range(len(self.X_train))]
        for test_sample in X_test:
            predictions.append(myutils.predict_instance(instance=test_sample, header=header, tree=self.tree))
        return predictions

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format
        "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        # extract attribute names if they are not given
        tree_header = ["att" + str(index) for index in range(len(self.X_train[0]))]
        if attribute_names is None:
            attribute_names = tree_header.copy()
        decision_rules = myutils.generate_decision_rules(tree=self.tree, tree_header=tree_header,\
            attribute_names=attribute_names, class_name=class_name)
        if len(decision_rules) == 0:
            print("No decision rules were generated - I had a lot of trouble doing this")
        else:
            for rule in decision_rules:
                print(rule)

    # BONUS method
    def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
        """BONUS: Visualizes a tree via the open source Graphviz graph visualization package and
        its DOT graph language (produces .dot and .pdf files).

        Args:
            dot_fname(str): The name of the .dot output file.
            pdf_fname(str): The name of the .pdf output file generated from the .dot file.
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).

        Notes:
            Graphviz: https://graphviz.org/
            DOT language: https://graphviz.org/doc/info/lang.html
            You will need to install graphviz in the Docker container as shown in class to complete this method.
        """
        pass # TODO: (BONUS) fix this

class MyRandomForestClassifier():
    """Represents a Random Forest Classifier.

    Attributes:
        X_train (list of list of obj): training data used to generate the forest classifier
        y_train (list of obj): classifications for the given training data
        forest_classifier (list of MyDecisionTreeClassifier): trained decision tree classifiers that make up the random
        forest classifer
    """
    def __init__(self) -> None:
        """Initializer for MyRandomForestClassifier.
        """
        self.forest_classifier = None
    
    def fit(self, X_train: list, y_train: list, N: int, F: int, M: int, random_state: int = None) -> None:
        """Generates N "random" decision trees using X_train and y_train and takes the M most accurate decision trees to
        make up the forest classifer.

        Args:
            X_train (list of list of obj): training data to build the decision tree classifiers from
            y_train (list of obj): classifications for the given training data
            N (int): number of decision tree classifiers to build
            F (int): size of the attribute sets used to build the decision trees
            M (int): number of desired decision trees to create the random forest classifier from
            random_state (int): random state to seed the bootstrap sample random number generator (helpful for testing)
        """
        # build N decision trees based off bootstrapped samples of the given X_train and y_train lists
        all_decision_trees = list()
        for iteration in range(0, N):
            # use a modified random state for reproduceability (helpful for testing)
            random_state_bootstrap = random_state + iteration if random_state is not None else None
            X_sample, X_validation, y_sample, y_validation = myevaluation.bootstrap_sample(X=X_train, y=y_train, n_samples=F, random_state=random_state_bootstrap)
            decision_tree = MyDecisionTreeClassifier()
            decision_tree.fit(X_train=X_sample, y_train=y_sample)
            # calculate the accuracy of the decision tree
            y_predictions = decision_tree.predict(X_test=X_validation)
            decision_tree_accuracy = myevaluation.accuracy_score(y_true=y_validation, y_pred=y_predictions)
            all_decision_trees.append([decision_tree, decision_tree_accuracy])
        all_decision_trees = sorted(all_decision_trees, key=lambda l:l[1])[:M]
        self.forest_classifier = [decision_tree_row[0] for decision_tree_row in all_decision_trees]

    def predict(self, test_instances: list) -> list:
        """Makes predictions for the given test instances based on the list of decision tree classifiers. Uses majority
        voting to make predictions based on the M most accurate decision trees.

        Args:
            test_instances (list of list of obj): test instances to classify

        Returns:
            list of obj: predictions for each given test instance
        """
        all_predictions = list()
        for test_instance in test_instances:
            instance_forest_predictions = dict()
            for decision_tree in self.forest_classifier:
                # get a prediction from each decision tree in the forest classifier
                prediction = decision_tree.predict([test_instance])
                if instance_forest_predictions.get(prediction[0]) is None:
                    instance_forest_predictions[prediction[0]] = 1
                else:
                    instance_forest_predictions[prediction[0]] += 1
            instance_forest_predictions = dict(sorted(instance_forest_predictions.items(), key=lambda item: item[0]))
            # select the most commonly predicted value from the forest of decision trees
            max_instance_prediction_count = max(list(instance_forest_predictions.values()))
            for prediction, prediction_count in instance_forest_predictions.items():
                if max_instance_prediction_count == prediction_count:
                    all_predictions.append(prediction)
                    break
        return all_predictions

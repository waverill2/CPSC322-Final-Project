"""test_myclassifiers.py 
@author Joshua Seward

Tests for classifiers in the mysklearn package.
"""

import numpy as np

from mysklearn.myclassifiers import MyNaiveBayesClassifier
from mysklearn.mysimplelinearregressor import MySimpleLinearRegressor
from mysklearn.myclassifiers import MySimpleLinearRegressionClassifier,\
    MyKNeighborsClassifier,\
    MyDummyClassifier,\
    MyDecisionTreeClassifier, \
    MyRandomForestClassifier
from mysklearn.myutils import basic_discretizer1, basic_discretizer2

# constants used in test_simple_linear_regression_classifier_fit
np.random.seed(0) # seed the random number generator for reproducible test results
X_TRAIN_BASIC_EXAMPLE1 = [[value] for value in range(100)]
Y_TRAIN_BASIC_EXAMPLE1 = [row[0] * 2 + np.random.normal(0, 25) for row in X_TRAIN_BASIC_EXAMPLE1]
X_TEST_BASIC_EXAMPLE1 = [[150], [-150], [0], [100], [500]]
SLOPE_BASIC_EXAMPLE1 = 1.924917458430444
INTERCEPT_BASIC_EXAMPLE1 = 5.211786196055144
Y_PREDICTED_BASIC_EXAMPLE1 = ["high", "low", "low", "high", "high"]
X_TRAIN_BASIC_EXAMPLE2 = [[value] for value in range(50)]
Y_TRAIN_BASIC_EXAMPLE2 = [row[0] / 3 + np.random.normal(0, 15) for row in X_TRAIN_BASIC_EXAMPLE2]
SLOPE_BASIC_EXAMPLE2 = 0.1878862
INTERCEPT_BASIC_EXAMPLE2 = 7.4114773
X_TEST_BASIC_EXAMPLE2 = [[75], [-95], [0], [-10], [-150]]
Y_PREDICTED_BASIC_EXAMPLE2 = ["positive", "not positive", "positive", "positive", "not positive"]

# constants used in test_kneighbors_classifier_kneighbors and test_kneighbors_classifier_predict
X_TRAIN_CLASS_EXAMPLE1 = [[1, 1], 
                          [1, 0], 
                          [0.33, 0], 
                          [0, 0]]
Y_TRAIN_CLASS_EXAMPLE1 = ["bad", "bad", "good", "good"]
X_TEST_CLASS_EXAMPLE1 = [[0.33, 1]]
DISTANCES_CLASS_EXAMPLE1 = [[.67, 1.0, 1.05]]
INDICES_CLASS_EXAMPLE1 = [[0, 2, 3]]
Y_PREDICTED_CLASS_EXAMPLE1 = ["good"]
X_TRAIN_CLASS_EXAMPLE2 = [[3, 2],
                          [6, 6],
                          [4, 1],
                          [4, 4],
                          [1, 2],
                          [2, 0],
                          [0, 3],
                          [1, 6]]
Y_TRAIN_CLASS_EXAMPLE2 = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
X_TEST_CLASS_EXAMPLE2 = [[2,3]]
DISTANCES_CLASS_EXAMPLE2 = [[1.414214, 1.414214, 2]]
INDICES_CLASS_EXAMPLE2 = [[0, 4, 6]]
Y_PREDICTED_CLASS_EXAMPLE2 = ["yes"]
HEADER_BRAMER_EXAMPLE = ["Attribute 1", "Attribute 2"]
X_TRAIN_BRAMER_EXAMPLE = [[0.8, 6.3],
                          [1.4, 8.1],
                          [2.1, 7.4],
                          [2.6, 14.3],
                          [6.8, 12.6],
                          [8.8, 9.8],
                          [9.2, 11.6],
                          [10.8, 9.6],
                          [11.8, 9.9],
                          [12.4, 6.5],
                          [12.8, 1.1],
                          [14.0, 19.9],
                          [14.2, 18.5],
                          [15.6, 17.4],
                          [15.8, 12.2],
                          [16.6, 6.7],
                          [17.4, 4.5],
                          [18.2, 6.9],
                          [19.0, 3.4],
                          [19.6, 11.1]]
Y_TRAIN_BRAMER_EXAMPLE = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-",\
                          "-", "-", "+", "+", "+", "-", "+"]
X_TEST_BRAMER_EXAMPLE = [[9.1, 11.0]]
DISTANCES_BRAMER_EXAMPLE = [[0.608, 1.237, 2.202, 2.802, 2.915]]
INDICES_BRAMER_EXAMPLE = [[6, 5, 7, 4, 8]]
Y_PREDICTED_BRAMER_EXAMPLE = ["+"]

# constants used in test_dummy_classifier_fit and test_dummy_classifier_predict
X_TRAIN_DUMMY_EXAMPLE = [[value] for value in range(100)]
X_TEST_DUMMY_EXAMPLE = [[150], [-10], [200]]
Y_TRAIN_DUMMY_EXAMPLE1 = list(np.random.choice(["yes", "no"], 100, replace=True, p=[0.7, 0.3]))
Y_PREDICTED_DUMMY_EXAMPLE1 = ["yes", "yes", "yes"]
Y_TRAIN_DUMMY_EXAMPLE2 = list(np.random.choice(["yes", "no", "maybe"], 100, replace=True, p=[0.2, 0.6, 0.2]))
Y_PREDICTED_DUMMY_EXAMPLE2 = ["no", "no", "no"]
Y_TRAIN_DUMMY_EXAMPLE3 = list(np.random.choice(["up", "down", "left", "right"], 100, replace=True, p=[0.25, 0.15, 0.2, 0.4]))
Y_PREDICTED_DUMMY_EXAMPLE3 = ["right", "right", "right"]

# constants used in test_naive_bayes_classifier_fit andtest_naive_bayes_classifier_predict
COL_NAMES_NAIVE_BAYES_INCLASS = ["att1", "att2"]
X_TRAIN_NAIVE_BAYES_INCLASS = [
    [1, 5], # yes
    [2, 6], # yes
    [1, 5], # no
    [1, 5], # no
    [1, 6], # yes
    [2, 6], # no
    [1, 5], # yes
    [1, 6] # yes
]
Y_TRAIN_NAIVE_BAYES_INCLASS = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"]
PRIORS_NAIVE_BAYES_INCLASS = {"no":3/8, "yes":5/8}
POSTERIORS_NAIVE_BAYES_INCLASS = {"att1":{1:{"no":2/3, "yes":4/5},
                                          2:{"no":1/3, "yes":1/5}},
                                  "att2":{5:{"no":2/3, "yes":2/5},
                                          6:{"no":1/3, "yes":3/5}}}
X_TEST_NAIVE_BAYES_INCLASS = [[1,5]]
Y_ACTUAL_NAIVE_BAYES_INCLASS = ["yes"]
COL_NAMES_NAIVE_BAYES_IPHONE = ["standing", "job_status", "credit_rating"]
X_TRAIN_IPHONE = [
    ["1", "3", "fair"],
    ["1", "3", "excellent"],
    ["2", "3", "fair"],
    ["2", "2", "fair"],
    ["2", "1", "fair"],
    ["2", "1", "excellent"],
    ["2", "1", "excellent"],
    ["1", "2", "fair"],
    ["1", "1", "fair"],
    ["2", "2", "fair"],
    ["1", "2", "excellent"],
    ["2", "2", "excellent"],
    ["2", "3", "fair"],
    ["2", "2", "excellent"],
    ["2", "3", "fair"]
]
Y_TRAIN_IPHONE = ["no", "no", "yes", "yes", "yes", "no", "yes", "no",
                              "yes", "yes", "yes", "yes", "yes", "no", "yes"]
PRIORS_NAIVE_BAYES_IPHONE = {"no":5/15, "yes":10/15}
POSTERIORS_NAIVE_BAYES_IPHONE = {"att1":{        "1":{"no":3/5, "yes":2/10},
                                                 "2":{"no":2/5, "yes":8/10}},
                                 "att2":{        "1":{"no":1/5, "yes":3/10},
                                                 "2":{"no":2/5, "yes":4/10},
                                                 "3":{"no":2/5, "yes":3/10}},
                                 "att3":{     "fair":{"no":2/5, "yes":7/10},
                                         "excellent":{"no":3/5, "yes":3/10}}}
X_TEST_IPHONE = [["2", "2", "fair"]]
Y_ACTUAL_IPHONE = ["yes"]
COL_NAMES_NAIVE_BAYES_BRAMER = ["day", "season", "wind", "rain"]
X_TRAIN_NAIVE_BAYES_BRAMER = [
    ["weekday", "spring", "none", "none"],
    ["weekday", "winter", "none", "slight"],
    ["weekday", "winter", "none", "slight"],
    ["weekday", "winter", "high", "heavy"],
    ["saturday", "summer", "normal", "none"],
    ["weekday", "autumn", "normal", "none"],
    ["holiday", "summer", "high", "slight"],
    ["sunday", "summer", "normal", "none"],
    ["weekday", "winter", "high", "heavy"],
    ["weekday", "summer", "none", "slight"],
    ["saturday", "spring", "high", "heavy"],
    ["weekday", "summer", "high", "slight"],
    ["saturday", "winter", "normal", "none"],
    ["weekday", "summer", "high", "none"],
    ["weekday", "winter", "normal", "heavy"],
    ["saturday", "autumn", "high", "slight"],
    ["weekday", "autumn", "none", "heavy"],
    ["holiday", "spring", "normal", "slight"],
    ["weekday", "spring", "normal", "none"],
    ["weekday", "spring", "normal", "slight"]
]
Y_TRAIN_NAIVE_BAYES_BRAMER = ["on time", "on time", "on time", "late", "on time", "very late",
                              "on time", "on time", "very late", "on time", "cancelled", "on time",
                              "late", "on time", "very late", "on time", "on time", "on time",
                              "on time", "on time"]
PRIORS_NAIVE_BAYES_BRAMER = {"cancelled":1/20, "late":2/20, "on time":14/20, "very late":3/20}
POSTERIORS_NAIVE_BAYES_BRAMER = {"att1":{ "weekday":{"cancelled":0/1, "late":1/2, "on time":9/14, "very late":3/3},
                                         "saturday":{"cancelled":1/1, "late":1/2, "on time":2/14, "very late":0/3},
                                           "sunday":{"cancelled":0/1, "late":0/2, "on time":1/14, "very late":0/3},
                                          "holiday":{"cancelled":0/1, "late":0/2, "on time":2/14, "very late":0/3}},
                                 "att2":{  "spring":{"cancelled":1/1, "late":0/2, "on time":4/14, "very late":0/3},
                                           "summer":{"cancelled":0/1, "late":0/2, "on time":6/14, "very late":0/3},
                                           "autumn":{"cancelled":0/1, "late":0/2, "on time":2/14, "very late":1/3},
                                           "winter":{"cancelled":0/1, "late":2/2, "on time":2/14, "very late":2/3}},
                                 "att3":{    "none":{"cancelled":0/1, "late":0/2, "on time":5/14, "very late":0/3},
                                             "high":{"cancelled":1/1, "late":1/2, "on time":4/14, "very late":1/3},
                                           "normal":{"cancelled":0/1, "late":1/2, "on time":5/14, "very late":2/3}},
                                 "att4":{    "none":{"cancelled":0/1, "late":1/2, "on time":5/14, "very late":1/3},
                                           "slight":{"cancelled":0/1, "late":0/2, "on time":8/14, "very late":0/3},
                                            "heavy":{"cancelled":1/1, "late":1/2, "on time":1/14, "very late":2/3}}}
X_TEST_NAIVE_BAYES_BRAMER = [["weekday", "winter", "high", "heavy"]]
Y_ACTUAL_NAIVE_BAYES_BRAMER = ["very late"]

# constants used in test_decision_tree_classifier_fit and test_decision_tree_classifier_predict
IN_CLASS_INTERVIEW_HEADER = ["level", "lang", "tweets", "phd", "interviewed_well"]
IN_CLASS_INTERVIEW_TABLE = [
        ["Senior", "Java", "no", "no", "False"],
        ["Senior", "Java", "no", "yes", "False"],
        ["Mid", "Python", "no", "no", "True"],
        ["Junior", "Python", "no", "no", "True"],
        ["Junior", "R", "yes", "no", "True"],
        ["Junior", "R", "yes", "yes", "False"],
        ["Mid", "R", "yes", "yes", "True"],
        ["Senior", "Python", "no", "no", "False"],
        ["Senior", "R", "yes", "no", "True"],
        ["Junior", "Python", "yes", "no", "True"],
        ["Senior", "Python", "yes", "yes", "True"],
        ["Mid", "Python", "no", "yes", "True"],
        ["Mid", "Java", "yes", "no", "True"],
        ["Junior", "Python", "no", "yes", "False"]
    ]
X_TRAIN_IN_CLASS_INTERVIEW = [row[:len(row)-1] for row in IN_CLASS_INTERVIEW_TABLE]
Y_TRAIN_IN_CLASS_INTERVIEW = [row[len(row)-1] for row in IN_CLASS_INTERVIEW_TABLE]
# note: this tree uses the generic "att#" attribute labels because fit() does not and should not accept attribute names
# note: the attribute values are sorted alphabetically
IN_CLASS_INTERVIEW_DECISION_TREE = \
        ["Attribute", "att0",
            ["Value", "Junior", 
                ["Attribute", "att3",
                    ["Value", "no", 
                        ["Leaf", "True", 3, 5]
                    ],
                    ["Value", "yes", 
                        ["Leaf", "False", 2, 5]
                    ]
                ]
            ],
            ["Value", "Mid",
                ["Leaf", "True", 4, 14]
            ],
            ["Value", "Senior",
                ["Attribute", "att2",
                    ["Value", "no",
                        ["Leaf", "False", 3, 5]
                    ],
                    ["Value", "yes",
                        ["Leaf", "True", 2, 5]
                    ]
                ]
            ]
        ]
X_TEST_IN_CLASS_INTERVIEW = [
    ["Junior", "Java", "yes", "no"],
    ["Junior", "Java", "yes", "yes"],
]
Y_ACTUAL_IN_CLASS_INTERVIEW = ["True", "False"]
BRAMER_DEGREES_HEADER = ["SoftEng", "ARIN", "HCI", "CSA", "Project", "Class"]
BRAMER_DEGREES_TABLE = [
        ["A", "B", "A", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "A", "FIRST"],
        ["A", "A", "A", "B", "B", "SECOND"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["A", "A", "B", "B", "A", "FIRST"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "B", "SECOND"],
        ["A", "A", "A", "A", "A", "FIRST"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["A", "B", "B", "A", "B", "SECOND"],
        ["B", "B", "B", "B", "A", "SECOND"],
        ["A", "A", "B", "A", "B", "FIRST"],
        ["B", "B", "B", "B", "A", "SECOND"],
        ["A", "A", "B", "B", "B", "SECOND"],
        ["B", "B", "B", "B", "B", "SECOND"],
        ["A", "A", "B", "A", "A", "FIRST"],
        ["B", "B", "B", "A", "A", "SECOND"],
        ["B", "B", "A", "A", "B", "SECOND"],
        ["B", "B", "B", "B", "A", "SECOND"],
        ["B", "A", "B", "A", "B", "SECOND"],
        ["A", "B", "B", "B", "A", "FIRST"],
        ["A", "B", "A", "B", "B", "SECOND"],
        ["B", "A", "B", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "B", "SECOND"],
    ]
X_TRAIN_BRAMER_DEGREES = [row[:len(row)-1] for row in BRAMER_DEGREES_TABLE]
Y_TRAIN_BRAMER_DEGREES = [row[len(row)-1] for row in BRAMER_DEGREES_TABLE]
BRAMER_DEGREES_DECISION_TREE = \
    ["Attribute", "att0",
        ["Value", "A",
            ["Attribute", "att4",
                ["Value", "A",
                    ["Leaf", "FIRST", 5, 14],
                ],
                ["Value", "B",
                    ["Attribute", "att3",
                        ["Value", "A",
                            ["Attribute", "att1",
                                ["Value", "A",
                                    ["Leaf", "FIRST", 1, 2],
                                ],
                                ["Value", "B",
                                    ["Leaf", "SECOND", 1, 2],
                                ],
                            ],
                        ],
                        ["Value", "B",
                            ["Leaf", "SECOND", 7, 9],
                        ],
                    ],
                ],
            ],
        ],
        ["Value", "B",
            ["Leaf", "SECOND", 12, 26],
        ],
    ]
X_TEST_BRAMER_DEGREES = [
    ["B", "B", "B", "B", "B"], 
    ["A", "A", "A", "A", "A"], 
    ["A", "A", "A", "A", "B"],
]
Y_ACTUAL_BRAMER_DEGREES = ["SECOND", "FIRST", "FIRST"]
IPHONE_HEADER = ["standing", "job_status", "credit_rating", "buys_iphone"]
# X_TRAIN, Y_TRAIN, X_TEST, and Y_TEST are made earlier in the naive bayes constants
IPHONE_DECISION_TREE = \
    ["Attribute", "att0",
        ["Value", "1",
            ["Attribute", "att1",
                ["Value", "1", 
                    ["Leaf", "yes", 1, 5],
                ],
                ["Value", "2",
                    ["Attribute", "att2", 
                        ["Value", "excellent",
                            ["Leaf", "yes", 1, 2],
                        ],
                        ["Value", "fair", 
                            ["Leaf", "no", 1, 2],
                        ],
                    ],
                ],
                ["Value", "3",
                    ["Leaf", "no", 2, 5],
                ],
            ],
        ],
        ["Value", "2",
            ["Attribute", "att2",
                ["Value", "excellent",
                    ["Leaf", "no", 4, 10],
                ],
                ["Value", "fair",
                    ["Leaf", "yes", 6, 10],
                ],
            ],
        ],
    ]

def test_simple_linear_regression_classifier_fit():
    # test against the basic example results from class
    linear_regressor = MySimpleLinearRegressor()
    linear_regression_classifier = MySimpleLinearRegressionClassifier(basic_discretizer1, linear_regressor)
    linear_regression_classifier.fit(X_train=X_TRAIN_BASIC_EXAMPLE1, y_train=Y_TRAIN_BASIC_EXAMPLE1)
    assert np.isclose(linear_regression_classifier.regressor.slope, SLOPE_BASIC_EXAMPLE1)
    assert np.isclose(linear_regression_classifier.regressor.intercept, INTERCEPT_BASIC_EXAMPLE1)
    # test against a custom test case
    linear_regressor = MySimpleLinearRegressor()
    linear_regression_classifier = MySimpleLinearRegressionClassifier(basic_discretizer2, linear_regressor)
    linear_regression_classifier.fit(X_train=X_TRAIN_BASIC_EXAMPLE2, y_train=Y_TRAIN_BASIC_EXAMPLE2)
    assert np.isclose(linear_regression_classifier.regressor.slope, SLOPE_BASIC_EXAMPLE2)
    assert np.isclose(linear_regression_classifier.regressor.intercept, INTERCEPT_BASIC_EXAMPLE2)

def test_simple_linear_regression_classifier_predict():
    # test against the basic example results from class
    linear_regressor = MySimpleLinearRegressor()
    linear_regression_classifier = MySimpleLinearRegressionClassifier(basic_discretizer1, linear_regressor)
    linear_regression_classifier.fit(X_TRAIN_BASIC_EXAMPLE1, Y_TRAIN_BASIC_EXAMPLE1)
    y_predicted = linear_regression_classifier.predict(X_TEST_BASIC_EXAMPLE1)
    assert y_predicted == Y_PREDICTED_BASIC_EXAMPLE1
    # test against the a custom test case
    linear_regressor = MySimpleLinearRegressor()
    linear_regression_classifier = MySimpleLinearRegressionClassifier(basic_discretizer2, linear_regressor)
    linear_regression_classifier.fit(X_TRAIN_BASIC_EXAMPLE2, Y_TRAIN_BASIC_EXAMPLE2)
    y_predicted = linear_regression_classifier.predict(X_TEST_BASIC_EXAMPLE2)
    assert y_predicted == Y_PREDICTED_BASIC_EXAMPLE2

def test_kneighbors_classifier_kneighbors():
    # test against the distances and neighbor indices found in class for example 1
    kneighbors_classifier = MyKNeighborsClassifier()
    kneighbors_classifier.fit(X_TRAIN_CLASS_EXAMPLE1, Y_TRAIN_CLASS_EXAMPLE1)
    distances, neighbor_indices = kneighbors_classifier.kneighbors(X_TEST_CLASS_EXAMPLE1)
    assert np.allclose(distances, DISTANCES_CLASS_EXAMPLE1, atol=1e-2)
    assert np.allclose(neighbor_indices, INDICES_CLASS_EXAMPLE1)
    # test against the distances and neighbor indices found in class for example 2
    kneighbors_classifier = MyKNeighborsClassifier()
    kneighbors_classifier.fit(X_TRAIN_CLASS_EXAMPLE2, Y_TRAIN_CLASS_EXAMPLE2)
    distances, neighbor_indices = kneighbors_classifier.kneighbors(X_TEST_CLASS_EXAMPLE2)
    assert np.allclose(distances, DISTANCES_CLASS_EXAMPLE2, atol=1e-3)
    assert np.allclose(neighbor_indices, INDICES_CLASS_EXAMPLE2)
    # test against the distances and neighbor indices for the example in the Bramer textbook
    kneighbors_classifier = MyKNeighborsClassifier(n_neighbors=5)
    kneighbors_classifier.fit(X_TRAIN_BRAMER_EXAMPLE, Y_TRAIN_BRAMER_EXAMPLE)
    distances, neighbor_indices = kneighbors_classifier.kneighbors(X_TEST_BRAMER_EXAMPLE)
    assert np.allclose(distances, DISTANCES_BRAMER_EXAMPLE, atol=1e-3)
    assert np.allclose(neighbor_indices, INDICES_BRAMER_EXAMPLE)

def test_kneighbors_classifier_predict():
    # test against the y_predicted value found in class for example 1
    kneighbors_classifier = MyKNeighborsClassifier()
    kneighbors_classifier.fit(X_TRAIN_CLASS_EXAMPLE1, Y_TRAIN_CLASS_EXAMPLE1)
    y_predicted = kneighbors_classifier.predict(X_TEST_CLASS_EXAMPLE1)
    assert y_predicted == Y_PREDICTED_CLASS_EXAMPLE1
    # test against the y_predicted value found in class for example 2
    kneighbors_classifier = MyKNeighborsClassifier()
    kneighbors_classifier.fit(X_TRAIN_CLASS_EXAMPLE2, Y_TRAIN_CLASS_EXAMPLE2)
    y_predicted = kneighbors_classifier.predict(X_TEST_CLASS_EXAMPLE2)
    assert y_predicted == Y_PREDICTED_CLASS_EXAMPLE2
    # test against the y_predicted value for the example in the Bramer textbook
    kneighbors_classifier = MyKNeighborsClassifier()
    kneighbors_classifier.fit(X_TRAIN_BRAMER_EXAMPLE, Y_TRAIN_BRAMER_EXAMPLE)
    y_predicted = kneighbors_classifier.predict(X_TEST_BRAMER_EXAMPLE)
    assert y_predicted == Y_PREDICTED_BRAMER_EXAMPLE

def test_dummy_classifier_fit():
    # test against the first given test case
    dummy_classifier = MyDummyClassifier()
    dummy_classifier.fit(X_TRAIN_DUMMY_EXAMPLE, Y_TRAIN_DUMMY_EXAMPLE1)
    assert dummy_classifier.most_common_label == "yes"
    # test against the second given test case
    dummy_classifier = MyDummyClassifier()
    dummy_classifier.fit(X_TRAIN_DUMMY_EXAMPLE, Y_TRAIN_DUMMY_EXAMPLE2)
    assert dummy_classifier.most_common_label == "no"
    # test against the custom test case
    dummy_classifier = MyDummyClassifier()
    dummy_classifier.fit(X_TRAIN_DUMMY_EXAMPLE, Y_TRAIN_DUMMY_EXAMPLE3)
    assert dummy_classifier.most_common_label == "right"

def test_dummy_classifier_predict():
    # test against the first given test case
    dummy_classifier = MyDummyClassifier()
    dummy_classifier.fit(X_TRAIN_DUMMY_EXAMPLE, Y_TRAIN_DUMMY_EXAMPLE1)
    y_predicted = dummy_classifier.predict(X_TEST_DUMMY_EXAMPLE)
    assert y_predicted == Y_PREDICTED_DUMMY_EXAMPLE1
    # test against the second given test case
    dummy_classifier = MyDummyClassifier()
    dummy_classifier.fit(X_TRAIN_DUMMY_EXAMPLE, Y_TRAIN_DUMMY_EXAMPLE2)
    y_predicted = dummy_classifier.predict(X_TEST_DUMMY_EXAMPLE)
    assert y_predicted == Y_PREDICTED_DUMMY_EXAMPLE2
    # test against the custom test case
    dummy_classifier = MyDummyClassifier()
    dummy_classifier.fit(X_TRAIN_DUMMY_EXAMPLE, Y_TRAIN_DUMMY_EXAMPLE3)
    y_predicted = dummy_classifier.predict(X_TEST_DUMMY_EXAMPLE)
    assert y_predicted == Y_PREDICTED_DUMMY_EXAMPLE3

def test_naive_bayes_classifier_fit():
    naive_bayes_classifier = MyNaiveBayesClassifier()
    # test against the in class test case 
    naive_bayes_classifier.fit(X_TRAIN_NAIVE_BAYES_INCLASS, Y_TRAIN_NAIVE_BAYES_INCLASS)
    assert naive_bayes_classifier.priors == PRIORS_NAIVE_BAYES_INCLASS
    assert naive_bayes_classifier.posteriors == POSTERIORS_NAIVE_BAYES_INCLASS
    # test against the iphone test case
    naive_bayes_classifier.fit(X_TRAIN_IPHONE, Y_TRAIN_IPHONE)
    assert naive_bayes_classifier.priors == PRIORS_NAIVE_BAYES_IPHONE
    assert naive_bayes_classifier.posteriors == POSTERIORS_NAIVE_BAYES_IPHONE
    # test against the Bramer test case
    naive_bayes_classifier.fit(X_TRAIN_NAIVE_BAYES_BRAMER, Y_TRAIN_NAIVE_BAYES_BRAMER)
    assert naive_bayes_classifier.priors == PRIORS_NAIVE_BAYES_BRAMER
    assert naive_bayes_classifier.posteriors == POSTERIORS_NAIVE_BAYES_BRAMER

def test_naive_bayes_classifier_predict():
    naive_bayes_classifier = MyNaiveBayesClassifier()
    # test against the in class test case 
    naive_bayes_classifier.fit(X_TRAIN_NAIVE_BAYES_INCLASS, Y_TRAIN_NAIVE_BAYES_INCLASS)
    y_predicted = naive_bayes_classifier.predict(X_TEST_NAIVE_BAYES_INCLASS)
    assert y_predicted == Y_ACTUAL_NAIVE_BAYES_INCLASS
    # test against the iphone test case
    naive_bayes_classifier.fit(X_TRAIN_IPHONE, Y_TRAIN_IPHONE)
    y_predicted = naive_bayes_classifier.predict(X_TEST_IPHONE)
    assert y_predicted == Y_ACTUAL_IPHONE
    # test against the Bramer test case
    naive_bayes_classifier.fit(X_TRAIN_NAIVE_BAYES_BRAMER, Y_TRAIN_NAIVE_BAYES_BRAMER)
    y_predicted = naive_bayes_classifier.predict(X_TEST_NAIVE_BAYES_BRAMER)
    assert y_predicted == Y_ACTUAL_NAIVE_BAYES_BRAMER

def test_decision_tree_classifier_fit():
    decision_tree_classifier = MyDecisionTreeClassifier()
    # test against the in class "interview" decision tree test case
    # print("\nTest Case 1")
    decision_tree_classifier.fit(X_train=X_TRAIN_IN_CLASS_INTERVIEW, y_train=Y_TRAIN_IN_CLASS_INTERVIEW)
    assert decision_tree_classifier.X_train == X_TRAIN_IN_CLASS_INTERVIEW
    assert decision_tree_classifier.y_train == Y_TRAIN_IN_CLASS_INTERVIEW
    assert decision_tree_classifier.tree == IN_CLASS_INTERVIEW_DECISION_TREE
    # test against the Bramer "degrees" decision tree test case
    # print("Test Case 2")
    decision_tree_classifier.fit(X_train=X_TRAIN_BRAMER_DEGREES, y_train=Y_TRAIN_BRAMER_DEGREES)
    assert decision_tree_classifier.X_train == X_TRAIN_BRAMER_DEGREES
    assert decision_tree_classifier.y_train == Y_TRAIN_BRAMER_DEGREES
    assert decision_tree_classifier.tree == BRAMER_DEGREES_DECISION_TREE
    # test against RQ "iPhone" decision tree test case
    # print("Test Case 3")
    decision_tree_classifier.fit(X_train=X_TRAIN_IPHONE, y_train=Y_TRAIN_IPHONE)
    assert decision_tree_classifier.X_train == X_TRAIN_IPHONE
    assert decision_tree_classifier.y_train == Y_TRAIN_IPHONE
    assert decision_tree_classifier.tree == IPHONE_DECISION_TREE

def test_decision_tree_classifier_predict():
    decision_tree_classifier = MyDecisionTreeClassifier()
    # test against the in class "interview" decision tree test case
    # print("\nTest Case 1")
    decision_tree_classifier.fit(X_train=X_TRAIN_IN_CLASS_INTERVIEW, y_train=Y_TRAIN_IN_CLASS_INTERVIEW)
    predictions = decision_tree_classifier.predict(X_test=X_TEST_IN_CLASS_INTERVIEW)
    assert predictions == Y_ACTUAL_IN_CLASS_INTERVIEW
    # test against the Bramer "degrees" decision tree test case
    # print("Test Case 2")
    decision_tree_classifier.fit(X_train=X_TRAIN_BRAMER_DEGREES, y_train=Y_TRAIN_BRAMER_DEGREES)
    predictions = decision_tree_classifier.predict(X_test=X_TEST_BRAMER_DEGREES)
    assert predictions == Y_ACTUAL_BRAMER_DEGREES
    # test against RQ "iPhone" decision tree test case
    # print("Test Case 3")
    decision_tree_classifier.fit(X_train=X_TRAIN_IPHONE, y_train=Y_TRAIN_IPHONE)
    predictions = decision_tree_classifier.predict(X_test=X_TEST_IPHONE)
    assert predictions == Y_ACTUAL_IPHONE

def test_random_forest_classifier_predict():
    random_forest_classifier = MyRandomForestClassifier()
    num_decision_trees = 15
    attribute_set_size = len(IN_CLASS_INTERVIEW_TABLE)
    num_trees_in_forest = 10
    random_forest_classifier.fit(X_train=X_TRAIN_IN_CLASS_INTERVIEW, y_train=Y_TRAIN_IN_CLASS_INTERVIEW,
        N=num_decision_trees, F=attribute_set_size, M=num_trees_in_forest)
    predictions = random_forest_classifier.predict(test_instances=X_TEST_IN_CLASS_INTERVIEW)
    assert predictions == Y_ACTUAL_IN_CLASS_INTERVIEW

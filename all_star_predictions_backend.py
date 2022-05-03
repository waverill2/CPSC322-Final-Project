"""all_star_predictions_backend.py
@author Joshua Seward

Flask backend for the baseball all-star prediction app.
"""
from flask import Flask, request, jsonify, render_template
from os import environ
from pickle import load

app = Flask(__name__)

# constant discretization bin values (calculated earlier using equal-width binning)
GAMES_PLAYED_BINS = {
    1.0: [1.0, 21.25],
    2.0: [21.25, 41.5],
    3.0: [41.5, 61.75],
    4.0: [61.75, 82.0],
    5.0: [82.0, 102.5],
    6.0: [102.5, 122.5],
    7.0: [122.5, 142.75],
    8.0: [142.75, 163.0],
}
AT_BATS_BINS = {
    1.0: [0.0, 89.5],
    2.0: [89.5, 179.0],
    3.0: [179.0, 268.5],
    4.0: [268.5, 358.0],
    5.0: [358.0, 447.5],
    6.0: [447.5, 537.0],
    7.0: [537.0, 626.5],
    8.0: [626.5, 716.0],
}
RUNS_BINS = {
    1.0: [0.0, 19.0],
    2.0: [19.0, 38.0],
    3.0: [38.0, 57.0],
    4.0: [57.0, 76.0],
    5.0: [76.0, 95.0],
    6.0: [95.0, 114.0],
    7.0: [114.0, 133.0],
    8.0: [133.0, 152.0],
}
HITS_BINS = {
    1.0: [0.0, 32.75],
    2.0: [32.75, 65.5],
    3.0: [65.5, 98.25],
    4.0: [98.25, 131.0],
    5.0: [131.0, 163.75],
    6.0: [163.75, 196.5],
    7.0: [196.5, 229.25],
    8.0: [229.25, 262.0],
}
DOUBLES_BINS = {
    1.0: [0.0, 7.38],
    2.0: [7.38, 14.75],
    3.0: [14.75, 22.12],
    4.0: [22.12, 29.5],
    5.0: [29.5, 36.88],
    6.0: [36.88, 44.24],
    7.0: [44.24, 51.62],
    8.0: [51.62, 59.0],
}
TRIPLES_BINS = {
    1.0: [0.0, 2.88],
    2.0: [2.88, 5.75],
    3.0: [5,75, 8.62],
    4.0: [8.62, 11.5],
    5.0: [11.5, 14.38],
    6.0: [14.38, 17.25],
    7.0: [17.25, 20.12],
    8.0: [20.12, 23.0],
}
HOME_RUNS_BINS = {
    1.0: [0.0, 9.12],
    2.0: [9.12, 18.25],
    3.0: [18.25, 27.38],
    4.0: [27.38, 36.5],
    5.0: [36.5, 45.62],
    6.0: [45.62, 54.75],
    7.0: [54.75, 63.88],
    8.0: [63.88, 73.0],
}
RUNS_BATTED_IN_BINS = {
    1.0: [0.0, 20.0],
    2.0: [20.0, 40.0],
    3.0: [40.0, 60.0],
    4.0: [60.0, 80.0],
    5.0: [80.0, 100.0],
    6.0: [100.0, 120.0],
    7.0: [120.0, 140.0],
    8.0: [140.0, 160.0],
}
STOLEN_BASES_BINS = {
    1.0: [0.0, 9.75],
    2.0: [9.75, 19.5],
    3.0: [19.5, 29.25],
    4.0: [29.25, 39.0],
    5.0: [39.0, 48.75],
    6.0: [48.75, 58.5],
    7.0: [58.5, 68.25],
    8.0: [68.25, 78.0],
}
WALKS_BINS = {
    1.0: [0.0, 29.0],
    2.0: [29.0, 58.0],
    3.0: [58.0, 87.0],
    4.0: [87.0, 116.0],
    5.0: [116.0, 145.0],
    6.0: [145.0, 174.0],
    7.0: [174.0, 203.0],
    8.0: [203.0, 232.0],
}
STRIKEOUTS_BINS = {
    1.0: [0.0, 27.88],
    2.0: [27.88, 55.75],
    3.0: [55.75, 83.62],
    4.0: [83.62, 111.5],
    5.0: [111.5, 139.38],
    6.0: [139.38, 167.25],
    7.0: [167.25, 195.12],
    8.0: [195.12, 223.0],
}
PUTOUTS_BINS = {
    1.0: [0.0, 199.62],
    2.0: [199.62, 399.25],
    3.0: [399.25, 598.88],
    4.0: [598.88, 798.5],
    5.0: [798.5, 998.12],
    6.0: [998.12, 1197.75],
    7.0: [1197.75, 1397.38],
    8.0: [1397.38, 1597.0],
}
ASSISTS_BINS = {
    1.0: [0.0, 70.12],
    2.0: [70.12, 140.25],
    3.0: [140.25, 210.38],
    4.0: [210.38, 280.5],
    5.0: [280.5, 350.62],
    6.0: [350.62, 420.75],
    7.0: [420.75, 490.88],
    8.0: [490.88, 561.0],
}
ERRORS_BINS = {
    1.0: [0.0, 4.5],
    2.0: [4.5, 9.0],
    3.0: [9.0, 13.5],
    4.0: [13.5, 18.0],
    5.0: [18.0, 22.5],
    6.0: [22.5, 27.0],
    7.0: [27.0, 31.5],
    8.0: [31.5, 36.0],
}
DISCRETIZED_INSTANCE_BINS = [None, GAMES_PLAYED_BINS, AT_BATS_BINS, RUNS_BINS, HITS_BINS, DOUBLES_BINS, TRIPLES_BINS,
    HOME_RUNS_BINS, RUNS_BATTED_IN_BINS, STOLEN_BASES_BINS, WALKS_BINS, STRIKEOUTS_BINS, PUTOUTS_BINS, ASSISTS_BINS, 
    ERRORS_BINS]

@app.route('/', methods=['GET'])
def index():
    """Route for the root page of the all-star predictor app. 

    Methods = GET

    Status Codes:
        200 - OK
    """
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    """Route for the predict API endpoint for the all star prediction app.

    Methods = POST

    Status Codes:
        200 - prediction was made succesfully
        400 - prediction was None (client error)
    """
    # get all the different fields required for the prediction instance
    year = request.form['year']
    games_played = request.form['games_played']
    at_bats = request.form['at_bats']
    runs = request.form['runs']
    hits = request.form['hits']
    doubles = request.form['doubles']
    triples = request.form['triples']
    home_runs = request.form['home_runs']
    runs_batted_in = request.form['runs_batted_in']
    stolen_bases = request.form['stolen_bases']
    walks = request.form['walks']
    strikeouts = request.form['strikeouts']
    putouts = request.form['putouts']
    assists = request.form['assists']
    errors = request.form['errors']
    instance = list()
    instance = [year, games_played, at_bats, runs, hits, doubles, triples, home_runs, runs_batted_in, stolen_bases, \
        walks, strikeouts, putouts, assists, errors]
    # predict the class for the given data and return it as JSON
    prediction = predict_all_star(instance=instance)
    if prediction == None:
        return render_template('error_making_prediction.html')
    return render_template('all_star_prediction.html', prediction=prediction)

def predict_all_star(instance:list) -> str:
    """Method to predict whether the given baseball player instance is an all star.

    Args:
        instance (list of str):

    Returns:
        str: predicted class if the player instance is an all all star (None if something goes wrong)
    """
    # unpickle the random forest classifier
    pickle_file = open('forest_classifier.p', 'rb')
    pickle_forest_classifier = load(pickle_file)
    pickle_file.close()
    # if the instance is not complete, return None
    for value in instance:
        if value is None:
            return None
    # discretize the instance so it can be used to make a prediction
    discretized_instance = discretize_instance(instance)
    print(instance)
    print(discretized_instance)
    # predict the discretized instance using the pickle forest classifier (only one prediction in the predictions list)
    prediction = pickle_forest_classifier.predict([discretized_instance])[0]
    return prediction

def discretize_instance(instance: list) -> list:
    """Discretizes each value in the given instance so a preiction can be made.

    Args:
        instance (list of str): instance to discretize the values of

    Returns:
        list of str: instance with each value discretized to match the training dataset used
                     for the random forest classifier
    """
    discretized_instance = instance.copy()
    # discretize each value in the instance as necessary
    for index in range(len(discretized_instance)):
        discretized_value = discretize_by_index(discretized_instance[index], index)
        discretized_instance[index] = discretized_value
    return discretized_instance

def discretize_by_index(value, index: int) -> float:
    """Discretizes the value at a given index of an instance. The index determines which stat is being discretized.

    Args:
        value (numeric): value to be discretized
        index (int): index of the value in the row

    Returns:
        float: discretized value
    """
    bins = DISCRETIZED_INSTANCE_BINS[index]
    numeric_value = float(value)
    if bins is None: # year - convert to float
        return numeric_value
    for discretized_value, bin_min_max in bins.items():
        bin_min = bin_min_max[0]
        bin_max = bin_min_max[1]
        if numeric_value >= bin_min and numeric_value < bin_max:
            return discretized_value
    # case for the value being higher than the last bin max
    max_bin_value = bins[len(bins.items())][1]
    if numeric_value > max_bin_value:
        return 8.0

if __name__ == '__main__':
    port = environ.get('PORT', 5000)
    app.run(debug=True, port=port) # TURN DEBUG MODE TO FALSE

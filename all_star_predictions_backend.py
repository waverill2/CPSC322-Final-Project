"""all_star_predictions_backend.py
@author Joshua Seward

Flask backend for the baseball all-star prediction app.
"""
from flask import Flask, request, jsonify, render_template
from os import environ
from pickle import load

app = Flask(__name__)

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
    # TODO - discretize the instance so it can be used to make a prediction
    # predict the discretized instance using the pickle forest classifier (only one prediction in the predictions list)
    prediction = pickle_forest_classifier.predict([instance])[0]
    return prediction

if __name__ == '__main__':
    port = environ.get('PORT', 5000)
    app.run(debug=True, port=port) # TURN DEBUG MODE TO FALSE

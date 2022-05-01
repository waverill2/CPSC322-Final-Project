"""all_star_predictions_backend.py
@author Joshua Seward

Flask backend for the baseball all-star prediction app.
"""
from flask import Flask, request, jsonify
from os import environ
from pickle import load

APP_NAME = 'Baseball All-Star Predictor'
app = Flask(APP_NAME)

@app.route('/', methods=['GET'])
def index():
    """Route for the root page of the all-star predictor app. 
    TODO - create an html template to take instances to predict.

    Methods = GET

    Status Codes:
        200 - OK
    """
    return f'<h1 style=\"text-align: center\"> Welcome to the {APP_NAME} App'

@app.route('/predict', methods=['GET'])
def predict():
    """Route for the predict API endpoint for the all star prediction app. 
    TODO - create an html template to display predictions.

    Methods = GET

    Status Codes:
        200 - prediction was made succesfully
        400 - prediction was None (client error)
    """
    # get all the different fields required for the prediction instance
    year = request.args.get('year', '')
    games_played = request.args.get('games_played', '')
    at_bats = request.args.get('at_bats', '')
    runs = request.args.get('runs', '')
    hits = request.args.get('hits', '')
    doubles = request.args.get('doubles', '')
    triples = request.args.get('triples', '')
    home_runs = request.args.get('home_runs', '')
    runs_batted_in = request.args.get('runs_batted_in', '')
    stolen_bases = request.args.get('stolen_bases', '')
    walks = request.args.get('walks', '')
    strikeouts = request.args.get('strikeouts', '')
    putouts = request.args.get('putouts', '')
    assists = request.args.get('assists', '')
    errors = request.args.get('errors', '')
    instance = [year, games_played, at_bats, runs, hits, doubles, triples, home_runs, runs_batted_in, stolen_bases, \
        walks, strikeouts, putouts, assists, errors, instance]
    # predict the class for the given data and return it as JSON
    prediction = predict_all_star(instance=instance)
    if prediction == None:
        return 'error making prediction', 400
    result = {'prediction' : prediction}
    return jsonify(result), 200

def predict_all_star(instance:list) -> str:
    """Method to predict whether the given baseball player instance is an all star.

    Args:
        instance (list of str):

    Returns:
        str: predicted class if the player instance is an all all star (None if something goes wrong)
    """
    # unpickle the random forest classifier
    pickle_file = open('forest_classifer.p', 'rb')
    pickle_forest_classifier = load(pickle_file)
    pickle_file.close()

    # predict the given instance using the pickle forest classifier (only one prediction in the predictions list)
    prediction = pickle_forest_classifier.predict([instance])[0]
    return prediction

if __name__ == '__main__':
    port = environ.get('PORT', 5000)
    app.run(debug=True, port=port) # TURN DEBUG MODE TO FALSE

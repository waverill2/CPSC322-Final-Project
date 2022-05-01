"""all_star_predictions_backend.py
@author Joshua Seward

Flask backend for the baseball all-star prediction app.
"""
from flask import Flask
import os

APP_NAME = 'Baseball All-Star Predictor'
app = Flask(APP_NAME)

@app.route('/', methods=['GET'])
def index():
    """Route for the root page of the all-star predictor app. TODO - create an html template to take predictions

    Methods = GET

    Status Codes:
        200- OK
    """
    return f'<h1 style=\"text-align: center\"> Welcome to the {APP_NAME} App'

if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    app.run(debug=True, port=port) # TURN DEBUG MODE TO FALSE

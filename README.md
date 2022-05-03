# Baseball All-Star Predictor  
Our project uses the random forest tree to predict whether a given set of stats (presumably for a baseball player in Major League Baseball) would result in an All-Star selection.

## Running the App  
To run the Flask app, simply enter the following command in your terminal (from the directory the project is in):

``` python all_star_predictions_backend.py ```

Next, naivgate to the link provided in the terminal after the app starts and make your predictions!

## Project Directory Organization  
* input_data - all of the input data taken from [kaggle's](https://www.kaggle.com/datasets/seanlahman/the-history-of-baseball?resource=download) "History of Baseball" dataset
* mysklearn - data science module built up throughout the semester (files were selected from either f the project collaborators)
* output_data - cleaned datasets that were used to create "final_table.txt" (the final table used to train the classifier)
* templates - html templates for the Flask app (Jinja2 style)
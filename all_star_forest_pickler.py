"""all_star_forest_pickler.py
@author Joshua Seward

Simple pickler file for a random forest classifier to predict a baseball player's all-star status.
"""
from os import path
import pickle

from mysklearn.myclassifiers import MyRandomForestClassifier
from mysklearn.mypytable import MyPyTable

# load the data from the 'final_table' file
FINAL_DATA_TABLE_PATH = path.join('output_data', 'final_table.txt')
all_star_data_table = MyPyTable()
all_star_data_table.load_from_file(FINAL_DATA_TABLE_PATH)

# create and train the forest classifier from the table data
all_star_forest_classifer = MyRandomForestClassifier()
y_train = all_star_data_table.get_column('class')
all_star_data_table.drop_column('class')
X_train = all_star_data_table.data
all_star_forest_classifer.fit(X_train=X_train, y_train=y_train, N=20, F=500, M=10)

# pickle the trained random forest clasifier
pickle_file = open('forest_classifier.p', 'wb')
pickle.dump(all_star_forest_classifer)
pickle_file.close()

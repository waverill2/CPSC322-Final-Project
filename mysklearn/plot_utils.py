import matplotlib.pyplot as plt
from sympy import rotations

import mysklearn.utils as utils

def plot_bar_chart(x_values: list, y_values: list, title: str=None, x_axis_name: str = None,
    y_axis_name: str = None, x_tick_labels:list = None, y_tick_labels: list = None):
    """Plots a bar graph using matplotlib.

    Args:
        x_values (list): list of values to plot along the x-axis
        y_values (list): list of values to plot along the y-axis
        title (str): OPTIONAL title to display on the graph 
        x_axis_name (str): OPTIONAL name for the x-axis
        y_axis_name (str): OPTIONAL name for the y-axis
        x_tick_labels (list of str): OPTIONAL parallel list to x-axis values for the names at each 
        value
        y_tick_labels (list of str): OPTIONAL parallel list to y-axis values for the names at each 
        value
    """
    plt.figure()
    plt.bar(x_values, y_values)
    if title != None:
        plt.title(title)
    if x_axis_name != None:
        plt.xlabel(x_axis_name)
    if y_axis_name != None:
        plt.ylabel(y_axis_name)
    if x_tick_labels != None:
        plt.xticks(x_values, labels=x_tick_labels, rotation=-75)
    if y_tick_labels != None:
        plt.yticks(y_values, labels=y_tick_labels, rotation=-75)
    plt.show()

def plot_pie_chart(values: list, labels: list, title: str = None):
    """Plots pie chart using matplotlib.

    Args:
        values (list): list of values to display in the pie chart
        labels (list): list of labels to display along the outside of the pie chart
        title (str): OPTIONAL title to display on the pie chart
    """
    plt.figure()
    plt.pie(values, labels=labels, autopct='%1.1f%%')
    if title != None:
        plt.title(title)
    plt.legend(labels=labels, bbox_to_anchor=(0.85,1.025), loc="upper left")
    plt.show()

def plot_equal_width_frequency_diagram(cutoffs: list, frequencies: list, title: str = None,
    x_axis_name: str = None, y_axis_name:str = None, x_tick_labels: list = None, 
    y_tick_labels: list = None):
    """Plots a frequency diagram using equal width cutoffs and matplotlib.

    Args:
        cutoffs (list): list of cutoff values to define the "bins"
        frequencies (list): list of values that represent the number of things that fall in each 
        "bin"
        title (str): OPTIONAL title to display on the graph 
        x_axis_name (str): OPTIONAL name for the x-axis
        y_axis_name (str): OPTIONAL name for the y-axis
        x_tick_labels (list of str): OPTIONAL parallel list to x-axis values for the names at each 
        value
        y_tick_labels (list of str): OPTIONAL parallel list to y-axis values for the names at each 
        value
    """
    plt.figure()
    plt.bar(cutoffs[:-1], frequencies, width=(cutoffs[1]-cutoffs[0]), edgecolor='black')
    if title != None:
        plt.title(title)
    if x_axis_name != None:
        plt.xlabel(x_axis_name)
    if y_axis_name != None:
        plt.ylabel(y_axis_name)
    if x_tick_labels != None:
        plt.xticks(cutoffs[:-1], labels=x_tick_labels, rotation=-45)
    if y_tick_labels != None:
        plt.xticks(cutoffs[:-1], labels=y_tick_labels)
    plt.show()

def plot_histogram(data: list, title: str = None, x_axis_name: str = None, 
    y_axis_name: str = None):
    """Plots a histogram using matplotlib.

    Args:
        data (list): data to be plotted on the histogram
        title (str): OPTIONAL title to display on the graph 
        x_axis_name (str): OPTIONAL name for the x-axis
        y_axis_name (str): OPTIONAL name for the y-axis
    """
    plt.figure()
    plt.hist(data, bins=10)
    if title != None:
        plt.title(title)
    if x_axis_name != None:
        plt.xlabel(x_axis_name)
    if y_axis_name != None:
        plt.ylabel(y_axis_name)
    plt.show()

def plot_scatter_chart(x_values: list, y_values: list, title: str = None, 
    x_axis_name: str = None, y_axis_name: str = None):
    """Plots a scatter chart using matplotlib. Calculates and shows a fit line created using 
    linear regression..

    Args:
        x_values (list):
        y_values (list):
        title (str): OPTIONAL title to display on the graph 
        x_axis_name (str): OPTIONAL name for the x-axis
        y_axis_name (str): OPTIONAL name for the y-axis
    """
    plt.figure()
    plt.scatter(x_values, y_values, marker='o', s=100)
    if title != None:
        plt.title(title)
    if x_axis_name != None:
        plt.xlabel(x_axis_name)
    if y_axis_name != None:
        plt.ylabel(y_axis_name)
    slope, intercept = utils.compute_slope_intercept(x=x_values, y=y_values)
    plt.plot([min(x_values), max(x_values)], 
        [slope * min(x_values) + intercept, slope * max(x_values) + intercept], c="r", lw=3)
    plt.tight_layout()
    plt.show()

def plot_box_chart(distributions: list, labels:list, title: str = None, x_axis_name: str = None,
    y_axis_name: str = None):
    """Plots a box chart using matplotlib.

    Args:
        distributions (list):
        labels (list):
        title (str): OPTIONAL title to display on the graph
        x_axis_name (str): OPTIONAL name for the x-axis
        y_axis_name (str): OPTIONAL name for the y-axis
    """
    plt.figure()
    plt.boxplot(distributions)
    if title != None:
        plt.title(title)
    if x_axis_name != None:
        plt.xlabel(x_axis_name)
    if y_axis_name != None:
        plt.ylabel(y_axis_name)
    plt.xticks(list(range(1, len(distributions) + 1)), labels, rotation=-75)
    plt.annotate("$\mu=100$", xy=(1.5, 105), xycoords="data", horizontalalignment="center")
    plt.annotate("$\mu=100$", xy=(0.5, 0.5), xycoords="axes fraction", 
                 horizontalalignment="center", color="blue")
    plt.show()

import numpy as np

from mysklearn.mypytable import MyPyTable

def get_column_list(table: list, header: list, col_name: str):
    """Gets the values in the desired column from a given 2D list of values.

    Args:
        table (list): 2D list of values
        header (list): header for the table
        col_name (str): nameof the desired column

    Returns:
        list: values in the desired column
    """
    column = []
    column_index = header.index(col_name)
    for row in table:
        column.append(row[column_index])
    return column

def get_frequencies(table:MyPyTable, col_name: str):
    """Gets the frequencies for each value of a numeric attribute. Gets
    sorted list of all values in the given column of the given table.
    Goes through sorted list and for each occurence, adds one to the 
    corresponding index of the occurrences list.

    Args:
        col_name (str): name of the column we are searching

    Return:
        dict: keys are the values in the column and values are the number 
        of times that they occur
    """
    col_frequencies = dict()
    col_data = table.get_column(col_identifier=col_name)
    for value in col_data:
        if value in col_frequencies.keys():
            col_frequencies[value] += 1
        else:
            col_frequencies[value] = 1
    return list(col_frequencies.keys()), list(col_frequencies.values())

def group_by(table: list, header: list, col_name: str):
    """Groups rows in the given table together using the given column name.
    Gets the index and column data of the desired index. Gets the "names" 
    of the groups using the values in the column (converts them to an ordered
    list of unique values). Goes through rows in the given table, grouping them 
    by their value in the desired column.
    
    Args:
        table (MyPyTable): table that we are searching
        col_name (str): name of the column we are searching
        
    Returns:
        list of list of list: ordered list of unique values in the desired column
        list of list: list of subtables group by the value in the desired
        column
    """
    group_by_col_index = header.index(col_name)
    group_by_col = get_column_list(table=table, header=header, col_name=col_name)
    group_by_split_col = []
    for value in group_by_col:
        split_value = value.split(",")
        for item in split_value:
            group_by_split_col.append(item)
    group_names = sorted(list(set(group_by_split_col)))
    group_subtables = [[] for _ in group_names]
    for row in table:
        group_by_val = row[group_by_col_index].split(",")
        for val in group_by_val:
            group_by_val_subtable_index = group_names.index(val)
            group_subtables[group_by_val_subtable_index].append(row.copy())
    return group_names, group_subtables

def compute_equal_width_cutoffs(values: list, number_of_bins: int):
    """Compute the cutoffs of "equal width" binning discretization. Gets the 
    width of the bins using the range of the given values and the given number 
    of bins. 

    Args:
        values (list): values to create bins from
        number_of_bins (int): the number of desired equal-width bins

    Returns:
        list of float: cutoff values for equal width bins
    """
    values_range = max(values) - min(values)
    bin_width = values_range / number_of_bins
    # use np.arrage b/c getting a range of floats
    cutoffs = list(np.arange(min(values), max(values), bin_width))
    cutoffs.append(max(values))
    cutoffs = [round(cutoff, 2) for cutoff in cutoffs]
    return cutoffs

def compute_bin_frequencies(values: list, cutoffs: list):
    """Compute the frequencies for each value in their respective bin as a part of 
    "equal width" binning discretization. Iterates through values and increases the 
    frequency of the corresponding cutoff from the given cutoffs table as necessary.

    Args:
        values (list): values used to count frequencies
        cutoffs (list of float): cutoff values of bins

    Returns:
        list of int: number of occurences that are in each "bin"
    """
    frequencies = [0 for _ in range(len(cutoffs)-1)] # because N + 1 cutoffs
    for value in values:
        if value == max(values): # fully closed last bin
            frequencies[-1] += 1
        else:
            for index in range(len(cutoffs)-1): # remaining half-open bins
                if cutoffs[index] <= value and value < cutoffs[index + 1]:
                    frequencies[index] += 1
    return frequencies

def compute_slope_intercept(x: list, y: list):
    """Computes the slope and y-intercept for a group of given values. Mostly used 
    to create a trendline.

    Args:
        x (list of float): represents x values of points
        y (list of float): represents y values of points

    Returns:
        float: slope of trendline of points
        float: y-intercept of trendline of points
    """
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    numerator = sum([(x[index] - mean_x) * (y[index] - mean_y) for index in range(len(x))])
    denominator = sum([(x[index] - mean_x) ** 2 for index in range(len(x))])
    m = numerator / denominator
    b = mean_y - m * mean_x
    return m, b

def compute_covariance(x_values: list, y_values: list):
    """Computes the covariance for the parallel lists given.

    Args:
        x_values (list): list of x values
        y_values (list): list of y values

    Returns:
        float: covariance between the given lists of values
    """
    x_mean = np.mean(x_values)
    y_mean = np.mean(y_values)
    numerator = 0
    for index in range(len(x_values)):
        numerator += (x_values[index] - x_mean) * (y_values[index] - y_mean)
    denominator = len(x_values)
    return numerator / denominator

def compute_correlation_coefficient(x_values: list, y_values: list):
    """Computes the correlation coefficient for the parallel lists given.

    Args:
        x_values (list): list of x values
        y_values (list): list of y values

    Returns:
        float: correlation coefficient between the given lists of values
    """
    x_mean = np.mean(x_values)
    y_mean = np.mean(y_values)
    numerator = 0
    for index in range(len(x_values)):
        numerator += (x_values[index] - x_mean) * (y_values[index] - y_mean)
    denominator_x = 0
    for index in range(len(x_values)):
        denominator_x += (x_values[index] - x_mean) ** 2
    denominator_y= 0
    for index in range(len(y_values)):
        denominator_y += (y_values[index] - y_mean) ** 2
    denominator = np.sqrt(denominator_x * denominator_y)
    return numerator / denominator


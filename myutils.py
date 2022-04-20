'''
Weston Averill
CPSC322
5/4/22
Final Project
'''

import numpy as np

def get_column(table, header, col_name):
    '''
    return a list with all values based on given col name
    '''
    col_index = header.index(col_name)
    col = []
    for row in table:
        value = row[col_index]
        if value != "NA":
            col.append(str(value))
    return col

def get_frequencies(table, header, col_name):
    '''
    return a tuple of lists to find frequences of col name
    '''
    col = get_column(table, header, col_name)
    col.sort() # inplace
    # parallel lists
    values = []
    counts = []
    for value in col:
        if value in values: # seen it before
            counts[-1] += 1 # okay because sorted
        else: # haven't seen it before
            values.append(value)
            counts.append(1)

    return values, counts # we can return multiple values in python
    # they are packaged into a tuple

def compute_equal_width_cutoffs(values, num_bins):
    '''
    find the cut offs for the number of bins
    '''
    values_range = max(values) - min(values)
    bin_width = values_range / num_bins # float
    # since bin_width is a float, we shouldn't use range() to generate a list
    # of cutoffs, use np.arange()
    cutoffs = list(np.arange(min(values), max(values), bin_width))
    cutoffs.append(max(values)) # exactly the max(values)
    # to handle round off error...
    # if your application allows, we should convert to int
    # or optionally round them
    cutoffs = [round(cutoff, 2) for cutoff in cutoffs]
    return cutoffs

def compute_bin_frequencies(values, cutoffs):
    '''
    compute the number of bin based on cutoffs
    '''
    freqs = [0 for _ in range(len(cutoffs) - 1)] # because N + 1 cutoffs

    for value in values:
        if value == max(values):
            freqs[-1] += 1 # add one to the last bin count
        else:
            for i in range(len(cutoffs) - 1):
                if cutoffs[i] <= value < cutoffs[i + 1]:
                    freqs[i] += 1
                    # add one to this bin defined by [cutoffs[i], cutoffs[i+1])
    return freqs

def compute_slope_intercept(x_param, y_param):
    '''
    compute and return the slope intercept
    '''
    meanx = np.mean(x_param)
    meany = np.mean(y_param)

    num = sum([(x_param[i] - meanx) * (y_param[i] - meany) for i in range(len(x_param))])
    den = sum([(x_param[i] - meanx) ** 2 for i in range(len(x_param))])
    m_val = num / den
    # y = mx + b => b = y - mx
    b_val = meany - m_val * meanx
    return m_val, b_val

def group_by(table, header, groupby_col_name):
    '''
    return groups
    '''
    groupby_col_index = header.index(groupby_col_name) # use this later
    groupby_col = get_column(table, header, groupby_col_name)
    group_names = sorted(list(set(groupby_col))) # e.g. [75, 76, 77]
    # print(group_names)
    group_subtables = [[] for _ in group_names] # e.g. [[], [], []]
    # print(group_subtables)

    for row in table:
        groupby_val = row[groupby_col_index] # e.g. this row's modelyear
        # which subtable does this row belong?
        # print("ksjdfbskjdbsd" + str(groupby_val))
        groupby_val_subtable_index = group_names.index(str(groupby_val))
        group_subtables[groupby_val_subtable_index].append(row.copy()) # make a copy

    return group_names, group_subtables

def discretize_yes_no(y):
    '''discretizer'''
    new_y = []
    for value in y:
        if value >= 100:
            new_y.append("high")
        else:
            new_y.append("low")
    return new_y

def compute_euclidean_distance(v1, v2):
    '''compute euclidean distance'''
    temp = []
    for i in range(len(v1)):
        if type(v1[i]) == str and type(v1[i]) == str:
            if v1[i] == v2[i]:
                temp.append(0)
            else:
                temp.append(1)
        else:
            temp.append((v1[i]-v2[i])**2)
    # return np.sqrt(sum([(v1[i] - v2[i]) ** 2 for i in range(len(v1))]))
    return np.sqrt(sum(temp))
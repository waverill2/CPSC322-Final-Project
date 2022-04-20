'''
Weston Averill
CPSC322
5/4/22
Final Project
'''
import copy
from builtins import str

#from tabulate import tabulate # uncomment if you want to use the pretty_print() method
# install tabulate with: pip install tabulate

# required functions/methods are noted with TODOs
# provided unit tests are in test_mypytable.py
# do not modify this class name, required function/method headers, or the unit tests
class MyPyTable:
    """Represents a 2D table of data with column names.
    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.
        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    # def pretty_print(self):
    #     """Prints the table in a nicely formatted grid structure.
    #     """
    #     print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).
        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        return len(self.data), len(self.column_names)

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.
        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.
        Returns:
            list of obj: 1D list of values in the column
        """
        column_index = col_identifier
        # if type(col_identifier) == type(""):
        if isinstance(col_identifier, str):
            try:
                column_index = self.column_names.index(col_identifier)
            except ValueError:
                print("Column name is not in the header")
        new_list = []
        for row in self.data:
            if include_missing_values is False and row[column_index] == "":
                continue
            else:
                new_list.append(row[column_index])

        return new_list

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).
        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        for row in self.data:
            # for i in range(len(row)):
            for i, value in enumerate(row):
                try:
                    # numeric_value = float(row[i])
                    row[i] = float(value)
                except ValueError:
                    # print(row[i], " could not be converted to numeric")
                    continue

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.
        Args:
            row_indexes_to_drop(list of int): list of row indexes to remove from the table data.
        """
        new_list = []
        # for i in range(len(self.data)):
        for i, value in enumerate(self.data):
            if i not in row_indexes_to_drop:
                new_list.append(value)
        self.data = new_list

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.
        Args:
            filename(str): relative path for the CSV file to open and load the contents of.
        Returns:
            MyPyTable: return self so the caller can write code like
                table = MyPyTable().load_from_file(fname)
        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        # open the file
        infile = open(filename, "r", encoding="utf8")
        lines = infile.readlines()
        # iterate over each line
        for line in lines:
            line = line.strip()
            values = line.split(",")
            self.data.append(values)

        self.column_names = self.data[0]
        self.data.pop(0)
        self.convert_to_numeric()

        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.
        Args:
            filename(str): relative path for the CSV file to save the contents to.
        Notes:
            Use the csv module.
        """
        self.data.insert(0, self.column_names)
        # open a file to write to
        outfile = open(filename, "w", encoding="utf8")
        # iterate over each row we will use to write with
        for row in self.data:
            for j in range(len(row) - 1):
                outfile.write(str(row[j]) + ",")
            outfile.write(str(row[-1]) + "\n")
        outfile.close()
        self.data.pop(0)

    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.
        Args:
            key_column_names(list of str): column names to use as row keys.
        Returns
            list of int: list of indexes of duplicate rows found
        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
                The first instance of a row is not considered a duplicate.
        """
        list_of_new_cols = []
        indexes = []
        index = 0
        # iterate over each row in our data table
        for row in self.data:
            # create a temporary list
            temp = []
            # iterate over the columns given to use
            for j in key_column_names:
                temp.append(row[self.column_names.index(j)])
            if temp in list_of_new_cols:
                indexes.append(index)
            else:
                list_of_new_cols.append(temp)
            index += 1

        # return the list of indexes
        return indexes

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        new_list = []
        has_na = False
        # iterate over row in our data table
        for row in self.data:
            # iterate over values in the row
            for j in row:
                # check to see if the value is missing
                if j == "N/A":
                    has_na = True
                    break
            if not has_na:
                new_list.append(row)
            else:
                has_na = False
        self.data = new_list

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
            by the column's original average.
        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        # first block of code to calculate the average of a row
        sum_of_cols = 0.0
        number_of_non_na = 0
        for row in self.data:
            if not row[self.column_names.index(col_name)] == "NA":
                sum_of_cols += float(row[self.column_names.index(col_name)])
                number_of_non_na += 1
        # get the average
        # now assign missing values with the average
        sum_of_cols /= number_of_non_na
        for row in self.data:
            if row[self.column_names.index(col_name)] == "NA":
                row[self.column_names.index(col_name)] = sum_of_cols

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.
        Args:
            col_names(list of str): names of the continuous columns to compute summary stats for.
        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]
        """
        new_list = []

        for j in col_names:
            temp = []
            # this will get all the values of a column in a sigle list
            for row in self.data:
                temp.append(row[self.column_names.index(j)])
            try:
                # find all the values for the summary
                minv = min(temp)
                maxv = max(temp)
                mid = (maxv+minv) / 2
                avg = sum(temp) / len(temp)
                temp.sort()
                median = 0
                if len(temp) % 2 != 0:
                    median = temp[len(temp) // 2]
                else:
                    median = (temp[len(temp) // 2] + temp[(len(temp) // 2) - 1]) / 2
                # add values to a list and then add it to the table
                stats_list = [j, minv, maxv, mid, avg, median]
                new_list.append(stats_list)
            except ValueError:
                print("exception occured")
        self.data = new_list
        return self

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
            with other_table based on key_column_names.
        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.
        Returns:
            MyPyTable: the inner joined table.
        """

        # create the new table for the inner join table
        new_table = []
        # iterate over the left table
        for self_row in self.data:
            # iterate over the right table
            for other_row in other_table.data:
                # create a temporary new row
                new_row = []
                # iterate over the key column names
                for j in key_column_names:
                    # the next blocks of code determine if the keys match
                    self_index = self.column_names.index(j)
                    other_index = other_table.column_names.index(j)
                    if self_row[self_index] == other_row[other_index]:
                        new_row.append(self_row[self_index])
                        if len(new_row) == len(key_column_names):
                            temp_self_row = copy.deepcopy(self_row)
                            temp_other_row = copy.deepcopy(other_row)
                            temp_list = []
                            for col in other_table.column_names:
                                if col not in key_column_names:
                                    other_ind = other_table.column_names.index(col)
                                    temp_list.append(temp_other_row[other_ind])
                            # finish creating the combined row and then add it to the new table
                            new_row = temp_self_row + temp_list
                            new_table.append(new_row)

        # now we need to make the header
        new_header = []
        self_columns = self.column_names
        other_columns = other_table.column_names
        # combine the columns of the two tables
        for j in key_column_names:
            new_header.append(j)
            self_columns.remove(j)
            other_columns.remove(j)
        new_header = new_header + self_columns + other_columns

        self.column_names = new_header
        self.data = new_table
        # print(self.data)
        return self

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with
            other_table based on key_column_names.
        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.
        Returns:
            MyPyTable: the fully outer joined table.
        Notes:
            Pad the attributes with missing values with "NA".
        """

        # create the combined header first to help with the outer
        new_header = copy.deepcopy(self.column_names)
        for col in other_table.column_names:
            if col not in self.column_names:
                new_header.append(col)

        # first complete an inner join
        new_table = []
        # iterate over all rows in the left table
        for self_row in self.data:
            # iterate over the rigth table
            for other_row in other_table.data:
                new_row = []
                # iterate over the key column names
                for j in key_column_names:
                    # the next blck checks to see if the keys match
                    self_index = self.column_names.index(j)
                    other_index = other_table.column_names.index(j)
                    if self_row[self_index] == other_row[other_index]:
                        new_row.append(self_row[self.column_names.index(j)])
                        if len(new_row) == len(key_column_names):
                            temp_self_row = copy.deepcopy(self_row)
                            temp_other_row = copy.deepcopy(other_row)
                            temp_list = []
                            for col in other_table.column_names:
                                if col not in key_column_names:
                                    other_ind = other_table.column_names.index(col)
                                    temp_list.append(temp_other_row[other_ind])
                            # create the new combined row and add it to the new table
                            new_row = temp_self_row + temp_list
                            new_table.append(new_row)

        # loop over the left table again to find the rows that were not added in the inner join
        for self_row in self.data:
            if not self.self_is_in_table(self_row, other_table, key_column_names):
                new_row = []
                for j in new_header:
                    if j in self.column_names:
                        new_row.append(self_row[self.column_names.index(j)])
                    else:
                        new_row.append("NA")
                new_table.append(new_row)

        # loop over the right table to find the rows that were not added in the inner join
        for other_row in other_table.data:
            if not self.other_is_in_table(other_row, other_table, key_column_names):
                new_row = []
                for j in new_header:
                    if j in other_table.column_names:
                        new_row.append(other_row[other_table.column_names.index(j)])
                    else:
                        new_row.append("NA")
                new_table.append(new_row)

        self.column_names = new_header
        self.data = new_table
        return self

    # helper function to determine if a row was added during the inner join
    def self_is_in_table(self, row, other_table, key_column_names):
        '''helper for outer join'''
        for other_row in other_table.data:
            temp_row = []
            for j in key_column_names:
                if row[self.column_names.index(j)] == other_row[other_table.column_names.index(j)]:
                    temp_row.append(row[self.column_names.index(j)])
                    if len(temp_row) == len(key_column_names):
                        return True
        return False

    # helper function to determine if a row was added during the inner join
    def other_is_in_table(self, row, other_table, key_column_names):
        '''helper for outer join'''
        for self_row in self.data:
            temp_row = []
            for j in key_column_names:
                if row[other_table.column_names.index(j)] == self_row[self.column_names.index(j)]:
                    temp_row.append(row[other_table.column_names.index(j)])
                    if len(temp_row) == len(key_column_names):
                        return True
"""myutils.py
@author Joshua Seward

Reusable general-purpose functions for PA7
"""
import attr
import numpy as np
from math import log2

def basic_discretizer1(value) -> str:
    """Function that discretizes values according to the requirements for PA4.

    Args:
        value (numeric val): numeric value to be discretized

    Returns:
        str: discretized label for the given value
    """
    return "high" if value >= 100 else "low"

def basic_discretizer2(value) -> str:
    """Function that discretizes values as positive or not positive.

    Args:
        value (numeric val): numeric value to be discretized

    Returns:
        str: discretized label for given value
    """
    return "positive" if value > 0 else "not positive"

def euclidean_distance(v1: list, v2: list):
    """Computes the euclidean distance between two parallel vectors.

    Args:
        v1 (list): list representation of a vector
        v2 (list): list representation of a vector

    Return:
        float: euclidean distance between the two vectors
    """
    sum = 0
    for index in range(len(v1)):
        sum += (v2[index] - v1[index]) ** 2
    return np.sqrt(sum)

def doe_mpg_discretizer(mpg_value):
    """Function that discretizes the given mpg value according to the DOE 
    mpg ratings.

    Args:
        mpg_value (numeric val): mpg value to be discretzed

    Returns:
        int: the DOE mpg rating associated with the given mpg value
    """
    if mpg_value >= 45:
        return 10
    elif mpg_value < 45 and mpg_value >= 37:
        return 9
    elif mpg_value < 37 and mpg_value >= 31:
        return 8
    elif mpg_value < 31 and mpg_value >= 27:
        return 7
    elif mpg_value < 27 and mpg_value >= 24:
        return 6
    elif mpg_value < 24 and mpg_value >= 20:
        return 5
    elif mpg_value < 20 and mpg_value >= 17:
        return 4
    elif mpg_value < 17 and mpg_value >= 15:
        return 3
    elif mpg_value < 15 and mpg_value >= 14:
        return 2
    elif mpg_value <= 13:
        return 1

def randomize_in_place(a_list: list, parallel_list:list=None, random_state:int=None):
    """Shuffles the given list. If a parallel list is given, both lists are shuffled in parallel.

    Args:
        a_list (list of obj): list to be shuffled
        parallel_list (list of obj): OPTIONAL list to be shuffled in parallel with a_list
        random_state (int): integer used for seeding a random number generator for reproducible results
    """
    if random_state is not None:
        np.random.seed(random_state)
    for index in range(0, len(a_list)):
        random_index = np.random.randint(0, len(a_list))
        a_list[index], a_list[random_index] = a_list[random_index], a_list[index]
        if parallel_list != None:
            parallel_list[index], parallel_list[random_index] = parallel_list[random_index], parallel_list[index]

def group_by_result(instance_indices:list, results: list):
    """Groups the different instance indices by their result. Helper function for stratified kfold
    cross validation. Gets unique values from the given results and creates group of given indices
    that are associated with the respective result.

    Args:
        instance_indices (list of int): list of indices for rows in the table
        results (list of obj): list of the results of each instance (parallel to instance_indices)

    Returns:
        list of list of int: instance indices grouped by their result
    """
    groups = []
    result_values = sorted(list(set(results)))
    for result_value in result_values:
        group = []
        for index in range(len(instance_indices)): 
            if results[index] == result_value:
                group.append(instance_indices[index])
        groups.append(group)
    return groups

def tdidt(current_instances:list, header:list, attribute_domains:dict, available_attributes:list,\
    previous_num_instances: int = None) -> list:
    """Basic approach for Top Down Induction of Decision Trees (uses recursion!!):

    select an attribute to split on
    group data by attribute domains (creates pairwise disjoint partitions)
    for each partition, repeat unless one of the following occurs (base case)
       CASE 1: all class labels of the partition are the same => make a leaf node
       CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
       CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote 
               leaf node

    Args:
        current_instances (list of list of obj): instances to partition
        header (list of str): list of the names of the attributes
        attribute_domains (dict of int:list of str): dict of the possible values for each attribute index (indexed from 
        the header)
        previous_num_instances (int): the number of instances in the previous iteration (None if this is the first iteration)

    Returns:
        list of list of ... of list of obj: multi-tiered list representing the decision tree
    """
    # select an attribute to split on
    split_attribute = select_attribute(current_instances=current_instances, header=header, attribute_domains=attribute_domains,\
        available_attributes=available_attributes)
    available_attributes.remove(split_attribute)
    tree = ["Attribute", split_attribute] # current "Attribute" node
    # group data by attribute domains
    partitions = partition_instances(instances=current_instances, header=header, attribute_domains=attribute_domains,\
        split_attribute=split_attribute)
    for attribute_value, attribute_partition in partitions.items():
        # print("partition:", attribute_value, attribute_partition)
        value_subtree = ["Value", attribute_value]
        # CASE 1: all class labels of the partition are the same => make a leaf node
        if len(attribute_partition) > 0 and all_same_class(attribute_partition):
            # print("CASE 1")
            # make a "Leaf" node
            attribute_class = attribute_partition[0][len(attribute_partition[0])-1]
            leaf_node = ["Leaf", attribute_class, len(attribute_partition), len(current_instances)]
            value_subtree.append(leaf_node)
            tree.append(value_subtree)
        # CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
        elif len(attribute_partition) > 0 and len(available_attributes) == 0:
            # print("CASE 2")
            # handle clash w/ majority vote "Leaf" node
            attribute_classes = dict()
            for instance in attribute_partition:
                attribute_class = instance[len(instance)-1]
                if attribute_classes.get(attribute_class) is None:
                    attribute_classes[attribute_class] = 1
                else:
                    attribute_classes[attribute_class] += 1
            attribute_classes = dict(sorted(attribute_classes.items(), key=lambda item: item[0]))
            # get the max value
            max_class_value = max(list(attribute_classes.values()))
            max_class = None
            # find the key associated with the max value
            for key, value in attribute_classes.items():
                if max_class_value == value:
                    max_class = key
                    break # exit the loop since we have found the key
            leaf_node = ["Leaf", max_class, len(attribute_partition), len(current_instances)]
            value_subtree.append(leaf_node)
            tree.append(value_subtree)
        # CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote 
        #         leaf node
        elif len(attribute_partition) == 0:
            # print("CASE 3")
            # "backtrack" to replace the attribute node with a majority leaf node by replacing the current "Attribute" node w/
            # a majority vote "Leaf" node
            instance_classes = dict()
            for instance in current_instances:
                instance_class = instance[len(instance)-1]
                if instance_classes.get(instance_class) is None:
                    instance_classes[instance_class] = 1
                else:
                    instance_classes[instance_class] += 1
            instance_classes = dict(sorted(instance_classes.items(), key=lambda item: item[0]))
            # get the max value
            max_class_value = max(list(instance_classes.values()))
            max_class = None
            # find the key associated with the max value
            for key, value in instance_classes.items():
                if max_class_value == value:
                    max_class = key
                    break # exit the loop since we have found the key
            tree = ["Leaf", max_class, len(current_instances), previous_num_instances]
        else: # the previous conditions were all false so recurse
            # print("RECURSE")
            subtree = tdidt(current_instances=attribute_partition, header=header, attribute_domains=attribute_domains,
                available_attributes=available_attributes.copy(), previous_num_instances=len(current_instances))
            # NOTE - we use .copy() b/c we change the list earlier in the tdidt function
            # append subtree to value_subtree and to the tree appropriately
            value_subtree.append(subtree)
            tree.append(value_subtree)
    return tree

def select_attribute(current_instances: list, header:list, attribute_domains: list, available_attributes: list) -> str:
    """Function to select an attribute for tdidt. Uses entropy to select an attribute.

    Args:
        current_instances (list of list of obj): remaining instances to be put in the tree
        header (list of str): list of the names of the attributes
        attribute_domains (dict of int:list of str): dict of the possible values for each attribute index (indexed from 
        the header)
        available_attributes (list of str): remaining attributes that have not yet been selected

    Returns:
        str: selected attribute
    """
    # compute E_start
    E_start = compute_entropy(current_instances)
    # compute E_new for each attribute
    num_instances = len(current_instances)
    attribute_E_new = dict()
    for attribute in available_attributes:
        attribute_E_new[attribute] = 0
        # partition the available attribute's instances by value
        attribute_partitions = partition_instances(instances=current_instances, header=header,\
            attribute_domains=attribute_domains, split_attribute=attribute)
        for _, partition in attribute_partitions.items():
            # get the number of instances with the desired attribute value
            attribute_value_count = len(partition)
            # calculate E_value using the instances in the partition
            E_value = compute_entropy(partition)
            attribute_E_new[attribute] += (attribute_value_count/num_instances) * E_value
    # compute Gain for each attribute
    attribute_gain = dict()
    for attribute in available_attributes:
        attribute_gain[attribute] = E_start - attribute_E_new[attribute]
    # select the attribute with the largest Gain (Estart - Enew)
    max_gain_value = max(list(attribute_gain.values()))
    selected_attribute = None
    for attribute_name, gain_value in attribute_gain.items():
        if gain_value == max_gain_value:
            selected_attribute = attribute_name
            break
    return selected_attribute

def compute_entropy(instances: list) -> float:
    """Computes the entropy value from a given set of instances.

    Args:
        instances (list of list of str): 2d list of values in a table

    Returns:
        float: entropy value from the given instances
    """
    value_counts = dict()
    for instance in instances:
        instance_class = instance[len(instance)-1]
        if value_counts.get(instance_class) is None:
            value_counts[instance_class] = 1
        else:
            value_counts[instance_class] += 1
    value_priors = dict(sorted(value_counts.items(), key=lambda item: item[0]))
    num_instances = len(instances)
    for prior in list(value_priors.keys()):
        value_priors[prior] = value_priors[prior] / num_instances
    entropy = 0
    for prior in list(value_priors.values()):
        entropy += prior * log2(prior)
    entropy = entropy * -1
    return entropy

def partition_instances(instances:list , header:list , attribute_domains:dict, split_attribute) -> dict:
    """Groups the instances by the attribute's domain values.

    Args:
        instances (list of list of obj): instances to partition
        header (list of str): list of the names of the attributes
        attribute_domains (dict of int:list of str): dict of the possible values for each attribute index (indexed from the header)
        split_attribute (obj): attribute to use to partition the instances

    Return:
        dict of obj:(list of obj): partitioned instances based on split_attribute
    """
    partitions = dict() # key (string) : value (subtable)
    attribute_index = header.index(split_attribute)
    attribute_domain = attribute_domains[split_attribute]
    for attribute_value in attribute_domain:
        partitions[attribute_value] = list()
        for instance in instances:
            if instance[attribute_index] == attribute_value:
                partitions[attribute_value].append(instance)
    return partitions

def all_same_class(partition: list) -> bool:
    """Checks if every instance in the given partition is of the same class.

    Args:
        partition (list of list): partition to check if all instances are of the same class

    Returns:
        bool: True if all instances in the partition are of the same class
    """
    desired_class_index = len(partition[0])-1
    # get the class to compare the rest of the partition too (last index of an instance)
    desired_class = partition[0][desired_class_index]
    for instance in partition:
        if instance[desired_class_index] != desired_class:
            return False
    return True

def predict_instance(instance: list, header: list, tree:list) -> str:
    """Recursively traverses the given decision tree and to get a prediction for the given instance.

    Args:
        instance (list of str): instance to make a prediction for
        header (list of str): names of the attributes in the instance
        tree (list of list of ... list of obj): multi-tiered list representing a decision tree

    Returns:
        str: predicted class for the given instance
    """
    if tree[0] == "Attribute":
        # get the value of the attribute for the instance
        attribute_index = header.index(tree[1])
        instance_attribute_value = instance[attribute_index]
        for value_subtree in tree[2:]:
            if value_subtree[1] == instance_attribute_value:
                return predict_instance(instance=instance, header=header, tree=value_subtree[2])
    elif tree[0] == "Leaf":
        return tree[1]

def generate_decision_rules(tree:list, tree_header:list, attribute_names: list, class_name: str) -> list:
    """Traverses all nodes of a tree and generates decision rules.

    Args:
        tree (list of list of ... list of obj): multi-tiered list representing a decision tree
        tree_header (list of str): attribute names as used in the tree (corresponds with attribute names)
        attribute_names(list of str or None): A list of attribute names to use in the decision rules
        class_name(str): A string to use for the class name in the decision rules

    Returns:
        list of str: list of decision rules
    """
    decision_rules = list()
    # NOTE - maybe do recursion and start writing the rule once you reach the leaf node, prepending the rule 
    # until you reach the root???
    if tree[0] == "Attribute":
        for subtree in tree[2:]:
            root_rule = f"IF {attribute_names[tree_header.index(tree[1])]}"
            rule = root_rule + generate_decision_rule(subtree, tree_header, attribute_names, class_name)
            decision_rules.append(rule)
    return decision_rules

def generate_decision_rule(tree:list, tree_header:list, attribute_names: list, class_name:str) -> str:
    """An attempt at a recursive function to generate decision tree rules.
    """
    if tree[0] == "Leaf":
        return f" THEN {class_name} = {tree[1]}"
    elif tree[0] == "Attribute":
        for subtree in tree[2:]:
            return f" AND {attribute_names[tree_header.index(tree[1])]}" + generate_decision_rule(tree=subtree,\
                tree_header=tree_header, attribute_names=attribute_names, class_name=class_name)
    elif tree[0] == "Value":
        subtree = tree[2]
        return f" == {tree[1]}" + generate_decision_rule(tree=subtree, tree_header=tree_header,\
            attribute_names=attribute_names, class_name=class_name)


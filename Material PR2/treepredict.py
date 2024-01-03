#!/usr/bin/env python3
import random
import sys
import collections
from math import log2
from typing import List, Tuple

# Used for typing
Data = List[List]


def read(file_name: str, separator: str = ",") -> Tuple[List[str], Data]:
    """
    t3: Load the data into a bidimensional list.
    Return the headers as a list, and the data
    """
    header = None
    data = []
    with open(file_name, "r") as fh:
        for line in fh:
            values = line.strip().split(separator)
            if header is None:
                header = values
                continue
            data.append([_parse_value(v) for v in values])
    return header, data


def _parse_value(v: str):
    try:
        if float(v) == int(v):
            return int(v)
        else:
            return float(v)
    except ValueError:
        return v
    # try:
    #     return float(v)
    # except ValueError:
    #     try:
    #         return int(v)
    #     except ValueError:
    #         return v


def unique_counts(part: Data):
    """
    t4: Create counts of possible results
    (the last column of each row is the
    result)
    """
    return dict(collections.Counter(row[-1] for row in part))

    # results = collections.Counter()
    # for row in part:
    #     c = row[-1]
    #     results[c] += 1
    # return dict(results)

    # results = {}
    # for row in part:
    #     c = row[-1]
    #     if c not in results:
    #         results[c] = 0
    #     results[c] += 1
    # return results


def gini_impurity(part: Data):
    """
    t5: Computes the Gini index of a node
    """
    total = len(part)
    if total == 0:
        return 0

    results = unique_counts(part)
    imp = 1
    for v in results.values():
        imp -= (v / total) ** 2
    return imp


def entropy(part: Data):
    """
    t6: Entropy is the sum of p(x)log(p(x))
    across all the different possible results
    """
    total = len(part)
    results = unique_counts(part)

    probs = (v / total for v in results.values())
    return -sum(p * log2(p) for p in probs)

    # imp = 0
    # for v in results.values():
    #     p = v / total
    #     imp -= p * log2(p)
    # return imp


def _split_numeric(prototype: List, column: int, value):
    return prototype[column] >= value


def _split_categorical(prototype: List, column: int, value: str):
    return prototype[column] == value
    #raise NotImplementedError


def divideset(part: Data, column: int, value) -> Tuple[Data, Data]:
    """
    t7: Divide a set on a specific column. Can handle
    numeric or categorical values
    """
    set1 = []
    set2 = []

    if isinstance(value, (int, float)):
        split_function = _split_numeric
    else:
        split_function = _split_categorical

    for row in part:
        set1.append(row) if split_function(row, column, value) else set2.append(row)

    return (set1, set2)


class DecisionNode:
    def __init__(self, col=-1, value=None, results=None, tb=None, fb=None):
        """
        t8: We have 5 member variables:
        - col is the column index which represents the
          attribute we use to split the node
        - value corresponds to the answer that satisfies
          the question
        - tb and fb are internal nodes representing the
          positive and negative answers, respectively
        - results is a dictionary that stores the result
          for this branch. Is None except for the leaves
        """
        self.col = col
        self.value = value
        self.results = results
        self.tb = tb
        self.fb = fb
        #raise NotImplementedError


def buildtree(part: Data, scoref=entropy, beta=0):
    """
    t9: Define a new function buildtree. This is a recursive function
    that builds a decision tree using any of the impurity measures we
    have seen. The stop criterion is max_s\Delta i(s,t) < \beta
    """
    if len(part) == 0:
        return DecisionNode()

    current_score = scoref(part)

    if current_score == 0:
        results = unique_counts(part)

        if len(results) == 1:
            return DecisionNode(results=results)

        else:
            return DecisionNode(results=random_voting(results))

    best_gain, best_criteria, best_sets = search_best_decision(part, scoref)

    if best_gain < beta:
        results = unique_counts(part)

        if len(results) == 1:
            return DecisionNode(results=results)

        else:
            return DecisionNode(results=random_voting(results))


    return DecisionNode(col=best_criteria[0],
                        value=best_criteria[1],
                        tb=buildtree(best_sets[0], scoref, beta),
                        fb=buildtree(best_sets[1], scoref, beta))

def search_best_decision(part: Data, scoref=entropy):
    """
    Search for the best decision (best gain, criteria and sets) to make, based on the best gain possible
    """
    best_gain = 0
    best_criteria = None
    best_sets = None

    # Count the number of columns (minus the label)
    column_count = len(part[0]) - 1

    for col in range(column_count):
        # Generate the list of different values in this column
        column_values = {}
        for row in part:
            column_values[row[col]] = 1

        # Now try dividing the rows up for each value in this column
        for value in column_values.keys():
            (trueSet, falseSet) = divideset(part, col, value)

            # Information gain
            p = len(trueSet) / len(part)
            gain = scoref(part) - p * scoref(trueSet) - (1 - p) * scoref(falseSet)
            if gain > best_gain and len(trueSet) > 0 and len(falseSet) > 0:
                best_gain = gain
                best_criteria = (col, value)
                best_sets = (trueSet, falseSet)

    return best_gain, best_criteria, best_sets

def random_voting(results):
    labels = [label for label in results.keys()]
    return random.choice(labels)

def iterative_buildtree(part: Data, scoref=entropy, beta=0):
    """
    t10: Define the iterative version of the function buildtree
    """
    stack = []
    root = DecisionNode()
    stack.append(root, part)

    while len(stack) > 0:
        node, data = stack.pop()

        if len(data) == 0:
            continue

        current_score = scoref(data)

        if current_score == 0:
            results = unique_counts(data)

            if len(results) == 1:
                node.results = results

            else:
                node.results = random_voting(results)

            continue

        best_gain, best_criteria, best_sets = search_best_decision(data, scoref)

        if best_gain < beta:
            results = unique_counts(data)
            if len(results) == 1:
                node.results = results

            else:
                node.results = random_voting(results)

            continue

        node.col = best_criteria[0]
        node.value = best_criteria[1]
        node.tb = DecisionNode()
        node.fb = DecisionNode()
        stack.append(node.tb, best_sets[0])
        stack.append(node.fb, best_sets[1])

    return root


def classify(tree, values):
    if tree.results is not None:
        return tree.results

    value = values[tree.col]

    if isinstance(value, (int, float)):
        branch = tree.tb if value >= tree.value else tree.fb
    else:
        branch = tree.tb if value == tree.value else tree.fb

    return classify(branch, values)


def branchPruning(tree: DecisionNode, threshold: float, impurity=entropy):
    if tree.tb.results is None:
        branchPruning(tree.tb, threshold)
    if tree.fb.results is None:
        branchPruning(tree.fb, threshold)

    if tree.tb.results is not None and tree.fb.results is not None:
        tb, fb = [], []
        for v, c in tree.tb.results.items():
            tb += [[v]] * c
        for v,c in tree.fb.results.items():
            fb += [[v]] * c

        p = len(tb) / len(tb + fb)
        delta = impurity(tb + fb) - p * impurity(tb) - (1 - p) * impurity(fb)

        if delta < threshold:
            tree.tb, tree.fb = None, None
            tree.results = unique_counts(tb + fb)

def print_tree(tree, headers=None, indent=""):
    """
    t11: Include the following function
    """
    # Is this a leaf node?
    if tree.results is not None:
        print(tree.results)
    else:
        # Print the criteria
        criteria = tree.col
        if headers:
            criteria = headers[criteria]
        print(f"{indent}{criteria}: {tree.value}?")

        # Print the branches
        print(f"{indent}T->")
        print_tree(tree.tb, headers, indent + "  ")
        print(f"{indent}F->")
        print_tree(tree.fb, headers, indent + "  ")


def print_data(headers, data):
    colsize = 15
    print('-' * ((colsize + 1) * len(headers) + 1))
    print("|", end="")
    for header in headers:
        print(header.center(colsize), end="|")
    print("")
    print('-' * ((colsize + 1) * len(headers) + 1))
    for row in data:
        print("|", end="")
        for value in row:
            if isinstance(value, (int, float)):
                print(str(value).rjust(colsize), end="|")
            else:
                print(value.ljust(colsize), end="|")
        print("")
    print('-' * ((colsize + 1) * len(headers) + 1))


def main():
    try:
        filename = sys.argv[1]
    except IndexError:
        filename = "decision_tree_example.txt"

    # header, data = read(filename)
    # print_data(header, data)

    # print(unique_counts(data))

    # print(gini_impurity(data))
    # print(gini_impurity([]))
    # print(gini_impurity([data[0]]))

    # print(entropy(data))
    # print(entropy([]))
    # print(entropy([data[0]]))

    headers, data = read(filename)
    tree = buildtree(data)
    print_tree(tree, headers)


if __name__ == "__main__":
    main()
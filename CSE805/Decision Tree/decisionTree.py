#DECISION TREE
#Graphical representation of all the possible solutions to a decision based on some conditions
from pyparsing import conditionAsParseAction

#ROOT NODE: represents the entire population or sample that gets divided into two or more homogenous sets
#LEAF NODE: represents the final decision or prediction. No further splits occur at these nodes
#SPLITTING: dividing the root node/sub-node into different parts on the basis of some condition
#Metrics for Splitting
#Gini Impurity: Measures the likelihood of an incorrect classification of a new instance if it was randomly classified according to the distribution of classes in the dataset
#Information Gain: the decrease in entropy after a dataset is split on the basis of an attribute
#InfoGain = Entropy(s) - [(weighted average) X Entropy(each feature)]
#Entropy: measures the randomness of a class =p(yes)log2(p(yes) - p(no)log2p(no)

#BRANCH/SUBTREE: represent the outcome of a decision or test, leading to another node
#PRUNNING: removing unwanted branches from the tree(removing the nodes that provide little power in classifying instances)
#pre-pruning(early stopping): Stops the tree from growing once it meets certain criteria i.e maximum depth, minimum number of samples per leaf)
#post-pruning: removes branches from a fully grown tree that do not provide significant power

#Parent/Child node
#root node is the parent node and all other nodes branched from it are known as child node

training_data = [['Green', 3, 'Mango'], ['Yellow', 3, 'MAngo'],
                 ['Red', 1, 'Grape'], ['Red', 1, 'Grape'], ['Yellow', 3, 'Lemon']]

#column labels(only used to print the table)
header = ["color", "diameter", "label"]

#Finding unique values
def unique_vals(rows, col):
    """Find the unique values for a column in a dataset."""
    return set([row[col] for row in rows])

#unique_vals(training_data, 0)
#unique_vals(training_data, 1)

#Counting Class Labels
def class_counts(rows):
    """Counts the number of each type of example in the dataset"""
    counts = {}  # a dictionary of label -> count.
    for row in rows:
        #in our dataset format, the label is always the last column
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

#Checking If a Value Is Numeric
def is_numeric(value):
    """test if a value is numeric"""
    return isinstance(value, int) or isinstance(value, float)

#i.e is the color green?, is the diameter >= 3?
class Question:
    """A question is used to partition a dataset

    This class just records a 'column_number' (e.g, 0 for color) and a 'column_value' (e.g, Green). The 'match' method
    is used to compare the feature value in an example to the feature value stores in the question.
    """

    def __init__(self, column, value):
        self.column = column   #the index of the feature i.e 0 for color, 1 for diameter
        self.value = value     #is the value to compare i.e green, 3

    def match(self, example):
        #compare feature value in an example to the feature value in this question
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    #This is just a helper method to print the question in readable format
    def __repr__(self):
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (header[self.column], condition, str(self.value))

    def partition(rows, question):
        """Partitions of a dataset.
        For each row in the dataset, check if it matches the question.
        If so, add it to 'true rows', otherwise, add it to 'false rows'.
        """
        true_rows, false_rows = [], []
        for row in rows:
            if question.match(row):
                true_rows.append(row)
            else:
                false_rows.append(row)

        return true_rows, false_rows

    def gini(rows):
        counts = class_counts(rows)
        impurity = 1
        for lbl in counts:
            prob_of_lbl = counts[lbl] / float(len(rows))
            impurity -= prob_of_lbl ** 2
        return impurity

    def info_gain(left, right, current_uncertainty):
        """The uncertainty of the starting node minus the weighted impurity of two child nodes"""
        p = float(len(left)) / len(left) + len(right)
        return current_uncertainty - p * gini(left) - (1 - p) * gini(right)

    def find_best_fit(rows):
        best_gain = 0 #keep track of the best information gain
        best_question = None #keep train of the feature / value that produced it
        current_uncertainty = gini(rows)
        n_features = len(rows[0]) - 1  #number of columns

        for col in range(n_features):  #for each feature
            values = set([row[col] for row in rows])

            for col in range(n_features):  # For each feature
                values = set([row[col] for row in rows])

                for val in values:  # For each value
                    question = Question(col, val)

                    # Try splitting the dataset
                    true_rows, false_rows = partition(rows, question)

                    # Skip this split if it results in empty subsets
                    if len(true_rows) == 0 or len(false_rows) == 0:
                        continue

                    # Calculate the information gain from this split
                    gain = info_gain(true_rows, false_rows, current_uncertainty)

                    # If this is the best gain we've seen so far, keep track of it
                    if gain > best_gain:
                        best_gain = gain
                        best_question = question

            return best_gain, best_question

    def build_tree(rows, depth=0, max_depth=None):
        """Recursively build a decision tree."""
        gain, question = find_best_split(rows)

        #If no further split is beneficial, or if we've reached the maximum depth, return a leaf node
        if gain == 0 or (max_depth and depth >= max_depth):
            return class_counts(rows)

        # Otherwise, partition the data and build the subtrees
        true_rows, false_rows = partition(rows, question)

        # Recursively build the true and false branches of the tree
        true_branch = build_tree(true_rows, depth + 1, max_depth)
        false_branch = build_tree(false_rows, depth + 1, max_depth)

        return {
            'question': question,
            'true_branch': true_branch,
            'false_branch': false_branch
        }


# Example usage:
tree = build_tree(training_data, max_depth=3)
print(tree)

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pydot
from math import ceil, sqrt, log2
import sys
from collections import deque
from sklearn import preprocessing


#np.set_printoptions(threshold=sys.maxsize)

not_observed = 0

def load_data(data_path, labels_column_name, id_column_name=None, separator=','):
    '''
    Loads data from csv file to a pandas.Dataframe
    data is split into features (everything except labels) and labels

    Features are filtered for 'id' column if there exists one in data

    Input:
        -data_path (string): path do .csv file containing data
    
    Returns:
        -features (pandas.Dataframe): features
        -labels (pandas.Dataframe): labels
    '''
    data = pd.read_csv(data_path, sep=separator)
    
    features = data.loc[:, data.columns != labels_column_name]
    
    labels = (data.loc[:, labels_column_name])

    if not (id_column_name is None):
        features = features.loc[:, features.columns != id_column_name]

        
    return features, labels

def random_test_train_split(features, labels, test_percentage=20):
    '''
    Splits features and labels into training and testing dataset based on test_percentage

    Input:
        -features (pandas.Dataframe): features
        -labels (pandas.Dataframe): labels
        -test_percentage: percentage of data that will go into test dataset, everything else will be used for trainng
    
    Returns:
        train_features (pandas.Dataframe)
        train_labels (pandas.Dataframe)
        test_features (pandas.Dataframe)
        test_labels (pandas.Dataframe)
    '''
    p = np.random.permutation(len(features))
    features = features.reindex(p)
    labels = labels.reindex(p)

    if test_percentage == 0:
        return features, labels, np.array([]), np.array([])

    test_data_size = int(test_percentage / 100 * len(features))

    test_features = features.iloc[:test_data_size]

    test_labels = labels.iloc[:test_data_size]

    train_features = features.iloc[test_data_size:]
    train_labels = labels.iloc[test_data_size:]

    return train_features, train_labels, test_features, test_labels

class Node(object):

    index = 0

    def __init__(self, x, y, features, depth):
        self.x = x
        self.y = y
        self.features = features
        self.is_terminal = False
        self.my_class = None
        self.my_split_attribute = None
        self.children = {}
        self.children_quantities = {}
        self.index = Node.index
        Node.index += 1
        self.depth = depth


    def all_x_have_same_label(self):
        '''
        Tests if all remaining examples in dataset have the same label
        '''
        labels = set(self.y)
        return len(labels) == 1

    def no_features_left(self):
        '''
        Tests if there are no more features that can be used for split
        '''
        return len(self.features) == 0
    
    def set_class_to_most_common_label(self):
        '''
        Used for impure leaf nodes. Sets their class to be the most frequent class in dataset
        '''
        self.is_terminal = True

        values, counts = np.unique(self.y, return_counts=True)
        ind = np.argmax(counts)
        
        for value, count in zip(values, counts):
            self.children_quantities[value] = count

        self.my_class = values[ind]
    
    def get_classification_error(self):
        '''
        Returns the number of missclassified examples in this node
        '''
        values, counts = np.unique(self.y, return_counts=True)
        ind = np.argmax(counts)

        num_of_elems_in_most_common_class = counts[ind]
        num_of_elems_in_other_classes = self.y.shape[0] - num_of_elems_in_most_common_class
        
        return num_of_elems_in_other_classes
    
    def get_number_of_leafs_and_classification_error_in_them(self):
        '''
        Searches for the number of leafs that the node has
        Calculates the sum of all missclassified examples in the leafs
        '''

        deq = deque()
        deq.append(self)

        sum_of_missclassified_examples = 0
        number_of_leafs = 0

        while len(deq):
            node = deq.popleft()

            if node.is_terminal:
                number_of_leafs += 1
                sum_of_missclassified_examples += node.get_classification_error()
            else:
                for child in node.children.values():
                    deq.append(child)
            
        return sum_of_missclassified_examples, number_of_leafs


    def find_best_feature_for_split_entropy(self):
        '''
        Finds best attribute for splitting based on entropy
        Sets self.my_split_attribute to chosen attribute for splitting
        '''
        feature_entropies = {}

        for feature in self.features:
            feature_values = set(self.x[:, feature])

            total_num_of_examples = self.x.shape[0]
            total_feature_entropy = 0

            for feature_value in feature_values:
                feature_value_entropy = 0

                classes = self.y[self.x[:, feature] == feature_value]
                total_examples_with_feature_value = len(classes)
                classes_values = set(classes)

                for class_value in classes_values:
                    num_exampes_with_feature_value_and_class_value = sum(classes == class_value)
                    class_probability = num_exampes_with_feature_value_and_class_value / total_examples_with_feature_value
                    my_entropy = -class_probability * log2(class_probability)
                    feature_value_entropy += my_entropy

                
                feature_value_ratio = total_examples_with_feature_value / total_num_of_examples
                total_feature_entropy += feature_value_ratio * feature_value_entropy

            feature_entropies[feature] = total_feature_entropy

        feature_with_min_entropy = min(feature_entropies, key=feature_entropies.get)

        self.my_split_attribute = feature_with_min_entropy

    
    def find_best_feature_for_split_gini(self):
        '''
        Finds best attribute for splitting based on Gini impurity
        Sets self.my_split_attribute to chosen attribute for splitting
        '''
        feature_gini_impurities = {}

        for feature in self.features:
            feature_values = set(self.x[:, feature])
            total_num_of_examples = self.x.shape[0]
            total_feature_gini_impurity = 0

            for feature_value in feature_values:
                feature_value_gini_impurity = 0

                classes = self.y[self.x[:, feature] == feature_value]
                total_examples_with_feature_value = len(classes)
                classes_values = set(classes)

                for class_value in classes_values:
                    num_exampes_with_feature_value_and_class_value = sum(classes == class_value)
                    class_probability = num_exampes_with_feature_value_and_class_value / total_examples_with_feature_value
                    my_gini_impurity = class_probability * (1-class_probability)
                    feature_value_gini_impurity += my_gini_impurity

                feature_value_ratio = total_examples_with_feature_value / total_num_of_examples
                total_feature_gini_impurity += feature_value_ratio * feature_value_gini_impurity

            feature_gini_impurities[feature] = total_feature_gini_impurity

        feature_with_min_gini_impurity = min(feature_gini_impurities, key=feature_gini_impurities.get)

        self.my_split_attribute = feature_with_min_gini_impurity


    def find_best_feature_for_classification_error(self):
        '''
        Finds best attribute for splitting based on classification error
        Sets self.my_split_attribute to chosen attribute for splitting
        '''
        feature_classification_errors = {}

        for feature in self.features:
            feature_values = set(self.x[:, feature])
            total_num_of_examples = self.x.shape[0]
            total_feature_classification_error = 0

            for feature_value in feature_values:
                feature_value_classification_error = 0

                classes = self.y[self.x[:, feature] == feature_value]
                total_examples_with_feature_value = len(classes)
                classes_values = set(classes)
                
                probabilities = []
                for class_value in classes_values:
                    num_exampes_with_feature_value_and_class_value = sum(classes == class_value)
                    class_probability = num_exampes_with_feature_value_and_class_value / total_examples_with_feature_value
                    probabilities.append(class_probability)
                
                feature_value_classification_error = 1 - max(probabilities)
                feature_value_ratio = total_examples_with_feature_value / total_num_of_examples
                total_feature_classification_error += feature_value_ratio * feature_value_classification_error

            feature_classification_errors[feature] = total_feature_classification_error

        feature_with_min_classification_error = min(feature_classification_errors, key=feature_classification_errors.get)

        self.my_split_attribute = feature_with_min_classification_error
    
    def find_best_feature_for_split_chi(self):
        '''
        Finds best attribute for splitting based on chi-squared test
        Sets self.my_split_attribute to chosen attribute for splitting
        '''
        feature_chi_values = {}

        for feature in self.features:
            feature_values = set(self.x[:, feature])
            #print(f'examining feature {feature}')
            total_num_of_examples = self.x.shape[0]
            class_values = set(self.y)
            

            total_chi_value = 0
            num_of_examples_with_feature_value = {}
            num_of_examples_with_label_value = {}

            for feature_value in feature_values:
                num_of_examples_with_feature_value[feature_value] = len(self.x[self.x[:,feature_value] == feature_value])
            
            for class_value in class_values:
                num_of_examples_with_label_value[class_value] = len(self.y == class_value)
            

            feature_chi_value = 0

            for feature_value in feature_values:
                for class_value in class_values:
                    indexes = np.logical_and(self.x[:,feature_value] == feature_value, self.y == class_value)
                    num_of_examples_with_feature_value_and_label_value = len(self.x[indexes])
                    
                    expected_value = num_of_examples_with_feature_value[feature_value] * num_of_examples_with_label_value[class_value] / total_num_of_examples
                    observed_value = num_of_examples_with_feature_value_and_label_value

                    feature_chi_value += ((observed_value - expected_value)**2 + 1) / (expected_value + 1)
            
            feature_chi_values[feature] = feature_chi_value
        
        feature_with_min_chi_value = min(feature_chi_values, key=feature_chi_values.get)

        self.my_split_attribute = feature_with_min_chi_value
    
    def find_best_feature_for_split(self, metric):
        if metric == 'entropy':
            self.find_best_feature_for_split_entropy()

        elif metric == 'gini':
            self.find_best_feature_for_split_gini()

        elif metric == 'chi':
            self.find_best_feature_for_split_chi()
        
        elif metric == 'classification_error':
            self.find_best_feature_for_classification_error()

        else:
            raise Exception("Valid values for metric are 'entropy', 'gini', 'chi', and 'classification_error'")

    def split_by_feature(self):
        '''
        Splits the node into children based on values of self.my_split_attribute
        This method can only be called after find_best_feature_for_split which finds best split attribute

        Each child represents one of the attributes values
        Each child inherits only the examples with that attribute value
        Each child inherits examples with all attributes except the attribute used for splitting (because we cannot use the same attribute for splitting more than once)
        Each child knows its depth

        Parent stores reference to child in a dictionary where key is feature value
        Parent stores how many examples are given to each child
        '''
        feature_values = set(self.x[:, self.my_split_attribute])

        child_features = list(self.features)
        child_features.remove(self.my_split_attribute)
        number_of_examples = len(self.x)

        for feature_value in feature_values:
            indexes = np.where(self.x[:, self.my_split_attribute] == feature_value)
            child_x = self.x[indexes]
            child_y = self.y[indexes]
            
            child = Node(child_x, child_y, child_features, depth=self.depth+1)
            self.children[feature_value] = child
        
        values, counts = np.unique(self.y, return_counts=True)
        for value, count in zip(values, counts):
            self.children_quantities[value] = count
        
        return self.children

    def get_child(self, x):
        '''
        Used for decision-nodes only (cannot be used for leaf nodes)
        Returns child with the feature attribute value equal to x.feature_attribute_value
        If there isn't such child then returns random child where children with more examples
            in them have higher probability in being chosen
        '''

        if self.is_terminal:
            raise Exception("Leaf nodes have no children")
        
        value = x[self.my_split_attribute]
        if value in self.children:
            child = self.children[value]
            return child
        else:
            children_keys = list(self.children.keys())
            total_examples = len(self.x)
            children_probabilities = [len(child.x) / total_examples for child in self.children.values()]
            random_child_key = np.random.choice(children_keys, p=children_probabilities)
            random_child = self.children[random_child_key]
            return random_child

class DecisionTree(object):

    def __init__(self, impurity_function='gini', discretization_method='dynamic', max_depth=-1, min_leaf_size=-1, number_of_discr_classes=3, pruning_method=None):
        self.tree_root = None 
        self.impurity_function = impurity_function
        self.column_index_to_feature_name_mapping = {}
        self.number_to_label_value_mapping = {}
        self.discretization_method = discretization_method
        self.number_of_discr_classes = number_of_discr_classes
        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size
        self.numerical_to_categorical_values_mappings = {}
        self.feature_values = {}
        self.pruning_method = pruning_method

    
    def display_tree(self, path):
        '''
        Saves a .png image of a tree to a path from argument path
        '''

        colors = ['green', 'red', 'blue', 'purple', 'brown']
        classes = sorted(list((set(self.tree_root.y))))
        class_to_color_map = {}

        for i, class_name in enumerate(classes):
            class_to_color_map[class_name] = colors[i % len(colors)]

        
        if self.tree_root is None:
            raise Exception("Tree must be trained first in order to be displayed")

        node = self.tree_root
        if node.is_terminal:
            label = str(node.my_class)
            color = class_to_color_map[node.my_class]
            #color = 'green'
        else:
            label = str(self.column_index_to_feature_name_mapping[node.my_split_attribute])
            color = 'gold'

        # label += '\n' + str(node.x.shape[0])
        class_label = ''
        for i, class_name in enumerate(classes):
            class_num = np.sum(node.y == class_name)
            class_label += f'{class_name} : {class_num}'
            if i != len(classes) - 1:
                class_label += ', '
        label += f'\n {class_label}'
        G = pydot.Dot(graph_type='digraph')
        G.add_node(pydot.Node(name='0', label=label, style='filled', color=color, shape='box'))

        if self.tree_root.is_terminal:
            G.write_png(path)
            return

        stack = [(self.tree_root, 0)]
        
        index = 0
        while(len(stack)): 

            node, parent_index = stack.pop()

            for child_key in node.children:
                child = node.children[child_key]
                index += 1
                if child.is_terminal:
                    label = str(child.my_class)
                    color = class_to_color_map[child.my_class]
                    #color = 'green'
                else:
                    stack.append((child, index))
                    label = str(self.column_index_to_feature_name_mapping[child.my_split_attribute])
                    color = 'gold'
                
                #print(label)
                #label += '\n' + str(child.x.shape[0])
                class_label = ''
                for i, class_name in enumerate(classes):
                    class_num = np.sum(child.y == class_name)
                    class_label += f'{class_name} : {class_num}'
                    if i != len(classes) - 1:
                        class_label += ', '
                label += f'\n {class_label}'
                G.add_node(pydot.Node(name=str(index), label=label, style='filled', color=color, shape='box'))
                edge_label = child_key
                G.add_edge(pydot.Edge(str(parent_index), str(index), label=edge_label))
                
        
        G.write_png(path)


    def change_column_names_to_numbers(self, features):
        '''
        Changes column names from strings to numbers:
            For example: ['height', 'weight', 'bmi'] --> [0, 1, 2]
        Pairs of column name and number are stored in dict so it can be reversed
            For example: column_index_to_feature_name_mapping['height'] = 0
        '''
        
        column_index_to_feature_name_mapping = {}
        for column_index, column_name in enumerate(features.columns):
            column_index_to_feature_name_mapping[column_index] = column_name

        self.column_index_to_feature_name_mapping = column_index_to_feature_name_mapping
    
        
    def split_to_intervals_with_same_length(self, features_df, column_name, number_of_discr_classes):
        '''
        Discretization of attribute in column column_name to number_of_discr_classes values
        This discretization is done so each resulting interval has the same size
        For example values of height between 150cm and 210cm would be split into 3 intervals as:
            ['shorter than 170cm', 'between 170cm and 190cm', 'taller than 190cm']

        All feature values are stored in dictionary for each column:
        For example:
            feature_values['height'] = ['shorter than 170cm', 'between 170cm and 190cm', 'taller than 190cm']
        
        All borders of feature values are stored in dict for each column
        For example:
            numerical_to_categorical_values_mappings['height'] = [170, 190] (as values 170 and 190 are used as borders of intervals) 
        '''
        feature_column = features_df[column_name]
        min_value = feature_column.min()
        max_value = feature_column.max()

        # print(f'min = {min_value} max = {type(max_value)} classmin = {min_value} classmax = {type(max_value)}')

        split_vectors = []
        split_size = (max_value - min_value) / number_of_discr_classes

        left_split_border = min_value
        for i in range(number_of_discr_classes-1):
            right_split_border = left_split_border + split_size

            if i == 0:
                split_vector = feature_column <= right_split_border
            else:
                split_vector = np.logical_and(left_split_border <= feature_column, feature_column <= right_split_border)
            
            split_vectors.append((split_vector, left_split_border, right_split_border))

            if i == number_of_discr_classes-2:
                split_vector = right_split_border < feature_column
                split_vectors.append((split_vector, left_split_border, right_split_border))

            left_split_border = right_split_border

        splitting_borders = []
        values = []

        for split_index, (split_vector, left_split_border, right_split_border) in enumerate(split_vectors):
            if split_index == 0:
                features_df.loc[split_vector, column_name] = f'x <= {right_split_border}'
                values.append(f'x <= {right_split_border}')
            elif split_index == len(split_vectors) - 1:
                features_df.loc[split_vector, column_name] = f'x > {right_split_border}'
                values.append(f'x > {right_split_border}')
            else:
                features_df.loc[split_vector, column_name] = f'{left_split_border} < x <= {right_split_border}'
                values.append(f'{left_split_border} < x <= {right_split_border}')
            
            splitting_borders.append(right_split_border)

        self.feature_values[column_name] = values
        self.numerical_to_categorical_values_mappings[column_name] = splitting_borders
        
    
                
    def split_to_intervals_with_same_size(self, features_df, column_name, number_of_discr_classes):
        '''
        Discretization of attribute in column column_name to number_of_discr_classes intervals
        This discretization is done so each resulting interval will have the same number of examples

        All feature values are stored in dictionary for each column:
        For example:
            feature_values['height'] = ['shorter than 170cm', 'between 170cm and 190cm', 'taller than 190cm']
        
        All borders of feature values are stored in dict for each column
        For example:
            numerical_to_categorical_values_mappings['height'] = [170, 190] (as values 170 and 190 are used as borders of intervals) 
        '''
        features_column = features_df[column_name]

        sorted_array = np.sort(features_column)
        m = features_column.shape[0]

        min_value = sorted_array[0]
        max_value = sorted_array[m-1]

        split_vectors = []

        number_of_examples_in_split = ceil(m / number_of_discr_classes)
        left_split_border_value = min_value

        for i in range(number_of_discr_classes-1):
            right_split_border_index = min(number_of_examples_in_split * (i+1), m-1)
            right_split_border_value = sorted_array[right_split_border_index]

            if i == 0:
                split_vector = features_column <= right_split_border_value
            else:
                split_vector = np.logical_and(left_split_border_value < features_column, features_column <= right_split_border_value)

            #split_vector = np.logical_and(left_split_border_value < features_df, features_df <= right_split_border_value)
            split_vectors.append((split_vector, left_split_border_value, right_split_border_value))

            if i == number_of_discr_classes-2:
                split_vector = right_split_border_value < features_column
                split_vectors.append((split_vector, left_split_border_value, right_split_border_value))

            left_split_border_value = right_split_border_value

        #a[a == max_value] = number_of_discr_classes-1
        splitting_borders = []
        values = []
        for split_index, (split_vector, left_split_border, right_split_border) in enumerate(split_vectors):
            if split_index == 0:
                features_df.loc[split_vector, column_name] = f'x <= {right_split_border}'
                values.append(f'x <= {right_split_border}')
            elif split_index == len(split_vectors) - 1:
                features_df.loc[split_vector, column_name] = f'x > {right_split_border}'
                values.append(f'x > {right_split_border}')
            else:
                features_df.loc[split_vector, column_name] = f'{left_split_border} < x <= {right_split_border}'
                values.append(f'{left_split_border} < x <= {right_split_border}')

            splitting_borders.append(right_split_border)
        
        self.feature_values[column_name] = values
        self.numerical_to_categorical_values_mappings[column_name] = splitting_borders
        
    def dynamic_split(self, features_df, column_name, labels):
        '''
        Discretization of attribute in column column_name to
        This discretization is done as explained in paper Decision trees: an overview and their use in medicine (SI-2000)

        All feature values are stored in dictionary for each column:
        For example:
            feature_values['height'] = ['shorter than 170cm', 'between 170cm and 190cm', 'taller than 190cm']
        
        All borders of feature values are stored in dict for each column
        For example:
            numerical_to_categorical_values_mappings['height'] = [170, 190] (as values 170 and 190 are used as borders of intervals) 
        '''
        features_np = np.array(features_df[column_name]).flatten()

        m = features_np.shape[0]
        labels_np = np.array(labels).flatten()
        sorted_indexes = features_np.argsort()
        sorted_features = features_np[sorted_indexes]
        sorted_labels = labels_np[sorted_indexes]

        groups = []

        last_label = sorted_labels[0]
        last_feature_value = sorted_features[0]
        current_feature_value = sorted_features[0]
        group = [1, current_feature_value, current_feature_value, last_label]
        for i in range(1, m):
            current_label = sorted_labels[i]
            current_feature_value = sorted_features[i]

            if current_label == last_label:
                group[0] += 1
                group[2] = current_feature_value
            else:
                groups.append(group)
                group = [1, last_feature_value, current_feature_value, current_label]
                last_label = current_label 
            
            last_feature_value = current_feature_value

        groups.append(group)

        last_index = len(groups) - 1

        if len(groups) < 2:
            return
        if len(groups) == 2:
            border = groups[0][2]

            group_indexes_less_than_border = border >= features_np
            group_indexes_more_than_border = np.logical_not(group_indexes_less_than_border)

            features_df.loc[group_indexes_less_than_border, column_name] = f'x <= {border}'
            features_df.loc[group_indexes_more_than_border, column_name] = f'x > {border}'
            return

        first_group = groups[0]
        second_group = groups[1]
        third_group = groups[2]
        merged_groups = []

        i = 2
        while(True):
            if second_group[0] == 1 and first_group[3] == third_group[3] and first_group[0] > 1 and third_group[0] > 1:
                members = first_group[0] + second_group[0] + third_group[0]
                start = first_group[1]
                end = third_group[2]
                label = first_group[3]
                merged_group = [members, start, end, label]
                #merged_groups.append(merged_group)

                if i == last_index:
                    merged_groups.append(merged_group)
                    break
                if i == last_index - 1:
                    merged_groups.append(merged_group)
                    last_group = groups[last_index]
                    merged_groups.append(last_group)
                    break
                first_group = merged_group
                second_group = groups[i+1]
                third_group = groups[i+2]
                i += 2

            else:
                merged_groups.append(first_group)
                if i == last_index:
                    merged_groups.append(second_group)
                    merged_groups.append(third_group)
                    break
                else:
                    first_group = second_group
                    second_group = third_group
                    third_group = groups[i + 1]
                    i += 1
        
        splitting_borders = []
        values = []

        for i, merged_group in enumerate(merged_groups):
            start = merged_group[1]
            end = merged_group[2]
            
            if i == 0:
                group_indexes = features_np <= end
                features_df.loc[group_indexes, column_name] = f'x <= {end}'
                values.append(f'x <= {end}')
            elif i == len(merged_groups) - 1:
                group_indexes = features_np > start
                features_df.loc[group_indexes, column_name] = f'x > {start}'
                values.append(f'x > {start}')
            else:
                group_indexes = np.logical_and(start < features_np, features_np <= end)
                features_df.loc[group_indexes, column_name] = f'{start} < x <= {end}'
                values.append(f'{start} < x <= {end}')

            splitting_borders.append(end)

        self.feature_values[column_name] = values
        #print(values)
        #print('\n\n')
        self.numerical_to_categorical_values_mappings[column_name] = splitting_borders

    def change_continuous_to_categorical_values(self, features, labels, number_of_discr_classes):
        '''
        Changes all columns with continuous values to categorical values
        Column is consiered continuous if it has more than 5 distinct values
        For example:
            height = [150, 155, 160, 175, 180, 185, 190, 195]
        '''
        for column_index, column_name in enumerate(features.columns):

            values = set(features[column_name])

            # column is continous if it has more than 5 distinct values
            if len(values) > 5:
                if self.discretization_method == 'same_length':
                    self.split_to_intervals_with_same_length(features, column_name, number_of_discr_classes)
                if self.discretization_method == 'same_size':
                    self.split_to_intervals_with_same_size(features, column_name, number_of_discr_classes)
                if self.discretization_method == 'dynamic':
                    self.dynamic_split(features, column_name, labels)

    def change_new_data_continuous_to_categorical_values(self, features):
        '''
        This method is used to change the values of features in testing dataset 
        from continuous to categorical in the same way that has been done in training dataset
        '''
        for column_name in self.numerical_to_categorical_values_mappings:
            border_values = self.numerical_to_categorical_values_mappings[column_name]

            patches = []

            left_border = border_values[0]
            for border_index, border_value in enumerate(border_values):
                right_border = border_value

                if border_index == 0:
                    group_indexes = features[column_name] <= right_border
                    patches.append((group_indexes, f'x <= {right_border}'))
                elif border_index == len(border_values) - 1:
                    group_indexes = features[column_name] > right_border
                    if self.discretization_method == 'dynamic':
                        group_indexes = features[column_name] >= right_border
                    patches.append((group_indexes, f'x > {right_border}'))
                else:
                    group_indexes = np.logical_and(left_border < features[column_name], features[column_name] <= right_border)
                    patches.append((group_indexes, f'{left_border} < x <= {right_border}'))

                left_border = right_border

            for patch in patches:
                group_indexes, value = patch
                features.loc[group_indexes, column_name] = value
    
    def prune_node_by_index(self, index):
        '''
        Prunes node with a given index from the tree
        '''

        deq = deque()
        deq.append(self.tree_root)

        while len(deq):
            node = deq.popleft()
            if node.index == index:
                #print(f"pruned {index}")
                node.set_class_to_most_common_label()
                return
            for child in node.children.values():
                deq.append(child)
        
    

    def reset_prunning(self):
        '''
        This function resets prunning
        If a node is declared as terminal but has children it means it was pruned
        This function puts its state back to not being terminal
        '''
    
        deq = deque()
        deq.append(self.tree_root)

        while len(deq):
            node = deq.popleft()
            if node.is_terminal and len(node.children):
                node.is_terminal = False
            for child in node.children.values():
                deq.append(child)
    
    def print_all_nodes_in_tree(self):
        deq = deque()
        deq.append(self.tree_root)
        num = 0

        while len(deq):
            num += 1
            node = deq.popleft()
            #print(node.index, end=" ")

            if not node.is_terminal:
                for child in node.children.values():
                    deq.append(child)
        
        print(f'num = {num}')


    def cost_complexity_prunning(self):
        
        prunning_indexes_list = []


        while(not self.tree_root.is_terminal):
            #self.print_all_nodes_in_tree()
            #input()
            deq = deque()
            deq.append(self.tree_root)

            min_alpha = float('inf')
            min_alpha_index = -1

            while len(deq):

                node = deq.popleft()

                if node.is_terminal:
                    continue
                
                my_error_rate = node.get_classification_error()
                leafs_error_rate, num_of_leafs = node.get_number_of_leafs_and_classification_error_in_them()

                alfa = (my_error_rate - leafs_error_rate) / num_of_leafs

                if alfa < min_alpha:
                    min_alpha = alfa
                    min_alpha_index = node.index
                
                for child in node.children.values():
                    deq.append(child)

            self.prune_node_by_index(min_alpha_index)
            prunning_indexes_list.append(min_alpha_index)
        
        print("finished pruning list")
        # Reset pruning so the tree is whole again
        self.reset_prunning()
        self.print_all_nodes_in_tree()
        min_pruning_error = float('inf')
        min_pruning_index = -1

        for i, index_for_pruning in enumerate(prunning_indexes_list):
            self.prune_node_by_index(index_for_pruning)
            self.print_all_nodes_in_tree()
            
            predictions = self.get_predictions(self.pruning_features)
            pruning_error = self.get_error(predictions, self.pruning_labels)
            print(f'current min error {min_pruning_error} and now is {pruning_error}')
            if pruning_error <= min_pruning_error:
                min_pruning_error = pruning_error
                min_pruning_index = i
            #input()
        
        self.reset_prunning()
        
        index = 0
        while(index <= min_pruning_index):
            index_for_pruning = prunning_indexes_list[index]
            self.prune_node_by_index(index_for_pruning)
            index += 1


    def pessimistic_error_pruning(self):

        deq = deque()

        node = self.tree_root
        deq.append(node)

        while(len(deq)):

            node = deq.popleft()

            children_quantities = node.children_quantities.values()
            total_quantity = sum(children_quantities)
            node_error = total_quantity - max(children_quantities)
            node_error += 1/2

            deq2 = deque()
            deq2.append(node)
            error_sum = 0
            while(len(deq2)):

                sub_node = deq2.popleft()
                for sub_child in sub_node.children.values():
                    if sub_child.is_terminal:
                        sub_children_quantities = sub_child.children_quantities.values()
                        sub_node_error = sum(sub_children_quantities) - max(sub_children_quantities)
                        sub_node_error += 1/2

                        error_sum += sub_node_error

                    else:
                        deq2.append(sub_child)
            
            standard_error = sqrt(node_error * (total_quantity - node_error) / total_quantity)

            if node_error < error_sum + standard_error:
                node.set_class_to_most_common_label()
            
            else:
                for child in node.children.values():
                    if not child.is_terminal:
                        deq.append(child)
    
    def reduced_error_pruning(self):

        node = self.tree_root

        stack1 = [node]
        stack2= []

        while(len(stack1)):

            node = stack1.pop()
            stack2.append(node)

            for child in node.children.values():
                if not child.is_terminal:
                    stack1.append(child)
        
        
        pruning_set_prediction = self.get_predictions(self.pruning_features)
        error_on_pruning_set = self.get_error(pruning_set_prediction, self.pruning_labels)
        
        while(len(stack2)):

            node = stack2.pop()

            node.set_class_to_most_common_label()

            new_predictions = self.get_predictions(self.pruning_features)
            new_error = self.get_error(new_predictions, self.pruning_labels)

            if new_error <= error_on_pruning_set:
                error_on_pruning_set = new_error
                node.children.clear()
            else:
                node.is_terminal = False

            
    def train_tree(self, x, y):
        
        data = x.copy()
        labels = y.copy()

        if self.pruning_method in ["cost_complexity", "reduced_error"]:
            pruning_data_percentage = 20
            pruning_data_size = int(pruning_data_percentage / 100 * data.shape[0])
            self.pruning_features = data.iloc[:pruning_data_size]
            self.pruning_labels = labels.iloc[:pruning_data_size]
            data = data.iloc[pruning_data_size:]
            labels = labels.iloc[pruning_data_size:]

        self.change_continuous_to_categorical_values(data, labels, self.number_of_discr_classes)

        #print(data)

        self.change_column_names_to_numbers(data)

        
        data, labels = np.array(data), np.array(labels)


        if len(data.shape) < 2:
            features = [0]
        else:    
            features = list(range(data.shape[1]))

        tree_root = Node(data, labels, features, depth=0)
    
        big_node_stack = [tree_root]
    
        while len(big_node_stack):

            bn = big_node_stack.pop()

            if bn.all_x_have_same_label() or bn.no_features_left() or bn.depth == self.max_depth or bn.x.shape[0] <= self.min_leaf_size:
                bn.set_class_to_most_common_label()
                continue

            bn.find_best_feature_for_split(self.impurity_function)
            children = bn.split_by_feature()
            big_node_stack += children.values()
    
        self.tree_root = tree_root

        self.display_tree("pydot_graph_pre_prune.png")

        if self.pruning_method != None:
            
            if self.pruning_method == "reduced_error":
                self.reduced_error_pruning()
            elif self.pruning_method == "pessimistic_error":
                self.pessimistic_error_pruning()
            elif self.pruning_method == "cost_complexity":
                self.cost_complexity_prunning()
            else:
                raise Exception("Only possible pruning methods are: reduced_error, pessimistic_error and cost_complexity")
    
    def get_class_for_data_point(self, x):
        
        if self.tree_root is None:
            raise Exception("Tree needs to be trained before it can classify")

        bn = self.tree_root
        while(True):
            if bn.is_terminal:
                return bn.my_class
            else:
                bn = bn.get_child(x)
    
    def get_predictions(self, features):
        np.random.seed(1302)
        data = features.copy()
        
        self.change_new_data_continuous_to_categorical_values(data)
        
        data = np.array(data)

        predictions = []
        for data_point in data:
            data_point_class = self.get_class_for_data_point(data_point)
            #print(data_point_class)
            # data_point_class_value = str(self.number_to_label_value_mapping[data_point_class])
            data_point_class_value = str(data_point_class)
            predictions.append(data_point_class_value)
        
        return predictions
    
    @staticmethod
    def get_error(predictions, labels):
         
        predictions = np.array(predictions)
        labels = np.array(labels)

        num_of_predictions = predictions.shape[0]
        
        num_of_different_labels = 0
        for pred, lab in zip(predictions, labels):
            #print(f'pred = {pred} lab = {lab}')
            if str(pred) != str(lab):
                #print(f'pred = {pred} lab = {lab}')
                num_of_different_labels += 1
        #num_of_different_labels = np.sum(predictions != labels)

        error = num_of_different_labels / num_of_predictions
        #print(f'num diff = {num_of_different_labels} num pred = {num_of_predictions}')
        return error


def main():
    np.random.seed(1302)
    data_path = "train.csv"

    #features, labels = load_data(data_path, 'diagnosis', 'id')
    features, labels = load_data(data_path, 'diabetes', separator=',', id_column_name='p_id')

    # features.drop('id', axis=1, inplace=True)

    train_features, train_labels, test_features, test_labels = random_test_train_split(features, labels, test_percentage=20)

    decision_tree = DecisionTree(impurity_function='gini', min_leaf_size=-1, discretization_method='same_size', max_depth=-1, number_of_discr_classes=3, pruning_method='cost_complexity')
    decision_tree.train_tree(train_features, train_labels)

    #decision_tree.display_tree("pydot_graph_pre_prune.png")
    #print('krecem pessimistic')
    
    #decision_tree.pessimistic_error_pruning()
    #print('zavrsio pessimistic')

    np.random.seed(1302)
    decision_tree.display_tree("pydot_graph_post_prune.png")

    predictions = decision_tree.get_predictions(train_features)
    print(f'train score = {1 - decision_tree.get_error(predictions, train_labels)}')

    predictions = decision_tree.get_predictions(test_features)

    print(f'test score = {1 - decision_tree.get_error(predictions, test_labels)}')

    
    #print(train_labels)
    #sklearn_tree = DecisionTreeClassifier(criterion='gini', random_state=1302)
    #sklearn_tree.fit(train_features, train_labels)
    #print(f'train score = {sklearn_tree.score(train_features, train_labels)}')
    #print(f'test score = {sklearn_tree.score(test_features, test_labels)}')
    ##print(sklearn_tree.get_params())
    #plt.figure(figsize=(12,12)) 
    #tree.plot_tree(sklearn_tree, fontsize=10)
    #plt.savefig("skelarn_tree.png")
    ##plt.show()

    
if __name__=="__main__":
    main()
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from math import log2
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pydot
from math import ceil, sqrt
import sys
from collections import deque

#np.set_printoptions(threshold=sys.maxsize)

not_observed = 0

def load_data(data_path):

    data = pd.read_csv(data_path, sep=',')

    features = data.iloc[:, :-1]
    labels = (data.iloc[:, -1])

    change_values_into_numbers(features)
    change_value_into_numbers(labels)
    features = np.array(features)
    labels = np.array(labels)

    return features, labels

def just_load_data(data_path):

    data = pd.read_csv(data_path, sep=',')

    features = data.iloc[:, :-1]
    labels = (data.iloc[:, -1])

    return features, labels

def random_test_train_split(features, labels, test_percentage=20):
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

def change_values_into_numbers(features):
    for column in features.columns:
        values = set(features[column])
        values = sorted(list(values))
        for i, value in enumerate(values):
            features.loc[features[column] == value, column] = i

def change_value_into_numbers(features):

    values = set(features)
    values = sorted(list(values))
    for i, value in enumerate(values):
        features[features == value] = i

class BigNode(object):

    def __init__(self, x, y, features, depth, index=0):
        self.x = x
        self.y = y
        self.features = features
        self.is_terminal = False
        self.my_class = None
        self.my_split_atribute = None
        self.children = {}
        self.children_quantities = {}
        self.index = index
        self.depth = depth


    def all_x_have_same_label(self):
        labels = set(self.y)
        return len(labels) == 1

    def no_features_left(self):
        return len(self.features) == 0
    
    def set_class_to_most_common_label(self):
        
        self.is_terminal = True

        values, counts = np.unique(self.y, return_counts=True)
        ind = np.argmax(counts)
        
        for value, count in zip(values, counts):
            self.children_quantities[value] = count

        self.my_class = values[ind]
    
    def find_best_feature_for_split_entropy(self):

        feature_entropies = {}

        for feature in self.features:
            feature_values = set(self.x[:, feature])
            #print(f'examining feature {feature}')
            total_num_of_examples = self.x.shape[0]
            total_feature_entropy = 0

            for feature_value in feature_values:
                #print(f'examining feature value {feature_value} for feature {feature}')
                feature_value_entropy = 0

                classes = self.y[self.x[:, feature] == feature_value]
                #print(classes)
                total_examples_with_feature_value = len(classes)
                classes_values = set(classes)

                for class_value in classes_values:
                    #print(f'examining class value {class_value}')
                    #print(f'classes == class value = {classes == class_value}')
                    num_exampes_with_feature_value_and_class_value = sum(classes == class_value)
                    class_probability = num_exampes_with_feature_value_and_class_value / total_examples_with_feature_value
                    my_entropy = -class_probability * log2(class_probability)
                    feature_value_entropy += my_entropy

                    #print(f'for feature value {feature_value} class {class_value} has class_probability = {class_probability}')
                    #print(f' num_exampes_with_feature_value {feature_value} and_class_value {class_value} is {num_exampes_with_feature_value_and_class_value} total_examples_with_feature_value = {total_examples_with_feature_value}')
                
                feature_value_ratio = total_examples_with_feature_value / total_num_of_examples
                total_feature_entropy += feature_value_ratio * feature_value_entropy
                #print(f'feature_value_ratio = {feature_value_ratio}')
                #print(f'feature_value_entropy = {feature_value_entropy}')
            #print('\n')
            feature_entropies[feature] = total_feature_entropy
        
        #print(f'features entropies = {feature_entropies}')

        feature_with_min_entropy = min(feature_entropies, key=feature_entropies.get)

        self.my_split_atribute = feature_with_min_entropy

    
    def find_best_feature_for_split_gini(self):
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

        self.my_split_atribute = feature_with_min_gini_impurity


    def find_best_feature_for_classification_error(self):
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

        self.my_split_atribute = feature_with_min_classification_error
    
    def find_best_feature_for_split_chi(self):

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

        self.my_split_atribute = feature_with_min_chi_value
    
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
        feature_values = set(self.x[:, self.my_split_atribute])

        child_features = list(self.features)
        child_features.remove(self.my_split_atribute)
        number_of_examples = len(self.x)

        for feature_value in feature_values:
            indexes = np.where(self.x[:, self.my_split_atribute] == feature_value)
            child_x = self.x[indexes]
            child_y = self.y[indexes]
            
            child = BigNode(child_x, child_y, child_features, depth=self.depth+1)
            self.children[feature_value] = child
        
        values, counts = np.unique(self.y, return_counts=True)
        for value, count in zip(values, counts):
            self.children_quantities[value] = count
        
        return self.children

    def get_child(self, x):
        global not_observed

        if self.is_terminal:
            raise Exception("Leaf nodes have no children")
        
        value = x[self.my_split_atribute]
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

    def __init__(self, impurity_function='gini', discretization_method='dynamic', max_depth=-1, min_leaf_size=-1, k=3, pruning_method=None):
        self.tree_root = None 
        self.impurity_function = impurity_function
        self.number_to_feature_value_mapping = {}
        self.column_index_to_feature_name_mapping = {}
        self.number_to_label_value_mapping = {}
        self.discretization_method = discretization_method
        self.k = k
        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size
        self.numerical_to_categorical_values_mappings = {}
        self.feature_value_to_number_mapping = {}
        self.feature_values = {}
        self.pruning_method = pruning_method

    
    def display_tree(self, path):

        if self.tree_root is None:
            raise Exception("Tree must be trained first in order to be displayed")

        node = self.tree_root
        if node.is_terminal:
            label = str(self.number_to_label_value_mapping[node.my_class])
            color = 'green'
        else:
            label = str(self.column_index_to_feature_name_mapping[node.my_split_atribute])
            color = 'gold'

            label += '\n' + str(node.x.shape[0])
        G = pydot.Dot(graph_type='digraph')
        G.add_node(pydot.Node(name='0', label=label, style='filled', color=color))

        stack = [(self.tree_root, 0)]
        
        index = 0
        while(len(stack)):

            node, parent_index = stack.pop()

            for child_key in node.children:
                child = node.children[child_key]
                index += 1
                if child.is_terminal:
                    label = str(self.number_to_label_value_mapping[child.my_class])
                    color = 'green'
                else:
                    stack.append((child, index))
                    label = str(self.column_index_to_feature_name_mapping[child.my_split_atribute])
                    color = 'gold'
                
                label += '\n' + str(child.x.shape[0])
                G.add_node(pydot.Node(name=str(index), label=label, style='filled', color=color))
                edge_label = self.number_to_feature_value_mapping[node.my_split_atribute][child_key]
                G.add_edge(pydot.Edge(str(parent_index), str(index), label=edge_label))
                
        
        G.write_png(path)


    def change_feature_values_into_numbers(self, features):
        
        number_to_feature_value_mapping = {}
        column_index_to_feature_name_mapping = {}
        feature_value_to_number_mapping = {}

        for column_index, column_name in enumerate(features.columns):
        
            values = self.feature_values[column_name]
            column_index_to_feature_name_mapping[column_index] = column_name

            number_to_value_mapping = {}
            value_to_number_mapping = {}

            patches = []
            for value_number, value in enumerate(values):
                patch_indexes = features[column_name] == value
                patches.append((patch_indexes, value_number))
                number_to_value_mapping[value_number] = value
                value_to_number_mapping[value] = value_number
            
            for patch in patches:
                patch_indexes, value_number = patch
                features.loc[patch_indexes, column_name] = value_number

            number_to_feature_value_mapping[column_index] = number_to_value_mapping
            feature_value_to_number_mapping[column_index] = value_to_number_mapping
                
        self.feature_value_to_number_mapping = feature_value_to_number_mapping
        self.number_to_feature_value_mapping = number_to_feature_value_mapping
        self.column_index_to_feature_name_mapping = column_index_to_feature_name_mapping

        
    def change_label_values_into_numbers(self, features):
        number_to_label_value_mapping = {}

        values = set(features)
        values = sorted(list(values))

        patches = []
        for value_number, value in enumerate(values):
            patch_indexes = features == value
            patches.append((patch_indexes, value_number))
            number_to_label_value_mapping[value_number] = value

        for patch in patches:
            patch_indexes, value_number = patch
            features[patch_indexes] = value_number
        
        self.number_to_label_value_mapping = number_to_label_value_mapping
    
        
    def split_to_intervals_with_same_length(self, features_df, column_name, k):
        feature_column = features_df[column_name]

        min_value = feature_column.min()
        max_value = feature_column.max()

        split_vectors = []
        split_size = (max_value - min_value) / k

        left_split_border = min_value
        for i in range(k-1):
            right_split_border = left_split_border + split_size

            if i == 0:
                split_vector = feature_column <= right_split_border
            else:
                split_vector = np.logical_and(left_split_border <= feature_column, feature_column <= right_split_border)
            
            split_vectors.append((split_vector, left_split_border, right_split_border))

            if i == k-2:
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
        
    
                
    def split_to_intervals_with_same_size(self, features_df, column_name, k):
 
        features_column = features_df[column_name]

        sorted_array = np.sort(features_column)
        m = features_column.shape[0]

        min_value = sorted_array[0]
        max_value = sorted_array[m-1]

        split_vectors = []

        number_of_examples_in_split = ceil(m / k)
        left_split_border_value = min_value

        for i in range(k-1):
            right_split_border_index = min(number_of_examples_in_split * (i+1), m-1)
            right_split_border_value = sorted_array[right_split_border_index]

            if i == 0:
                split_vector = features_column <= right_split_border_value
            else:
                split_vector = np.logical_and(left_split_border_value < features_column, features_column <= right_split_border_value)

            #split_vector = np.logical_and(left_split_border_value < features_df, features_df <= right_split_border_value)
            split_vectors.append((split_vector, left_split_border_value, right_split_border_value))

            if i == k-2:
                split_vector = right_split_border_value < features_column
                split_vectors.append((split_vector, left_split_border_value, right_split_border_value))

            left_split_border_value = right_split_border_value

        #a[a == max_value] = k-1
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

    def change_continuous_to_categorical_values(self, features, labels, k):

        for column_index, column_name in enumerate(features.columns):

            values = set(features[column_name])

            # column is continous if it has more than 5 distinct values
            if len(values) > 5:
                if self.discretization_method == 'same_length':
                    self.split_to_intervals_with_same_length(features, column_name, k)
                if self.discretization_method == 'same_size':
                    self.split_to_intervals_with_same_size(features, column_name, k)
                if self.discretization_method == 'dynamic':
                    self.dynamic_split(features, column_name, labels)

    def change_new_data_continuous_to_categorical_values(self, features):
        for column_name in self.numerical_to_categorical_values_mappings:
            border_values = self.numerical_to_categorical_values_mappings[column_name]

            patches = []

            left_border = border_values[0]
            for border_index, border_value in enumerate(border_values):
                right_border = border_value

                if border_index == 0:
                    group_indexes = features[column_name] <= right_border
                    #features.loc[group_indexes, column_name] = f'x <= {right_border}'
                    patches.append((group_indexes, f'x <= {right_border}'))
                elif border_index == len(border_values) - 1:
                    group_indexes = features[column_name] > right_border
                    if self.discretization_method == 'dynamic':
                        group_indexes = features[column_name] >= right_border
                    #features.loc[group_indexes, column_name] = f'x >= {left_border}'
                    patches.append((group_indexes, f'x > {right_border}'))
                else:
                    group_indexes = np.logical_and(left_border < features[column_name], features[column_name] <= right_border)
                    #features.loc[group_indexes, column_name] = f'{left_border} <= x <= {right_border}'
                    patches.append((group_indexes, f'{left_border} < x <= {right_border}'))

                left_border = right_border

            for patch in patches:
                group_indexes, value = patch
                features.loc[group_indexes, column_name] = value
    
    def change_new_data_feature_values_into_numbers(self, x):
        for column_index, column_name in enumerate(x.columns):
            feature_values_to_numbers = self.feature_value_to_number_mapping[column_index]
            #print(x[column_name])
            #print('\n' + column_name)
            patches = []
            for feature_value in feature_values_to_numbers:
                number = feature_values_to_numbers[feature_value]
                group_indexes = x[column_name] == feature_value
                
                #print(feature_value)
                patches.append((group_indexes, number))
          

            for patch in patches:
                group_indexes, value = patch
                x.loc[group_indexes, column_name] = value 
                #print(x.loc[group_indexes, column_name])
                #print()

    

    
            
    def pessimistic_error_prunning(self):

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
                print("prune")
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
                print("pruning")
            else:
                print(f'old error is {error_on_pruning_set} while new error is {new_error}')
                node.is_terminal = False

            
    def train_tree(self, x, y):
        
        data = x.copy()
        labels = y.copy()

        if self.pruning_method != None:
            pruning_data_percentage = 20
            pruning_data_size = int(pruning_data_percentage / 100 * data.shape[0])
            self.pruning_features = data.iloc[:pruning_data_size]
            self.pruning_labels = labels.iloc[:pruning_data_size]
            data = data.iloc[pruning_data_size:]
            labels = labels.iloc[pruning_data_size:]

        self.change_continuous_to_categorical_values(data, labels, self.k)

        #print(data)

        self.change_feature_values_into_numbers(data)
        self.change_label_values_into_numbers(labels)
        data, labels = np.array(data), np.array(labels)


        if len(data.shape) < 2:
            features = [0]
        else:    
            features = list(range(data.shape[1]))
        tree_root = BigNode(data, labels, features, depth=0)
    
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

        if self.pruning_method != None:
            if self.pruning_method == "reduced_error":
                pass
                #self.reduced_error_pruning()
            elif self.pruning_method == "pessimistic_error":
                #self.pessimistic_error_prunning()
                print("nothing")
            else:
                pass
                #raise Exception("Only possible pruning methods are: reduced_error")
    
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
        #print(data)
        self.change_new_data_feature_values_into_numbers(data)
        
        data = np.array(data)

        predictions = []
        for data_point in data:
            data_point_class = self.get_class_for_data_point(data_point)
            #print(data_point_class)
            data_point_class_value = str(self.number_to_label_value_mapping[data_point_class])
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
    features, labels = just_load_data(data_path)

    features.drop('p_id', axis=1, inplace=True)

    train_features, train_labels, test_features, test_labels = random_test_train_split(features, labels, test_percentage=20)

    decision_tree = DecisionTree(impurity_function='gini', min_leaf_size=-1, discretization_method='same_size', max_depth=-1, k=3, pruning_method="nista")
    decision_tree.train_tree(train_features, train_labels)
    decision_tree.display_tree("pydot_graph_pre_prune.png")
    print('krecem pessimistic')
    decision_tree.pessimistic_error_prunning()
    print('zavrsio pessimistic')
    decision_tree.display_tree("pydot_graph_post_prune.png")

    predictions = decision_tree.get_predictions(train_features)
    print(f'train score = {1 - decision_tree.get_error(predictions, train_labels)}')

    predictions = decision_tree.get_predictions(test_features)

    print(f'test score = {1 - decision_tree.get_error(predictions, test_labels)}')

    return
    #predictions = decision_tree.get_predictions(test_features)
    #print(f'test score = {1 - decision_tree.get_error(predictions, test_labels)}')
    global not_observed
    print(not_observed)
    
    train_labels=train_labels.astype('int')
    test_labels=test_labels.astype('int')
    sklearn_tree = DecisionTreeClassifier(criterion='gini', random_state=1302)
    sklearn_tree.fit(train_features, train_labels)
    print(f'train score = {sklearn_tree.score(train_features, train_labels)}')
    #print(f'test score = {sklearn_tree.score(test_features, test_labels)}')
    #print(sklearn_tree.get_params())
    #tree.plot_tree(sklearn_tree)
    #plt.show()

    
if __name__=="__main__":
    main()
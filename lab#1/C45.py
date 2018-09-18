# Author: Yilin Zheng
# Date: 18 Sept, 2018
# Reference code: https://github.com/AliceDudu/Learning-Notes/blob/master/Machine-Learning-Algorithms/\
#                 DecisionTrees/self-learning-c45algorithm-steps.ipynb

from math import log
import operator


class C45():
    
    
    def __init__(self):
        pass
    
    
    def preprocess(self, dataset, continus_features):
        """
        preprocess continus data, here I just simple keep the integer part
        """
        pass
        for data in dataset:
            for feature in continus_features:
                data[feature] = int(data[feature])
        return dataset
    

    def fit(self, dataset, labels, continus_features=None):
        """
        fit data into model
        """
        if continus_features is not None:
            dataset = self.preprocess(dataset, continus_features)
        return self.create_tree(dataset, labels)
    
    
    def cal_entropy(self, dataset):
        """
        calculate Shannon entropy
        """
        entry_num = len(dataset)
        label_counts = {}
        for data in dataset:
            data_label = data[-1]  # default that the last value of the data is its label
            if data_label not in label_counts.keys():
                label_counts[data_label] = 0
            label_counts[data_label] += 1
        entropy = 0.0
        for label in label_counts:
            prob = float(label_counts[label])/entry_num
            entropy -= prob * log(prob, 2)
        return entropy
    
    
    def classify_node(self, labels):
        """
        label the node of tree
        """
        label_counts = {}
        for label in labels:
            if label not in label_counts.keys():
                label_counts[label] = 0
            label_counts[label] += 1
        sorted_label_counts = sorted(label_counts.iteritems(), key=operator.itemgetter(1), reversed=True)
        return sorted_label_counts[0][0]
    
    
    def select_feature(self, dataset):
        """
        select the best feature to span the brach
        """
        feature_num = len(dataset[0])-1
        base_entropy = self.cal_entropy(dataset)
        max_entropy_gain_ratio = 0.0
        best_feature = -1
        for i in range(feature_num):
            feature_values = set([data[i] for data in dataset])
            new_entrypy = 0.0
            split_info = 0.0
            for value in feature_values:
                subdataset = self.remove_entry(dataset, i, value)
                prob = len(subdataset)/float(len(dataset))
                new_entrypy += prob * self.cal_entropy(subdataset)
                split_info += -prob * log(prob, 2)
            entroy_gain = base_entropy - new_entrypy
            if (split_info == 0):
                continue
            entropy_gain_ratio = entroy_gain / split_info
            if (entroy_gain > max_entropy_gain_ratio):
                max_entropy_gain_ratio = entropy_gain_ratio
                best_feature = i
        return best_feature

    
    def remove_entry(self, dataset, entry_num, value):
        """
        remove the entry in data, which is used as a label on node
        the generated new dataset does not contain any value of used feature
        """
        new_dataset = []
        for data in dataset:
            if data[entry_num] == value:                
                new_data = data[:entry_num] + data[entry_num+1:]  
                new_dataset.append(new_data)            
        return new_dataset

    
    def create_tree(self, dataset, labels):
        """
        create tree recursively
        """
        classes = [data[-1] for data in dataset]
        if classes.count(classes[0]) == len(classes):
            return classes[0]
        if len(dataset[0]) == 1:
            return self.classify_node(classes)
        best_feature = self.select_feature(dataset)
        best_feature_label = labels[best_feature]
        decision_tree = {best_feature_label:{}}
        del(labels[best_feature])
        feature_values = set([data[best_feature] for data in dataset])
        for value in feature_values:
            sublabels = labels[:]
            decision_tree[best_feature_label][value] = self.create_tree(self.remove_entry(dataset, best_feature, value),\
                                                                       sublabels)
        return decision_tree  

    
    def predict_a_data(self, decision_tree, test_data, labels):
        """
        predict label for a data
        """
        root_feature = list(decision_tree.keys())[0]
        second_dict = decision_tree[root_feature]
        feature_index = labels.index(root_feature)
        feature_label = None
        for key in second_dict.keys():
            if test_data[feature_index] == key:
                if type(second_dict[key]).__name__ == 'dict': 
                    feature_label = self.predict_a_data(second_dict[key], test_data, labels)
                else:
                    feature_label = second_dict[key]      
        return feature_label
    

    def predict(self, decision_tree, test_dataset, labels):
        """
        predict labels for all data
        """
        feature_label_all = []
        for test_data in test_dataset:
            feature_label_all.append(self.predict_a_data(decision_tree, test_data, labels))
        return feature_label_all
# Functions for downloading and reading SUBPOP Astra Zeneca dataset
from __future__ import print_function
import gzip
import os
import urllib

import numpy as np
import pandas as pd

from sklearn.cross_validation import train_test_split

LABELED_RL_PATH = '../subpop_data/B6_robust_linear_training.txt'
UNLABELED_RL_PATH = '../subpop_data/B6_robust_linear_test.txt'
FEATURES = 97
N_CLASSES = 23

'''
Wrapper for Shantanus txt input to numpy dataset

Input: Tab Separated Txt 
Output: Dictionary with numpy arrays
'''
def convert_txt_to_npy(txt_path, labeled=True):
    print("=> Extracting data from txt")
    df = pd.read_csv(txt_path, sep='\t', header=None)
    dic = {}
    
    if labeled:
        # First index is label, second index is compound        
        # Convert string labels to numbers
        y = df[0].values
        y_unique = np.unique(y)
        y_dic = {}
        for i, label in enumerate(y_unique):
            y_dic[label] = i

        y_numeric = []
        for el in y:
            y_numeric += [y_dic[el]]
            
        dic['labels'] = np.array(y_numeric)
        dic['compounds'] = df[1].values
        dic['data'] = df.iloc[:,2:].values
        print('=> Extracted %i labeled objects with %i features' % (df.iloc[:,2:].shape))
    else:
        # First index is object_id, second index is compound
        dic['object_id'] = df[0].values
        dic['compound'] = df[1].values
        dic['data'] = df.iloc[:,2:].values
        print('=> Extracted %i unlabeled objects with %i features' % (df.iloc[:,2:].shape))
    return dic

'''
Function converting labels into one_hot representations
'''
def convert_to_one_hot(labels):
    from keras.utils import np_utils
    return np_utils.to_categorical(labels, N_CLASSES)

'''
Train-Test Split function

return X_train, X_test, y_train, y_test
'''
def split_train_test(dic, test_size=0.2):
    print("=> Split labeled data into %.02f training and %.02f test" % ((1 - test_size), test_size))
    return train_test_split(dic['data'], dic['labels'], test_size=test_size, random_state=42)


'''
Class DataSet from inspired by input.py from TensorFlow examples
'''
class DataSet(object):
    
    def __init__(self, data, labels, fake_data=False):
        if fake_data:
            self._num_examples = 1000
        else:
            assert data.shape[0] == labels.shape[0], ("data.shape: %s labels.shape %s" % (data.shape, labels.shape))
            self._num_examples = data.shape[0]
            
        self._data = data
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        
    @property
    def data(self):
        return self._data
    
    @property
    def labels(self):
        return self._labels
    
    @property
    def num_examples(self):
        return self._num_examples
    
    @property
    def epochs_completed(self):
        return self._epochs_completed
    
    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set"""
        if fake_data:
            fake_data = [1.0 for _ in xrange(FEATURES)]
            fake_label = 0
            return [fake_data for _ in xrange(batch_size)], [fake_label for _ in xrange(batch_size)]
            
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self.num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._data = self._data[perm]
        self._labels = self._labels[perm]
        # Start next epoch
        start = 0
        self._index_in_epoch = batch_size
        assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._data[start:end], self._labels[start:end]
            
'''
SemiDataSet inspired by Ladder Network Implementation
'''
class SemiDataSet(object):
    def __init__(self, unlabeled_data, labeled_data, labels):
        self.n_labeled = labels.shape[0]
        
        # Unlabeled DataSet
        l_arr = np.array([0 for _ in xrange(unlabeled_data.shape[0])])
        self.unlabeled_ds = DataSet(unlabeled_data, l_arr)
        
        # Corrected Labeled DataSet
        self.labeled_ds = DataSet(labeled_data, labels)
        
    def next_batch(self, batch_size):
        unlabeled_data, _ = self.unlabeled_ds.next_batch(batch_size)
        if batch_size > self.n_labeled:
            labeled_data, labels = self.labeled_ds.next_batch(self.n_labeled)
        else:
            labeled_data, labels = self.labeled_ds.next_batch(batch_size)
        data = np.vstack([labeled_data, unlabeled_data])
        return data, labels

'''
Read data and convert it into an DataSet object
'''
def read_subpop_data(one_hot=True, fake_data=False, test_size=0.2):

	labeled_dic = convert_txt_to_npy(LABELED_RL_PATH)
	unlabeled_dic = convert_txt_to_npy(UNLABELED_RL_PATH, labeled=False)
	X_train, X_test, y_train, y_test = split_train_test(labeled_dic, test_size=test_size)

	class DataSets(object):
	    pass
	data_sets = DataSets()

	if one_hot:
		y_train = convert_to_one_hot(y_train)
		y_test = convert_to_one_hot(y_test)

	data_sets = DataSets()
	data_sets.test = DataSet(X_test, y_test)
	data_sets.train = SemiDataSet(unlabeled_dic['data'], X_train, y_train)

	return data_sets
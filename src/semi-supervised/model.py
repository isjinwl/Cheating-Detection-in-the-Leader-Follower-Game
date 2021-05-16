#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Model class label propagation"""

import pandas as pd

class LabelPropagation:
    """
    Label propagation class.
    """
    def __init__(self, trainDataDF, trainLabelDF, testDataDF, testLabelDF):
        """
        Setting up the label propagation object.
        :param trainDataDF: The training set.
        :param trainLabelDF: Labels of the training set.
        :param testDataDF: The testing set.
        :param testLabelDF: Labels of the testing set.
        """
        self.trainDataDF = trainDataDF
        self.trainLabelDF = trainLabelDF
        self.testDataDF = testDataDF
        self.testLabelDF = testLabelDF
        self.merge_data()
    
    def merge_data(self):
        """
        Merging the training set and the testing set.
        """
        mergeDataDF = pd.concat([self.trainDataDF, self.testDataDF], sort='True')
        mergeDataDF = mergeDataDF.reset_index(drop=True)
        self.mergeDataDF = mergeDataDF  
    
    def distance_matrix_setup(self):
        """
        Calculating the euclidean distance.
        """
        object_num = self.mergeDataDF.shape[0]
        matrix_distance = np.zeros((object_num, object_num), dtype=float, order='C')
        for i in range(object_num):
            


    def MST_kruscal(self):
        """
        """




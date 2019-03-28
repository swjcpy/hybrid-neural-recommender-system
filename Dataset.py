'''
    Created on Aug 8, 2016
    Processing datasets.
    
    @author: Xiangnan He (xiangnanhe@gmail.com)
    '''

"""
    Revision: Weijie Sun (swjcpy@umich.edu)
    """
import scipy.sparse as sp
import numpy as np

class Dataset(object):
    '''
        classdocs
        '''
    
    def __init__(self, path):
        '''
            Constructor
            '''
        self.trainMatrix = self.load_rating_file_as_matrix(path + "train_data.csv")
        self.testRatings = self.load_rating_file_as_list(path + "test_data.csv")
        self.testNegatives = self.load_negative_file(path + "test_negative.csv")
        self.trainFeature = self.load_train_feature(path + "users.csv")
        assert len(self.testRatings) == len(self.testNegatives)
        
        self.num_users, self.num_items = self.trainMatrix.shape
    
    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            # while line != None and line != "":
            while line is not None and line != "":
                arr = line.strip().split(",")
                user, item = int(arr[0]), int(float(arr[1]))
                ratingList.append([user, item])
                line = f.readline()
        return ratingList
    
    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            # while line != None and line != "":
            while line is not None and line != "":
                line = line.split('"')[1]
                arr = line.strip().split(",")
                negatives = []
                for x in arr[:]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList
    
    def load_rating_file_as_matrix(self, filename):
        '''
            Read .rating file and Return dok matrix.
            The first line of .rating file is: num_users\t num_items
            '''
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            # while line != None and line != "":
            while line is not None and line != "":
                arr = line.strip().split(",")
                u, i = int(arr[0]), int(float(arr[1]))
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        # Construct matrix
        mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            # while line != None and line != "":
            while line is not None and line != "":
                arr = line.strip().split(",")
                user, item = int(arr[0]), int(float(arr[1]))
                mat[user, item] = 1.0
                line = f.readline()
        return mat

def load_train_feature(self, filename):
    feature_list = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(',')
            if line[1] == 'M':
                line[1] = 1
                elif line[1] == 'F':
                    line[1] = -1
            else:
                line[1] = 0
                try:
                    line[2] = int(float(line[2]))
                    line[3] = int(float(line[3]))
                    line[4] = int(line[4].split('-')[0])
            except:
                line[2] = 0
                line[3] = 0
                line[4] = 0
                feature_list.append(line[1:])

    return feature_list

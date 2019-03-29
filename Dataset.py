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
import pdb

num_samples = 6040
num_genders = 2
num_jobs = 21
num_ages = 56

class Dataset(object):
    '''
        classdocs
        '''
    
    def __init__(self, path):
        '''
            Constructor
            '''
        self.trainMatrix = self.load_rating_file_as_matrix(path + ".train.rating")
        self.testRatings = self.load_rating_file_as_list(path + ".test.rating")
        self.testNegatives = self.load_negative_file(path + ".test.negative")
        self.userFeature = self.load_train_feature(path + ".users")
        self.itemFeature = self.load_item_feature(path + ".movies")
        assert len(self.testRatings) == len(self.testNegatives)
        
        self.num_users, self.num_items = self.trainMatrix.shape
    
    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            # while line != None and line != "":
            while line is not None and line != "":
                arr = line.strip().split("\t")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                line = f.readline()
        return ratingList
    
    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            # while line != None and line != "":
            while line is not None and line != "":
                arr = line.strip().split("\t")
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
                arr = line.strip().split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        # Construct matrix
        mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            # while line != None and line != "":
            while line is not None and line != "":
                arr = line.strip().split("\t")
                user, item = int(arr[0]), int(arr[1])
                mat[user, item] = 1.0
                line = f.readline()
        return mat

    def load_train_feature(self, filename):
        feature_list = []
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split("\t")
                line[1] = int(line[1])
                line[2] = int(line[2])
                line[3] = int(line[3])
                line[4] = int(line[4])
                feature_list.append(line[1:])

        return feature_list

    def load_item_feature(self, filename):
        # build dictionary
        word_dict = {}
        max_length = 0
        sentences = []
        
        with open(filename, 'r') as f:
            lines = f.readlines()

            for line in lines:
                line = line.strip().split(" ")
                sentences.append(line)
                max_length = max(max_length, len(line))
                for word in line[1:]:
                    if word not in word_dict:
                        word_dict[word] = len(word_dict) + 1
        print("movie dictionary size: ")
        print(len(word_dict))

        # define feature list
        feature_list = {}
        print("max length: ")
        print(max_length)

        for i, sentence in enumerate(sentences):
            temp = np.zeros((max_length-1))
            idx = int(sentence[0])
            for j, word in enumerate(sentence[1:]):
                temp[j] = word_dict[word]
            feature_list[idx] = temp

        return feature_list
            

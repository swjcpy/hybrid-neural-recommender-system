'''
This piece of code is for train test file generation that matches the model training pipeline
Created by Shiyu Wang shiyuw@umich.edu
'''
import numpy as np
import pandas as pd
import random
import pdb

def main(path, ratings_file, train_file, test_file, test_negative_file, movieidx, negative_samples=100, num_movies=3592):
    
    ## process the rating file

    all_data = []
    with open(path + ratings_file, "r") as f:
        line = f.readline()
        while line is not None and line != "":
            line = line.split("::")
            line[0] = int(line[0])
            line[1] = int(line[1])
            line[2] = int(line[2])
            if len(all_data) < line[0]:
                all_data.append([])

            all_data[line[0]-1].append((line[1], line[2]))
            line = f.readline()
    print("finish loading all data")
    # generate train & test file
    train = []
    test = []
    test_negative = []
    for i, l in enumerate(all_data):
        # train sample
        for j in range(0,len(l)-1):
            train.append((i+1, l[j]))
        # test sample
        test.append((i+1, l[len(l)-1]))
        # test negative sample
        test_negative.append([])
        for k in range(negative_samples):
            randnum = np.random.randint(num_movies)
            if randnum not in all_data[i] and randnum in movieidx:
                test_negative[i].append(randnum)
    print("finish train/test split and negative number generation")
    # write to file
    with open(path + train_file, "w") as f:
        for line in train:
            f.write("%d\t%d\t%d\n" % (line[0], line[1][0], line[1][1]))
        f. close()
    print("finish writing training to file")
    with open(path + test_file, "w") as f:
        for line in test:
            f.write("%d\t%d\t%d\n" % (line[0], line[1][0], line[1][1]))
        f.close()
    print("finish writing testing to file")
    with open(path + test_negative_file, "w") as f:
        for line in test_negative:
            for item in line:
                f.write(str(item) + "\t")
            f.write("\n")
        f.close()
    print("finish writing test negative to file")
         
def user(path, users_file, saved_users_file):
    user_data = []
    num_users = 6040
    num_genders = 2
    num_age = 56
    num_jobs = 21
    max_num = 0
    with open(path + users_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split("::")
            line[0] = int(line[0])
            if line[1] == "F":
                line[1] = 1 + num_users
            else:
                line[1] = 2 + num_users
            line[2] = int(line[2]) + num_users + num_genders
            line[3] = int(line[3]) + num_users + num_genders + num_age
            line[4] = int(line[4][:5]) + num_users + num_genders + num_age + num_jobs
            max_num = max(max_num, line[4])
            user_data.append(line)

    print(max_num)
    
    with open(path + saved_users_file, "w") as f:
        for data in user_data:
            f.write("%d\t%d\t%d\t%d\t%d\n" % (data[0], data[1], data[2], data[3], data[4]))
        f.close()
    print("finish writing user info to file")

def movie(path, movie_file, saved_movie_file):
    movie_data = []
    random_word = []
    # read random text and save to list
    with open(path + "randomtext", 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(" ")
            for word in line:
                random_word.append(word)

    # process movie data
    print("processing movie data")
    movies = pd.read_csv(path + 'movies.csv', engine='python')
    movies = movies[['0', '1','2','overview']]
    shape = movies.shape[0]
    movieidx = []
    for i in range(shape):
        words = []
        words += [movies.iloc[i]['0']]
        movieidx.append(int(movies.iloc[i]['0']))
        words += movies.iloc[i]['1'].split(" ")
        words += movies.iloc[i]['2'].split("|")
        if  movies.iloc[i].isna()['overview']:
            temp = []
            randnums = random.sample(range(len(random_word)), 50)
            for randnum in randnums:
                temp.append(random_word[randnum])
                words += temp
        else:
            words += movies.iloc[i]['overview'].split(" ")
        movie_data.append(words)

    # write random_word to file
    with open(path + saved_movie_file, "w") as f:
        for movie in movie_data:
            for word in movie:
                f.write("%s " % word)
            f.write("\n") 
        f.close()
    print("finish writing movie info to file")

    return movieidx
    

if __name__ == "__main__":
    user("Data/", "users.dat", "ml-1m.users")
    movieidx = movie("Data/", "movies.csv", "ml-1m.movies")
    main("Data/", "ratings.dat", "ml-1m.train.rating", "ml-1m.test.rating", "ml-1m.test.negative", movieidx)
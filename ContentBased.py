import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from ast import literal_eval
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

meta_data = pd.read_csv("data/meta_data.csv", low_memory=False)

data = meta_data[['title', 'credits','overview','release_date',
       'production_countries', 'genres', 'vote_average']]
data = data.drop_duplicates()

tfidf = TfidfVectorizer(stop_words='english')
data['overview'] = data['overview'].fillna('')
tfidf_matrix = tfidf.fit_transform(data['overview'])
tfidf_matrix.shape #33087 different words to describe 10471 movies in the dataset

cosine_sim = linear_kernel(tfidf_matrix,tfidf_matrix)
indices = pd.Series(data.index, index=data['title']).drop_duplicates()

def get_recommendation(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return data['title'].iloc[movie_indices]


def create_movie(x):
    return x['production_countries']+x['genres']+x['credits']+str(x['release_date'])+str(x['vote_average'])

data['all'] = data.apply(create_movie, axis=1)

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(data['all'])
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
data = data.reset_index()
indices = pd.Series(data.index, index = data['title'])

# print('Overview Based')
# print(get_recommendation('Frozen (2013)') )# not very good
# print('-'*20)

print('Movies similar to Frozen (2013)')
print(get_recommendation('Frozen (2013)', cosine_sim2))

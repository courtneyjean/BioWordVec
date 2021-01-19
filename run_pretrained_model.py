from gensim.models import FastText
import numpy as np

filename = 'data/bio_embedding_intrinsic'

model = FastText.load(filename)
print(model)

test_word_1 = "computer"
print("TEST 1")
print(model.most_similar(test_word_1))

test_word_2 = "heart attack"
print("TEST 2")
print(model.most_similar(test_word_2))

word_vectors = model.wv

print(word_vectors.get_vector('office').shape)

words = word_vectors.get_vector('heart')
corpus =['computer', 'travel', 'dog']

distances = word_vectors.distances(words, corpus)

print("DISTANCES")
print(distances)

words = word_vectors.get_vector('heart')
corpus =[word_vectors.get_vector('computer'), word_vectors.get_vector('travel'), word_vectors.get_vector('dog')]

distances = word_vectors.cosine_similarities(words, corpus)

print("COSINE SIMILARITY")
print(distances)

print('Word Vector')
print(words)
print(words.shape)

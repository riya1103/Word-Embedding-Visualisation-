import pandas as pd
import re 
import os
import gensim
import numpy as np
from gensim.models import Word2Vec, FastText
import tensorflow as tf

with open('all_linkedin_skill_data.txt') as f:
    mylist = list(f)
    temp = mylist[0]
    l =  temp.split(']')
    l.pop(0)
    for x in range(0,len(l)):
        l[x] = l[x][3:]
        l[x] = l[x].split(', ')


l[:2]
l[0]


m=[[j.lower() for j in i] for i in l]

while([""]) in m:
    m.remove([""])
 
    
    
model = Word2Vec(l, min_count=1,size= 150,workers=3, window =3, sg = 1)
model2 = Word2Vec(l, min_count=1,size= 50,workers=3, window =3, sg = 0)
model['AML']

model.similarity('AML','Contract Negotiation')
model.similarity('')




remove_string = m[2824776]
remove_string
len(m)
m.remove(remove_string)

m2 = [x for x in m if x!=remove_string]
len(m2)
x=m2[1]
x+x
x=['tele']
m2[1]
model1=gensim.models.Word2Vec([m2], min_count=1,size= 150,workers=4, window =5, sg = 0)

model3=Word2Vec(m2, min_count=1,size= 150,workers=3, window =3, sg = 1)

model1['AML']

model1.similarity('AML','Contract Negotiation')
model1.most_similar('AML')
model1.most_similar('"Programming Languages: C/C++')

from sklearn import preprocessing
#calculating similarity by cosine distance
def cosine_distance (model, word,target_list , num) :
    cosine_dict ={}
    word_list = []
    a = model[word]
    for item in target_list :
        if item != word :
            b = model [item]
            cos_sim = np.dot(a, b)/(preprocessing.normalize(a)*preprocessing.normalize(b))
            cosine_dict[item] = cos_sim
    dist_sort=sorted(cosine_dict.items(), key=lambda dist: dist[1],reverse = True) ## in Descedning order 
    for item in dist_sort:
        word_list.append((item[0], item[1]))
    return word_list[0:num]

cosine_distance(model1,'AML',l[0],5)

import matplotlib.pyplot as plt
word='AML'
size=10

def display_closestwords_tsnescatterplot(model1, word, size):    
    arr = np.empty((0,size), dtype='f')
    word_labels = [word]
    close_words = model1.similar_by_word(word)
    arr = np.append(arr, np.array([model1[word]]), axis=0)
    for wrd_score in close_words:
        wrd_vector = model1[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)
        
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)
    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    plt.scatter(x_coords, y_coords)
    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
        plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
        plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
    plt.show()

display_closestwords_tsnescatterplot(model1, 'AML', 10) 
from sklearn.manifold import TSNE

def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model1.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()
tsne_plot(model1)

with open('closest_skills_validation_set.txt') as f:
    mylist1 = list(f)

df2=pd.DataFrame(mylist1)
validation_set=df2[0].str.split('->',n=1,expand=True)
df4=pd.DataFrame()

m2[1]
model1['telecommunications']

len(model1.vocab)

remove_string = m[2824776]
remove_string
len(m)
m.remove(remove_string)

len(m2)


aa =[]
df4[0]='result'

for i in range(0,50):
    aa.append(model3.most_similar(positive=validation_set[0][i],topn=20))
   
'''model1.most_similar('t')
model1.most_similar('e')
model1.most_similar('a')
model1.most_similar('+')'''

df4 = pd.DataFrame(aa)

for i in range(9,50):
    aa.append(model1.most_similar(positive=validation_set[0][i],topn=20))


differences = []

for list in aa:
    if list not in validation_set[1]:
        differences.append(list)
        
accuracy=[]
acc=0
for i in range(0,50):
    accuracy[i]=20-differences[i]
    acc=acc+accuracy[i]

accuracy_overall=acc/50
        
vocab = model1.vocabulary

model1.most_similar('telecommunications')
model1.most_similar('word')



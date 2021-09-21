#!/usr/bin/env python
# coding: utf-8

# In[1]:


from network import CNetwork


# In[2]:


import pandas as pd
import numpy as np


# In[3]:


import os


# In[4]:


import embeddings


# In[5]:


name_dataset = "dataset_1"
dataset_path = "datasetsv2/"
length_cut = 20000
random_flag = True
remove_punctuation = True
lemmatization_flag = True
feature_selection = 'common_words'
measures = ["dgr_n", "btw", "cc", "sp", "sp_std", "accs_h2", "accs_h3"]
embeddings_name ='w2v'
embedding_percentages = [index for index in range(1,21)]


# In[6]:


from utils import verifyDir
auxiliar_path = 'auxiliar_folder/' + name_dataset   + '/'
verifyDir(auxiliar_path)


# In[7]:


df = pd.read_csv(dataset_path + name_dataset + ".csv")


# In[8]:


df.head(5)


# In[9]:


from utils.text_processing import get_min_len_corpus


# In[10]:


print("Min Length:", get_min_len_corpus(list(df["text"])))


# In[11]:


from utils.text_processing import get_corpus, get_random_corpus


# In[12]:


texts = list(df['text'])


# In[13]:


corpus, segmented_corpus = get_corpus(texts, length_cut)


# In[14]:


selected_corpus, words_features, word_index, index_word = get_random_corpus(segmented_corpus, remove_punctuation=remove_punctuation, lemmatization_flag=lemmatization_flag,feature_selection = feature_selection)


# In[15]:


len(selected_corpus)


# In[16]:


labels = list(df['label'])


# In[17]:


total_classes = list(set(labels))  ## or author
print("Total classes: {}".format(len(total_classes)))
number_books = (df[df['label'] == total_classes[0]]).shape[0]
print("Total entities for each class in train: {}".format(number_books))
dict_categories = {cat: index for index, cat in enumerate(total_classes)}


# In[18]:


y = [dict_categories[y] for y in labels]


# In[19]:


total_classes


# In[20]:


print('Training word embeddings ....')
objEmb = embeddings.WordEmbeddings(corpus, embeddings_name)
model = objEmb.get_embedding_model()
#model = ''
print('Words trained: ',len(model))
print('Word embeddings sucessfully trained')


# In[21]:


dimensions = len(embedding_percentages) + 1


# In[22]:


def get_global_features(sequences, index_word):
    container_features = [[] for _ in range(dimensions)]
    for i,text in enumerate(sequences):
        obj = CNetwork(text, model=model, index_word=index_word, percentages=embedding_percentages, path="")
        networks = obj.create_networks()
        global_measures = [obj.get_network_global_measures(network, measures) for network in networks]
        for j in range(dimensions):
            container_features[j].append(global_measures[j])
    return np.array(container_features)


# In[ ]:


container_features = get_global_features(selected_corpus, index_word)


# In[ ]:


print("Lenght of features:", container_features[0].shape)


# In[ ]:


container_features[0]


# # Normalize data

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler(with_mean=True, with_std=True)


# # Classification

# In[ ]:


import classifierv2


# In[ ]:


for X in container_features:
    X = scaler.fit_transform(X)
    obj = classifierv2.Classification(X, y)
    scores = obj.classification()


# In[ ]:





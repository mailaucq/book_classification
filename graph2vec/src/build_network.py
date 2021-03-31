#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 16:39:50 2020

@author: usuario
"""

import csv
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk import bigrams
import igraph
from igraph import *
import json
import platform
import pandas as pd
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import model_selection as sk_ms
import os
import shutil
import xnet


def create_network(sentence):
    string_bigrams = bigrams(sentence)
    edges = []
    network = Graph()
    vertexs = list(set(sentence))
    for i in vertexs:
        network.add_vertex(str(i))
    for i in string_bigrams:
        edges.append((network.vs.find(str(i[0])), network.vs.find(str(i[1]))))    
    network.add_edges(edges)
    network.simplify()
    return network


def igraph2json(g, fileName='test.json'):
    N = g.vcount()
    E = g.ecount()
    edges_dict = {}
    edges_list = []
    for i in range(E):		
        edge = g.es[i].tuple
        edges = [int(g.vs[edge[0]]["name"]), int(g.vs[edge[1]]["name"])]
        edges_list.append(edges)
    edges_dict["edges"] = edges_list
    
    with open(fileName, 'w') as outfile:
        json.dump(edges_dict, outfile)


def create_auxiliar_dir(path):
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)


def remove_auxiliar_dir(path):
    try:
        shutil.rmtree(path)
    except OSError:
        print ("Removing of the directory %s failed" % path)
    else:
        print ("Successfully removed the directory %s " % path)


def read_result_file(path):
    index = 0
    result = dict()
    file = open(path)
    for line in file.readlines():
        line = line.rstrip('\n')
        value = float(line)
        result[index] = value
        index+=1
    return result


def shortest_path(network, features):
    average_short_path = network.shortest_paths(features)
    result = []
    for path in average_short_path:
        average = float(np.divide(np.sum(path), (len(path) - 1)))
        result.append(average)
    return result


def graph2vec(networks, embedding_dim):
    in_network = 'auxiliar_folder/'
    out_network = 'auxiliar_folder_features/'
    out_network_file = out_network + "nci1.csv"
    create_auxiliar_dir(in_network)
    create_auxiliar_dir(out_network)
    for i, network in networks.items():
        igraph2json(network, in_network + str(i) + ".json")
    if operating_system == 'linux':
        path_command = 'python /home/usuario/Documentos/projecto/graph2vec/src/graph2vec.py --input-path ' + in_network + ' --output-path ' + out_network_file + ' --dimensions ' + str(embedding_dim)
    os.system(path_command)
    g2v = pd.read_csv(out_network_file, sep=',', lineterminator='\n')
    g2v.set_index("type")
    all_network_features = [g2v.loc[i].values for i, _ in enumerate(networks)]
    remove_auxiliar_dir(in_network)
    remove_auxiliar_dir(out_network)
    return all_network_features


def symmetry(network, features):
    in_network = 'auxiliar_folder/'
    out_network = 'auxiliar_folder_features/'
    create_auxiliar_dir(in_network)
    create_auxiliar_dir(out_network)
    in_network = in_network + 'auxiliar_network.xnet'
    output = out_network + 'auxiliar.csv'
    xnet.igraph2xnet(network, in_network)
    if operating_system == 'linux':
        path_command = './extras/concentrics/linux/CVSymmetry_linux -c -M -l 3 ' + in_network + ' ' + output
    else:
        path_command = './extras/concentrics/mac/CVSymmetry_mac -c -M -l 3 ' + in_network + ' ' + output

    os.system(path_command)
    simetrias = pd.read_csv(output, sep='\t', lineterminator='\n')

    back_sym_2 = 'Backbone Accessibility h=2'
    merg_sym_2 = 'Merged Accessibility h=2'
    back_sym_3 = 'Backbone Accessibility h=3'
    merg_sym_3 = 'Merged Accessibility h=3'

    v1 = np.array(simetrias[back_sym_2])
    v2 = np.array(simetrias[merg_sym_2])
    v3 = np.array(simetrias[back_sym_3])
    v4 = np.array(simetrias[merg_sym_3])
    sim_results = [v1, v2, v3, v4]
    keys = ['bSym2', 'mSym2', 'bSym3', 'mSym3']
    final_results = dict()
    for key, values in zip(keys, sim_results):
        valid_syms = []
        for word in features:
            node = network.vs.find(name=word)
            valid_syms.append(values[node.index])
        final_results[key] = valid_syms
    remove_auxiliar_dir(in_network)
    remove_auxiliar_dir(out_network)
    return final_results


def accessibility(network, features, h):
    in_network = 'auxiliar_folder/'
    out_network = 'auxiliar_folder_features/'
    create_auxiliar_dir(in_network)
    create_auxiliar_dir(out_network)
    in_network = in_network + 'auxiliar_network.xnet'
    extra_file = out_network + 'acc_results.txt'
    xnet.igraph2xnet(network, in_network)
    if operating_system == 'linux':
        path_command = './extras/concentrics/linux/CVAccessibility_linux -l ' + str(h) + ' ' + in_network + ' > ' + extra_file
    else:
        path_command = './extras/concentrics/mac/CVAccessibility_mac -l ' + str(h) + ' ' + in_network + ' > ' + extra_file

    #print(path_command)
    os.system(path_command)
    accs_values = read_result_file(extra_file)
    result = []
    for word in features:
        node = network.vs.find(name=word)
        result.append(accs_values[node.index])
    remove_auxiliar_dir(in_network)
    remove_auxiliar_dir(out_network)
    return result


def get_common_words(texts):
    commom_words = texts[0]
    for index, i in enumerate(texts):
        commom_words = list(set(commom_words) & set(i))
    result = {word: index for index, word in enumerate(commom_words)}
    return result
    

def get_network_measures(network, features):
    found_features = []
    for word in features:
        try:
            node = network.vs.find(name=word)
        except:
            node = None
        if node is not None:
            found_features.append(word)
    dgr = network.degree(found_features)
    pr = network.pagerank(found_features)
    btw = network.betweenness(found_features)
    cc = network.transitivity_local_undirected(found_features)
    sp = shortest_path(network, found_features)
    symmetries = symmetry(network, found_features)
    bSym2 = symmetries['bSym2']
    mSym2 = symmetries['mSym2']
    bSym3 = symmetries['bSym3']
    mSym3 = symmetries['mSym3']
    accs_h2 = accessibility(network, found_features, 2)
    accs_h3 = accessibility(network, found_features, 3)
    measures = [dgr, pr, btw, cc, sp, bSym2, mSym2, bSym3, mSym3, accs_h2, accs_h3]
    #measures = [bSym2, mSym2, bSym3, mSym3, accs_h2, accs_h3]
    network_features = []
    for measure in measures:
        feature = [0.0 for _ in range(len(features))]
        for word, value in zip(found_features, measure):
            feature[features[word]] = value
        network_features.extend(feature)
    network_features = np.array(network_features)
    network_features[np.isnan(network_features)] = 0
    print('Len features:', len(network_features))
    return network_features


def classification(features, labels, kfold):
    c1 = DecisionTreeClassifier(random_state=0)
    c2 = KNeighborsClassifier(n_neighbors=5) ## testar outros parametros 3 41.6666666  ### 5 45.
    c3 = GaussianNB()
    c4 = SVC(kernel='linear', probability=True)
    classifiers = [c1,c2,c3,c4]
    results = []
    stds = []
    for i in classifiers:
        scores = sk_ms.cross_val_score(i, features, labels, cv=kfold, scoring='accuracy', n_jobs=-1, verbose=0)
        score = round(scores.mean() * 100, 2)
        sd = round(scores.std()*100, 2)
        results.append(score)
        stds.append(sd)
    return results, stds

      
plataforma = platform.platform()
if plataforma.find('Linux') != -1:
    operating_system = 'linux'
else:
    operating_system = 'mac'

#vocab_size = 20102# YOUR CODE HERE
embedding_dim = 256# YOUR CODE HERE
#max_length = 1700# YOUR CODE HERE
trunc_type = "post"# YOUR CODE HERE
padding_type = "post"# YOUR CODE HERE
oov_tok = "<OOV>"# YOUR CODE HERE
training_portion = .8
sentences = []
labels = []
remove_stop_words = True
stopwords = [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]
print("Size stopwords: {}".format(len(stopwords)))

dataset_file = "../datasets/dataset_3.csv"
dataset = pd.read_csv(dataset_file)
labels = dataset["label"].str.replace("_", "")
if remove_stop_words:
    for word in stopwords:
        dataset["words"] = dataset["words"].str.replace(" " + word + " ", " ")
sentences = dataset["words"].values
    
print("Size labels: {}".format(len(labels)))
print("Size sentences: {}".format(len(sentences)))

#tokenizer = Tokenizer(num_words=vocab_size,oov_token=oov_tok)# YOUR CODE HERE
tokenizer = Tokenizer(oov_token=oov_tok)# YOUR CODE HERE
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index# YOUR CODE HERE

print("Size vocabulary: {}".format(len(word_index)))
sequences = tokenizer.texts_to_sequences(sentences)# YOUR CODE HERE
min_length = np.min([len(sequences[i]) for i in range(len(sequences))])
print("Min length sequence: {}".format(min_length))
padded = pad_sequences(sequences, padding=padding_type, maxlen=min_length, truncating=trunc_type)# YOUR CODE HERE
print("Length padded: {}".format(len(padded[0])))
#print(sentences[0])
#print(sequences[0])
#print(padded[0])

label_tokenizer = Tokenizer()# YOUR CODE HERE
label_tokenizer.fit_on_texts(labels)
label_seq = np.array(label_tokenizer.texts_to_sequences(labels))# YOUR CODE HERE
authors = label_tokenizer.word_index

#print(label_seq[0])
#print(label_seq.shape)
print("Number of Authors: {}".format(len(authors)))

number_books = (labels == next(iter(authors))).sum()
print("Number of books per author: {}".format(number_books))

networks = {}
vertex_count = []
edges_count = []
for i in range(len(padded)):
    network = create_network(padded[i])
    networks[i] = network
    vertex_count.append(network.vcount())
    edges_count.append(network.ecount())
    #print('Nodes:', network.vcount(), '-', 'Edges:', network.ecount())


print('Nodes:', mean(vertex_count), '-', 'Edges:', mean(edges_count))
        
features = graph2vec(networks, embedding_dim)
#print(features)
label_seq = np.ravel(label_seq)
print(classification(features, label_seq, number_books))

word_common = get_common_words(padded)
print("Size word common: {}".format(len(word_common)))
all_network_features = [get_network_measures(net, word_common) for i, net in networks.items()]
print(classification(all_network_features, label_seq, number_books))    

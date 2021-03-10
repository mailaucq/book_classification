import pandas as pd
from collections import Counter
import numpy as np
import random
import embeddings
import network
import os
from sklearn.preprocessing import StandardScaler
import classifier
import shutil
import igraph2json
import platform


#df = pd.read_csv('datasets/dataset_1.csv') # 8 authors - 5 books per author  # min 40mil words # vanessa paper
#df3 = pd.read_csv('datasets/dataset_3.csv') # 8 authors - 6 books per author #min 30mil words # stanisz paper inf science
#df4 = pd.read_csv('datasets/brown_dataset.csv') #--> category # label  [seleccion aleatoria balancaemiento] # henrique brown corpus
#df2 = pd.read_csv('datasets/dataset_2.csv') # 20 authors - 5 books per author #min 3 mil words vannesa opcional

def join_lists(l1, l2):
    str_res = ''
    for x,y in zip(l1, l2):
        str_res+= str(x) + '(+/-' + str(y) + ') '
    return str_res


def graph2vec(path, operating_system, networks):
    in_network = path + 'dataset/'
    extra_file = path + 'features/nci1.csv'
    try: 
        os.mkdir(in_network)
        os.mkdir(path + 'features/')
    except:
        print("Existe")
    for i, netw in enumerate(networks):
        igraph2json.igraph2json1(netw, in_network + str(i) + ".json")
    if operating_system == 'linux':
        path_command = 'python ./graph2vec/src/graph2vec.py --input-path ' + in_network + ' --output-path ' + extra_file + ' --dimensions ' + str(512)
    os.system(path_command)
    g2v = pd.read_csv(extra_file, sep=',', lineterminator='\n')
    g2v.set_index("type")
    all_network_features = [g2v.loc[i].values for i, _ in enumerate(networks)]
    return all_network_features


class BookClassification(object):

    def __init__(self, dataset='vanessa', text_partition=1000, embeddings='w2v', feature_selection='common_words', sampling=1):
        self.dataset = self.load_dataset(dataset)
        print('Dataset ' + dataset + ' loaded')
        self.dataset.info()
        print()
        self.text_partition = text_partition
        self.embeddings = embeddings
        self.feature_selection = feature_selection
        self.number_iterations = sampling
        self.embedding_percentages = [index for index in range(1,21)]
         # [1, 5, 10, 15, 20]
        name = dataset + '_' + str(text_partition) + '_' + feature_selection +  '_' + str(self.number_iterations) 
        self.path = 'auxiliar_folder/' + name   + '/'
        self.output_file = 'results/' + name + '.txt'
        try: 
            os.mkdir(self.path)
        except:
            print("Existe")
        #os.mkdir(self.path)
        print(self.path)
        #a = input()
        plataforma = platform.platform()
        if plataforma.find('Linux') != -1:
            self.operating_system = 'linux'
        else:
            self.operating_system = 'mac'

    def load_dataset(self, name):
        if name == 'vanessa':
            path = 'datasets/dataset_1.csv'
        elif name == 'stanisz':
            path = 'datasets/dataset_3.csv'
        else:
            path = 'datasets/brown_dataset_balanced.csv'
        return pd.read_csv(path)

    def get_word_index(self, texts):
        word_index = {}
        index_word = {}
        count = 0
        for text in texts:
            for word in list(set(text)):
                if word not in word_index:
                    word_index[word] = str(count)
                    index_word[str(count)] = word
                    count += 1
        return word_index, index_word
    
    def get_sequences(self, texts, word_index):
        sequences = []
        for text in texts:
            sequence = []
            for w in text:
                sequence.append(word_index[w])
            sequences.append(sequence)
        return sequences
    
    def get_common_words(self, texts): 
        commom_words = texts[0]
        for index, i in enumerate(texts):
            commom_words = list(set(commom_words) & set(i))
        result = {word: index for index, word in enumerate(commom_words)}
        return result
    
    def get_top_words(self, texts, number=None):
        if self.feature_selection.find('top_')!=-1:
            top_words = int(self.feature_selection[self.feature_selection.rfind('_') + 1:])
        else:
            top_words = number
        all_words = []
        for text in texts:
            all_words.extend(list(set(text)))
        counts = Counter(all_words)
        features = counts.most_common(top_words)
        most_commom = dict()
        for index, feat in enumerate(features):
            most_commom[feat[0]] = index
        return most_commom

    # get corpus(array of texts), segmented corpus( texto dividido en m particiones de min_size, labels
    def get_corpus(self):
        labels = list(self.dataset['label'])
        texts = list(self.dataset['words'])
        corpus = [i.split() for i in texts]
        #corpus = [i[:self.text_partition] for i in corpus]
        min_size = self.text_partition
        size_partitions = []
        segmented_corpus = [] # segment corpus
        for book in corpus:
            partitions = int(round(len(book)/min_size,2) + 0.5) #? por que mas 0.5?
            segments = [book[int(round(min_size * i)): int(round(min_size * (i + 1)))] for i in range(partitions)]
            size_partitions.append(len(segments))
            segmented_corpus.append(segments)
        #self.iterations = int(np.mean(size_partitions))
        return corpus, segmented_corpus, labels

    # get random partitions of book	
    def get_random_corpus(self, segmented_corpus, mode_sequences=True):
        selected = []
        for partitions in  segmented_corpus:
            if self.number_iterations == 1:
                random_index = 0
            else:
                random_index = random.randint(0, len(partitions) - 1)
            selected.append(partitions[random_index])
        if mode_sequences:
            word_index, index_word = self.get_word_index(selected)
            selected = self.get_sequences(selected, word_index)
        if self.feature_selection == 'common_words':
            words_features = self.get_common_words(selected)
        else:
            words_features = self.get_top_words(selected)

        if len(words_features)==0:
            words_features = self.get_top_words(selected, 1)
        return selected, words_features, word_index, index_word

    def get_corpus_scores(self, corpus, classes, dict_categories, model, number_books):
        selected_corpus, words_features, word_index, index_word = self.get_random_corpus(corpus)
        print('Word features: ',len(words_features), words_features)
        labels = []
        dimensions = len(self.embedding_percentages) + 1
        all_features_container = [[] for _ in range(dimensions)]

        for index, (book, category) in enumerate(zip(selected_corpus, classes)):
            print('book:', index + 1)
            print('category:', category)
            labels.append(dict_categories[category])
            selected_partition = book
            obj = network.CNetwork(selected_partition, model,   index_word, self.embedding_percentages, self.path)
            cNetworks = obj.create_networks()
            all_network_features = [obj.get_network_measures(net, words_features) for net in cNetworks]
            for net_index, features in enumerate(all_network_features):
                all_features_container[net_index].append(features)
            print()
        print('Feature generation finished, now classifying\n')

        scaler = StandardScaler(with_mean=True, with_std=True)
        limiar_scores = []
        limiar_sds = []
        for limiar_index, limiar_features in enumerate(all_features_container):
            limiar_features = np.array(limiar_features)
            limiar_features = scaler.fit_transform(limiar_features)
            print(limiar_index, limiar_features.shape)
            obj = classifier.Classification(limiar_features, labels, number_books)
            scores, sds = obj.classification()
            limiar_scores.append(scores)
            limiar_sds.append(sds)
            print(scores, sds)
            print()
        return limiar_scores, limiar_sds
    
    def get_corpus_scores_g2v(self, segmented_corpus, classes, dict_categories, model, number_books):
        selected_corpus, words_features, word_index, index_word = self.get_random_corpus(segmented_corpus, mode_sequences=True)
        print('Word features: ',len(words_features), words_features)
        labels = []
        dimensions = len(self.embedding_percentages) + 1
        all_features_container = [[] for _ in range(dimensions)]
        cNetworks = []
        for index, (book, category) in enumerate(zip(selected_corpus, classes)):
            print('book:', index + 1)
            print('category:', category)
            labels.append(dict_categories[category])
            selected_partition = book
            obj = network.CNetwork(selected_partition, model, index_word, self.embedding_percentages, self.path)
            cNetworks = obj.create_networks()
            for net_index, features in enumerate(cNetworks):
                all_features_container[net_index].append(features)
            print()
        print('Feature generation finished, now classifying\n')
        
        scaler = StandardScaler(with_mean=True, with_std=True)
        limiar_scores = []
        limiar_sds = []
        for limiar_index, limiar_features in enumerate(all_features_container):
            #limiar_features = limiar_features)
            limiar_features = np.array(graph2vec(self.path, self.operating_system, limiar_features))
            #limiar_features = scaler.fit_transform(limiar_features)
            print(limiar_index, limiar_features.shape)
            obj = classifier.Classification(limiar_features, labels, number_books)
            scores, sds = obj.classification()
            limiar_scores.append(scores)
            limiar_sds.append(sds)
            print(scores, sds)
            print()
        return limiar_scores, limiar_sds
        """limiar_scores = []
        limiar_sds = []
        obj = classifier.Classification(all_network_features, labels, number_books)
        scores, sds = obj.classification()
        limiar_scores.append(scores)
        limiar_sds.append(sds)
        print(scores, sds)"""

    def classification_analysis(self):
        corpus, segmented_corpus, labels = self.get_corpus()
        classes = list(self.dataset['label'])  ## or 'author'
        total_classes = list(set(self.dataset['label']))  ## or author
        number_books = (self.dataset[self.dataset['label'] == total_classes[0]]).shape[0]
        dict_categories = list(set(classes))
        dict_categories = {cat: index for index, cat in enumerate(dict_categories)}
        print('Clases:', classes)
        print('Total classes:', len(total_classes), total_classes)
        print('Number books per classe: ', number_books)
        print(dict_categories)

        print('Training word embeddings ....')
        objEmb = embeddings.WordEmbeddings(corpus, self.embeddings)
        model = objEmb.get_embedding_model()
        #model = ''
        print('Words trained: ',len(model))
        print('Word embeddings sucessfully trained')

        dimensions = len(self.embedding_percentages) + 1
        iteration_score_container = [[] for _ in range(dimensions)]
        iteration_sd_container = [[] for _ in range(dimensions)]

        #self.get_corpus_scores(segmented_corpus, classes, dict_categories, model, number_books)
        print('\n\n')
        for i in range(self.number_iterations):
            print('Init of iteration ' + str(i+1) + ' .......')
            limiar_scores, limiar_sds = self.get_corpus_scores_g2v(segmented_corpus, classes, dict_categories, model, number_books)
            for index, (score, sd) in enumerate(zip(limiar_scores, limiar_sds)):
                iteration_score_container[index].append(score)
                iteration_sd_container[index].append(sd)
            print('End of iteration ' + str(i + 1) + ' .......')
            print('\n')

        print('\nFinal results:')
        file_result = open(self.output_file, 'w')

        for index, (it_score, it_sd) in enumerate(zip(iteration_score_container, iteration_sd_container)):
            it_score = np.array(it_score)
            it_sd = np.array(it_sd)
            it_score = np.mean(it_score, axis=0)
            it_sd = np.mean(it_sd, axis=0)
            it_score = [round(i,2) for i in it_score]
            it_sd = [round(i, 2) for i in it_sd]
            str_result = join_lists(it_score, it_sd)
            file_result.write(str_result + '\n')
            print(str_result)
            print()
        file_result.close()
        shutil.rmtree(self.path)
        

if __name__ == '__main__':

    dataset = 'brown' # 'vanessa' 'brown' 'stanisz'
    size = 2000
    feat_sel = 'common_words' # top_50  common_words
    iterations = 2
    obj = BookClassification(dataset=dataset, text_partition=size, feature_selection=feat_sel, sampling=iterations)
    obj.classification_analysis()
    

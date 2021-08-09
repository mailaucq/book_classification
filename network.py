
import igraph
from igraph import *
from nltk import bigrams
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy import integrate
import platform
import xnet
import pandas as pd


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

def get_largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)

class CNetwork(object):

    def __init__(self, document, model, index_word, percentages, path):
        self.document = document
        self.model = model
        self.percentages = percentages
        self.words = list(set(self.document))
        self.word_dict = {word:index for index, word in enumerate(self.words)}
        self.path = path
        self.index_word = index_word
        plataforma = platform.platform()
        if plataforma.find('Linux') != -1:
            self.operating_system = 'linux'
        else:
            self.operating_system = 'mac'

    def create_network(self):
        edges = []
        string_bigrams = bigrams(self.document)
        for i in string_bigrams:
            edges.append((i[0], i[1]))
            #edges.append((self.word_dict[i[0]], self.word_dict[i[1]]))

        network = Graph()
        network.add_vertices(self.words)
        #network.add_vertices(len(self.words))
        network.add_edges(edges)
        network.simplify()
        print('Nodes:', len(self.words), '-', 'Edges:', len(network.get_edgelist()))
        return network

    def add_embeddings(self, network):
        network_size = network.vcount()
        actual_edges = network.get_edgelist()
        num_edges = network.ecount()
        maximum_num_edges = int((network_size * (network_size - 1)) / 2)
        remaining_edges = maximum_num_edges - num_edges
        print('Testing available edges:', maximum_num_edges, remaining_edges)
        edges_to_add = []

        for percentage in self.percentages:
            value = int(num_edges * percentage / 100) + 1
            edges_to_add.append(value)
        #print(edges_to_add)
        #words = list(set(self.document))
        matrix = []
        for word in self.words:
            embedding = self.model[word]
            matrix.append(embedding)

        matrix = np.array(matrix)
        similarity_matrix = 1 - pairwise_distances(matrix, metric='cosine')
        similarity_matrix[np.triu_indices(network_size)] = -1
        similarity_matrix[similarity_matrix == 1.0] = -1
        largest_indices = get_largest_indices(similarity_matrix, maximum_num_edges)

        max_value = np.max(edges_to_add)
        counter = 0
        index = 0
        new_edges = []
        while counter < max_value:
            x = largest_indices[0][index]
            y = largest_indices[1][index]
            if not network.are_connected(x, y):
                new_edges.append((x, y))
                counter += 1
            index += 1

        networks = []
        for value in edges_to_add:
            edges = []
            edges.extend(actual_edges)
            edges.extend(new_edges[0:value])
            new_network = Graph()
            new_network.add_vertices(self.words)
            new_network.add_edges(edges)
            networks.append(new_network)
        return networks
    
    
    def add_embeddings2(self, network):
        network_size = network.vcount()
        actual_edges = network.get_edgelist()
        num_edges = network.ecount()
        maximum_num_edges = int((network_size * (network_size - 1)) / 2)
        remaining_edges = maximum_num_edges - num_edges
        print('Testing available edges:', maximum_num_edges, remaining_edges)
        edges_to_add = []

        for percentage in self.percentages:
            value = int(num_edges * percentage / 100) + 1
            edges_to_add.append(value)
        #print(edges_to_add)
        #words = list(set(self.document))
        matrix = []
        for word in self.words:
            embedding = self.model[self.index_word[word]]
            matrix.append(embedding)

        matrix = np.array(matrix)
        similarity_matrix = 1 - pairwise_distances(matrix, metric='cosine')
        similarity_matrix[np.triu_indices(network_size)] = -1
        similarity_matrix[similarity_matrix == 1.0] = -1
        largest_indices = get_largest_indices(similarity_matrix, maximum_num_edges)

        max_value = np.max(edges_to_add)
        counter = 0
        index = 0
        new_edges = []
        while counter < max_value:
            x = largest_indices[0][index]
            y = largest_indices[1][index]
            if not network.are_connected(self.words[x], self.words[y]):
                new_edges.append((self.words[x], self.words[y]))
                counter += 1
            index += 1

        networks = []
        for value in edges_to_add:
            edges = []
            edges.extend(actual_edges)
            edges.extend(new_edges[0:value])
            new_network = Graph()
            new_network.add_vertices(self.words)
            new_network.add_edges(edges)
            networks.append(new_network)
        return networks

    def create_networks(self):
        network = self.create_network()
        networks = self.add_embeddings2(network)
        networks.insert(0, network)

        prueba = [len(net.get_edgelist()) for net in networks]
        print('Num edges in networks:', prueba)
        return networks

    def get_weighted_network(self, words):
        matrix = []
        for word in words:
            embedding = self.model[word]
            matrix.append(embedding)
        matrix = np.array(matrix)
        similarity_matrix = 1 - pairwise_distances(matrix, metric='cosine')
        simList = similarity_matrix.tolist()
        return Graph.Weighted_Adjacency(simList, mode="undirected", attr="weight", loops=False)

    def get_alpha(self, k, p_ij):
        try:
            alpha = 1 - (k - 1) * integrate.quad(lambda x: (1 - x) ** (k - 2), 0, p_ij)[0]
        except:
            alpha = 1
        return alpha

    def disparity_filter(self, network):
        print('Calculating disparity filter')
        degree = network.degree()
        for vertex in range(network.vcount()):
            k = degree[vertex]
            neighbors = network.neighbors(vertex)
            if k > 1:
                sum_w = network.strength(vertex, weights=network.es['weight'])
                for v in neighbors:
                    w = network.es[network.get_eid(vertex, v)]['weight']
                    p_ij = float(np.absolute(w)) / sum_w
                    alpha_ij = self.get_alpha(k, p_ij)
                    network.es[network.get_eid(vertex, v)]['alpha'] = alpha_ij
            else:
                network.es[network.get_eid(vertex, neighbors[0])]['alpha'] = 0

    def add_filtered_embeddings(self, network):
        print('testing ...')
        weighted_network = self.get_weighted_network(self.words)
        #print('len edges', len(weighted_network.get_edgelist()))
        self.disparity_filter(weighted_network)
        #alphas = weighted_network.es['alpha']
        #print('len edges', len(weighted_network.get_edgelist()), len(alphas))

        extra_edges = []
        all_edges = weighted_network.es
        print('Looking for embedding edges')
        for edge_values in all_edges:
            edge = edge_values.tuple
            if not network.are_connected(edge[0], edge[1]):
                extra_edges.append(edge_values)

        sorted_extra_edges = sorted(extra_edges, key=lambda x: x["alpha"], reverse=False)
        sorted_extra_edges = sorted_extra_edges[0:len(network.get_edgelist())]

        worst_edges =  sorted_extra_edges[::-1]

        num_edges = len(sorted_extra_edges)#*2
        top_k_to_remove = []

        for percentage in self.percentages:
            top_k = int(num_edges * percentage / 100) + 1
            top_k_to_remove.append(top_k)

        auxiliar_network = Graph()
        auxiliar_network.add_vertices(self.words)
        edges = [(e.source, e.target) for e in sorted_extra_edges]
        auxiliar_network.add_edges(network.get_edgelist())
        auxiliar_network.add_edges(edges)

        print('Num edges:', num_edges)
        print('top k remove:',top_k_to_remove)
        print('worst edges:', len(worst_edges))
        filtered_networks = []
        for top in top_k_to_remove:
            remove = worst_edges[0:top]
            r_edges = [(e.source, e.target) for e in remove]
            new_network = auxiliar_network.copy()
            new_network.delete_edges(r_edges)
            filtered_networks.append(new_network)
        return filtered_networks


    def create_filtered_networks(self):
        network = self.create_network()
        networks = self.add_filtered_embeddings(network)
        networks.insert(0, network)

        prueba = [len(net.get_edgelist()) for net in networks]
        print('Num edges in networks:', prueba)
        a = input()
        return networks


    def get_frequency_counts(self, networks, features):
        found_features = []
        for word in features:
            try:
                node = networks[0].vs.find(name=word)
            except:
                node = None
            if node is not None:
                found_features.append(word)
        network_features = []
        for network in networks:
            dgr = network.degree(found_features)
            feature = [0.0 for _ in range(len(features))]
            for word, value in zip(found_features, dgr):
                feature[features[word]] = value
            #network_features.extend(feature)
            network_features.append(feature)
        network_features = np.array(network_features)
        return network_features

    def shortest_path(self, network, features=None):
        if features is not None:
            average_short_path = network.shortest_paths(features)
        else:
            average_short_path = network.shortest_paths()
        result = []
        for path in average_short_path:
            average = float(np.divide(np.sum(path), (len(path) - 1)))
            result.append(average)
        return result

    def symmetry(self, network, features=None):
        in_network = self.path + 'auxiliar_network.xnet'
        output = self.path + 'auxiliar.csv'
        xnet.igraph2xnet(network, in_network)
        if self.operating_system == 'linux':
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
            if features is not None:
                for word in features:
                    node = network.vs.find(name=word)
                    valid_syms.append(values[node.index])
            else:
                valid_syms = values
            final_results[key] = valid_syms
        return final_results

    def accessibility(self, network, h, features=None):
        in_network = self.path + 'auxiliar_network.xnet'
        extra_file = self.path + 'acc_results.txt'
        xnet.igraph2xnet(network, in_network)
        if self.operating_system == 'linux':
            path_command = './extras/concentrics/linux/CVAccessibility_linux -l ' + str(h) + ' ' + in_network + ' > ' + extra_file
        else:
            path_command = './extras/concentrics/mac/CVAccessibility_mac -l ' + str(h) + ' ' + in_network + ' > ' + extra_file

        #print(path_command)
        os.system(path_command)
        accs_values = read_result_file(extra_file)
        result = []
        if features is not None:
            for word in features:
                node = network.vs.find(name=word)
                result.append(accs_values[node.index])
        else:
            result = accs_values
        return result

        
    def get_network_measures(self, network, features, measures_names=None):
        found_features = []
        for word in features:
            try:
                node = network.vs.find(name=word)
            except:
                node = None
            if node is not None:
                found_features.append(word)
        measures = []
        if measures_names is None:
        	measures_names = ["sp"]#["dgr_n", "btw", "cc", "sp", "sp_std", "accs_h2", "accs_h3"]
        if "dgr_n" in measures_names:
                dgr_n = network.knn(found_features) #pr = network.pagerank(found_features)
                dgr_n = np.average(dgr_n)
                measures.append(dgr_n)
        if "btw" in measures_names:
                btw = network.betweenness(found_features)
                measures.append(btw)
        if "cc" in measures_names:
                cc = network.transitivity_local_undirected(found_features)
                measures.append(cc)
        if "sp" in measures_names:
                sp = self.shortest_path(network, found_features)
                measures.append(sp)
        if "sym" in measures_names:
                symmetries = self.symmetry(network, found_features)
                bSym2 = symmetries['bSym2']
                mSym2 = symmetries['mSym2']
                bSym3 = symmetries['bSym3']
                mSym3 = symmetries['mSym3']
                measures.append(bSym2)
                measures.append(mSym2)
                measures.append(bSym3)
                measures.append(mSym3)
        if "accs_h2" in measures_names:
                accs_h2 = self.accessibility(network, 2, found_features)
                measures.append(accs_h2)
        if "accs_h3" in measures_names:
                accs_h3 = self.accessibility(network, 3, found_features)
                measures.append(accs_h3)
        #measures = [dgr, pr, btw, cc, sp, bSym2, mSym2, bSym3, mSym3, accs_h2, accs_h3]
        #measures = [bSym2, mSym2, bSym3, mSym3, accs_h2, accs_h3]
        network_features = []
        for measure in measures:
        	print("MEASURE",measure)
        	feature = [0.0 for _ in range(len(features))]
        	for word, value in zip(found_features, measure):
        		feature[features[word]] = value
        	network_features.extend(feature)
        network_features = np.array(network_features)
        
        network_features[np.isnan(network_features)] = 0
        print('Len features:', len(network_features))
        print(network_features)
        return network_features

    def get_network_global_measures(self, network, measures):
        network_features = []
        if "dgr_n" in measures:
            dgr_n, _ = network.knn()
            dgr_n = np.average(dgr_n)
            network_features.append(dgr_n)#pr = np.average(network.pagerank())
        if "btw" in measures:
            btw = network.betweenness()
            btw = np.average([0 if np.isnan(b) else b for b in btw])
            network_features.append(btw)
        if "cc" in measures:
            cc = network.transitivity_local_undirected()
            cc = np.average([0 if np.isnan(c) else c for c in cc])
            network_features.append(cc)
        if "sp" in measures:
            sp1 = self.shortest_path(network)
            sp = np.average([0 if np.isnan(s) else s for s in sp1])
            network_features.append(sp)
        if "sp_std" in measures:
            #	sp1 = self.shortest_path(network)
            sp_std = np.std([0 if np.isnan(s) else s for s in sp1])
            network_features.append(sp_std)
            #symmetries = self.symmetry(network)
            #bSym2 = np.average(symmetries['bSym2'])
            #mSym2 = np.average(symmetries['mSym2'])
            #bSym3 = np.average(symmetries['bSym3'])
            #mSym3 = np.average(symmetries['mSym3'])
        if "accs_h2" in measures:
            accs_h2 = list(self.accessibility(network, 2).values())
            accs_h2 = np.average([0 if np.isnan(a) else a for a in accs_h2])
            network_features.append(accs_h2)
        if "accs_h3" in measures:
            accs_h3 = list(self.accessibility(network, 3).values())
            accs_h3 = np.average([0 if np.isnan(a) else a for a in accs_h3])
            network_features.append(accs_h3)
        network_features_arr = np.array(network_features)
        network_features_arr[np.isnan(network_features_arr)] = 0
        print('Len features:', len(network_features_arr))
        return network_features_arr
        


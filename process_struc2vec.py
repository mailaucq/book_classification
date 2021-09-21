import os
import numpy as np

def get_struc2vec(operating_system, networks, length_cut, limiar_index, words_features, index_word, dimension=512):
    all_network_features = []
    words_features = {index_word[k]:v for k,v in words_features.items()}
    print("len words features", len(words_features)) 
    if len(words_features) > 0:
        input_path = "struc2vec/graph/"+str(length_cut)+"_"+str(limiar_index)+"_"+str(dimension)+"/"
        output_path = "struc2vec/emb/"+str(length_cut)+"_"+str(limiar_index)+"_"+str(dimension)+"/"
        try:
            os.mkdir(input_path)
            os.mkdir(output_path)
        except:
            print("Existe")
        for i, netw in enumerate(networks):
            with open((input_path+str(i)+".edgelist"), "w") as f:
                v_names = netw.vs["name"]
                for edge in netw.get_edgelist():
                    (u,v) = edge
                    f.write(v_names[u] +" "+ v_names[v]+'\n')
        for i, netw in enumerate(networks):
            if operating_system == 'linux':
                path_command = "python struc2vec/src/main.py --dimensions "+str(dimension)+" --input "+input_path+str(i)+".edgelist --output "+output_path+str(i)+".emb"
                os.system(path_command)
                print("struc2vec", str(i))

        for i, netw in enumerate(networks):
            network_features = [[] for _ in range(len(words_features))]
            with open(output_path+str(i)+".emb", "r") as f:
                lines = f.readlines()
                num_token, dim = lines[0].split()
                for l in lines[1:]:
                    row = l.strip().split(" ")
                    id_node = row[0]
                    if index_word[id_node].lower() in words_features:
                        emb = [float(v) for v in row[1:]]
                        network_features[words_features[index_word[id_node].lower()]] = emb
                network_features = np.array(network_features).flatten()
                print(str(i), "len",len(network_features))
                all_network_features.append(network_features)   
    return np.array(all_network_features)

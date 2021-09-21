import json
import os
import numpy as np


def get_graph2vec(operating_system, networks, length_cut, limiar_index, dimension=512):
    input_path = "graph2vec/dataset/"+str(length_cut)+"_"+str(dimension)+"_"+str(limiar_index)+"/"
    output_path = 'graph2vec/features/'
    try: 
        os.mkdir(input_path)
        os.mkdir(output_path)
    except:
        print("Existe")
    for i, netw in enumerate(networks):
        with open((input_path+str(i)+".json"), "w") as f:
            edges = []
            v_names = netw.vs["name"]
            for edge in netw.get_edgelist():
                (u,v) = edge
                edges.append([int(v_names[u]), int(v_names[v])])
            json.dump({"edges":edges}, f)
    if operating_system == 'linux':
        #path_command = "python struc2vec/src/main.py --dimensions "+str(dimension)+" --input "+input_path+str(length_cut)+"_"+str(dimension)+"_"+str(i)+".edgelist --output "+output_path+str(length_cut)+"_"+str(dimension)+"_"+str(i)+".emb"
        path_command = 'python graph2vec/src/graph2vec.py --input-path ' + input_path + ' --output-path ' + output_path+str(length_cut)+'_'+str(dimension)+"_"+str(limiar_index)+'.csv' + ' --dimensions ' + str(dimension)
        os.system(path_command)
        print("graph2vec", str(i))
        
    all_network_features = [[] for _ in range(len(networks))]
    with open(output_path+str(length_cut)+"_"+str(dimension)+"_"+str(limiar_index)+".csv", "r") as f:
        lines = f.readlines()
        for l in lines[1:]:
            row = l.strip().split(",")
            id_graph = row[0]
            print("id graph", id_graph)
            network_features = np.array([float(v) for v in row[1:]])
            print(str(id_graph), "len",len(network_features))
            all_network_features[int(id_graph)] = network_features
    return np.array(all_network_features)
    
    
    
    

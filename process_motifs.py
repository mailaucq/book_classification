import os
import numpy as np

def get_motifs(operating_system, networks, length_cut, limiar_index):
    input_path = "motifs/dataset/"
    output_path = 'motifs/features/'
    prefix = str(length_cut) + "_" + str(limiar_index)  + "_"
    try: 
        os.mkdir(input_path)
        os.mkdir(output_path)
    except:
        print("Existe")
    input_path += prefix
    output_path += prefix 
    for i, netw in enumerate(networks):
        netw.write_pajek(input_path+str(i)+".paj")
    for i, netw in enumerate(networks):
        if operating_system == 'linux':
            path_command = 'python motifs/get_frequencies.py -r ' + input_path + str(i) + '.paj' + ' > ' + output_path+str(i)+'.emb'
            os.system(path_command)
            print("motifs", str(i))
    
    all_network_features = [[] for _ in range(len(networks))]
    for i in range(len(networks)):
        with open(output_path+str(i)+".emb", "r") as f:
            motifs_features = f.read().replace("\n","").replace("[","").replace("]","").split(",")
            motifs_features = [float(v) for v in motifs_features]
            print(motifs_features)
            
            network_features = np.array(motifs_features)
            print(str(i), "len",len(network_features))
            all_network_features[i] = network_features
    return np.array(all_network_features)

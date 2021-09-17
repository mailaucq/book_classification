import pandas as pd
import numpy as np
import igraph
import sys

#function that returns 1 when there is an arc leaving 
#the source and arriving at the target node
def isConnected(matrix, source, target):
	return matrix.iat[source,target]

#function that returns the frequency of all 13 directed subgraphs
def extract_subgraphs(network_file,is_absolute_output):
	#this code does not use subgraph[0], only indexes 1-13 to represent each subgraph
	subgraph = np.zeros(14) 
	g = igraph.read(network_file,format="pajek")
	g = g.simplify(combine_edges="sum")
	#get number of nodes
	size = g.vcount()
	matrix = pd.DataFrame(g.get_adjacency().data,index=range(size),columns=range(size))

	#loop in all nodes to count the subgraphs involving three different nodes A, B and C
	for A in range(0,size):
		#get neighbors of A (arcs leaving A)
		A_neighbors = np.where(matrix.iloc[A])[0]
		for B in A_neighbors:  
			if isConnected(matrix,B,A):
				#get neighbors of B (arcs leaving B)
				B_neighbors = np.where(matrix.iloc[B])[0].tolist()
				#A is a neighbor of B, but C has to be different from A
				B_neighbors.remove(A) 
				for C in B_neighbors: 
					if isConnected(matrix,C, B):
						if (isConnected(matrix,C,A) and isConnected(matrix,A,C)):
							subgraph[13] += 1
						elif (not isConnected(matrix,C,A) and isConnected(matrix,A,C)):
							subgraph[12] += 1
						elif (not isConnected(matrix,C,A) and not isConnected(matrix,A,C)):
							subgraph[8] += 1
					else:
						if (not isConnected(matrix,C,A) and not isConnected(matrix,A,C)): 
							subgraph[7] += 1
						elif (not isConnected(matrix,C,A) and isConnected(matrix,A,C)): 
							subgraph[11] += 1
			else:
				#get neighbors of B (arcs leaving B)
				B_neighbors = np.where(matrix.iloc[B])[0]
				for C in B_neighbors:
					if isConnected(matrix,C, B):
						if (not isConnected(matrix,C,A) and not isConnected(matrix,A,C)): 
							subgraph[3] += 1
						elif (not isConnected(matrix,C,A) and isConnected(matrix,A,C)): 
							subgraph[6] += 1
						elif (isConnected(matrix,C,A) and not isConnected(matrix,A,C)):  
							subgraph[10] += 1
					else:
						if (not isConnected(matrix,C,A) and not isConnected(matrix,A,C)): 
							subgraph[2] += 1 
						elif (not isConnected(matrix,C,A) and isConnected(matrix,A,C)): 
							subgraph[5] += 1
						elif (isConnected(matrix,C,A) and not isConnected(matrix,A,C)):  
							subgraph[9] += 1
				#get nodes whose arcs arrive at B	 	 
				incomingNodes = np.where(matrix[B])[0].tolist()
				#A has an arc arriving at B, but C has to be different from A 
				incomingNodes.remove(A)
				for C in incomingNodes: 
					if (not isConnected(matrix,B, C) and not isConnected(matrix,A,C) and not isConnected(matrix,C,A)): 
						subgraph[1] += 1  
				#get neighbors of A (arcs leaving A)
				other_A_neighbors = np.where(matrix.iloc[A])[0].tolist()
				#B is a neighbor of A, but C has to be different from B
				other_A_neighbors.remove(B)
				for C in other_A_neighbors:
					if (not isConnected(matrix,C,A) and not isConnected(matrix,B,C) and not isConnected(matrix,C,B)):  
						subgraph[4] += 1

	#some subgraphs are symmetric and, therefore, counted multiple times
	subgraph /= np.array([1,2,1,1,2,1,2,1,2,3,1,2,1,6])

	if is_absolute_output:
		return (subgraph[1:].astype("int")).tolist()
	else:
		return (subgraph[1:]/sum(subgraph)).tolist()


if __name__ == '__main__':

	if len(sys.argv) < 3: 
		sys.exit("Input format is wrong. Missing arguments.")

	option = sys.argv[1]
	if option == "-a": 
		is_absolute_output = True
	elif option == "-r": 
		is_absolute_output = False
	else: 
		sys.exit("Input format is wrong. Expecting -a or -r")

	network_file = sys.argv[2]
	values = extract_subgraphs(network_file,is_absolute_output)
	print(values)

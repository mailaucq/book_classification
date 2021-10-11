import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dimensions = range(21)
sizes = [300,900,1200,1500,1800,2100]
datasets = ["13authors"]#["stanisz", "brown", "vanessa"]
measures = ["dgr_n", "btw", "cc", "sp", "accs_h2", "accs_h3"]
measures_name = ["Average Neighbor Degree $<k_n>$", "Average Betweeness $<B>$", "Average Clustering Coeficient $<C>$", "Average Shortest Path Lenght $<l>$", r"Standard Desviation Shortest Path Lenght $\sigma{l}$", r"Average Accessibility $<\alpha^{h=2}$>", r"Average Accessibility $<\alpha^{h=3}$>"] #["Average Degree $<k_n>$", "Average PageRank $<PR>$", "Average Betweeness $<B>$", "Average Clustering Coeficient $<C>$", "Average Shortest Path Lenght $<l>$", "Average Symmetry Backbone $<Sb_{h=2}>$", "Average Symmetry Merge $<Sm_{h=2}>$", "Average Symmetry Backbone $<Sb_{h=3}>$", "Average Symmetry Merge $<Sm_{h=3}$>", r"Average Accessibility $<\alpha^{h=2}$>", r"Average Accessibility $<\alpha^{h=3}$>"]

for dataset in datasets:
	for measure, measure_name in zip(measures, measures_name):
		fig, ax = plt.subplots()
		barwidth = 0.2	
		for dim in range(0,21,5):
			means = []
			errors = []
			for size in sizes:
				series_ = []	
				df = pd.read_csv("results/"+dataset+"_"+str(size)+"_common_words_4_graph2vec.txtvariability")
				print(df.head(5))
				series_.append(df[df["i_percentage"]==dim][measure])
				df_ = pd.concat(series_, ignore_index=True)
				mean = df_.mean()
				std = df_.std()
				print(mean, std)
				means.append(mean)
				errors.append(std)

			materials = sizes
			r1 = np.arange(len(materials))
			x_pos = [x + barwidth for x in r1]
			ax.bar(x_pos, means, yerr=errors, width=0.1, align='center', alpha=0.75, ecolor='gray', capsize=10, label=str(dim) + "%")
			barwidth += 0.15
		ax.set_ylabel('Coefficient of Variation')
		#ax.set_xlabel(str(dim))
		ax.set_xticks(x_pos)
		ax.set_xticklabels(materials)
		ax.set_title(measure_name)
		ax.yaxis.grid(True)

		# Save the figure and show
		plt.legend()
		plt.tight_layout()
		plt.savefig("results/"+dataset+"_"+ measure + str(dim) +'.png')
		plt.show()
	

'''

for measure, measure_name in zip(measures, measures_name):
	
	
	for size in sizes:
		means = []
		errors = []
	
		for dim in dimensions:
			series_ = []	
			for dataset in datasets:
				df = pd.read_csv("results/"+dataset+"_"+str(size)+"_common_words_4.txtvariability")
				print(df.head(5))
				series_.append(df[df["i_percentage"]==dim][measure])
			df_ = pd.concat(series_, ignore_index=True)
			mean = df_.mean()
			std = df_.std()
			print(mean, std)
			means.append(mean)
			errors.append(std)

		x_pos = np.arange(len(dimensions))

		fig, ax = plt.subplots()
		ax.bar(x_pos, means, yerr=errors, align='center', alpha=0.5, ecolor='black', capsize=10)
		ax.set_ylabel('Coefficient of Variation')
		ax.set_xlabel(str(size))
		ax.set_xticks(x_pos)
		ax.set_xticklabels(dimensions)
		ax.set_title(measure_name)
		ax.yaxis.grid(True)

		# Save the figure and show
		plt.tight_layout()
		plt.savefig(measure + str(size) +'.png')
		plt.show()
		
	
'''	

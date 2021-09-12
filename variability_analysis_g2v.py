import pandas as pd
from manager2 import BookClassification
        

if __name__ == '__main__':
    feat_sel = 'common_words' # top_50  common_words
    iterations = 4    
    datasets = ["13authors"]#['brown', 'vanessa', 'stanisz']
    #sizes = [300, 600, 900, 1500, 1800, 2100]
    sizes = [100,200, 300, 600, 900, 1200, 1500, 1800, 2100]
    #sizes = [100, 200, 300, 400, 500, 600]
    for dataset in datasets:
    	for size in sizes:
    		obj = BookClassification(dataset=dataset, text_partition=size, feature_selection=feat_sel, sampling=iterations)
	    	obj.variability_analysis_g2v()


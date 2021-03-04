
from manager2 import BookClassification

dataset = 'stanisz' # 'vanessa' 'brown' 'stanisz'

partitions = [100, 500, 1000, 2000, 5000, 10000]
#partitions = [2000, 5000, 10000]
#partitions = [2000]
#iterations = [3]
#iterations = [10,10,5]
feat_sel = 'common_words' # top_50  common_words

#for index, (text_size, its) in enumerate(zip(partitions,iterations)):
for index, text_size in enumerate(partitions):
    print('Testing text size ' + str(index+1) + ' de ' + str(len(partitions)), text_size)
    obj = BookClassification(dataset=dataset, text_partition=text_size, feature_selection=feat_sel, sampling=1)
    obj.classification_analysis()
    print('----------------')
    print('\n\n')

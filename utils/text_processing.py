import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import random
import string
from collections import Counter
from nltk.stem import WordNetLemmatizer


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
random.seed(10)

def get_word_index(texts):
	word_index = {}
	index_word = {}
	count = 0
	for words in texts:
	    for word in list(set(words)):
	        if word not in word_index:
                    word_index[word] = str(count)
                    index_word[str(count)] = word
                    count += 1
	return word_index, index_word
    
def get_sequence(words, word_index):
	sequence = []
	for w in words:
	    if w in word_index:
	    	sequence.append(word_index[w])
	    else:
	    	continue
	return sequence

def get_sequences(texts, word_index):
	sequences = []
	for words in texts:	
	    sequences.append(get_sequence(words, word_index))
	return sequences

def get_stop_words(texts):
	commom_words = [w.lower() for w in texts[0]]
	for index, i in enumerate(texts):
	    text = [w.lower() for w in i]
	    commom_words = list(set(commom_words) & set(text))
	commom_words = [w for w in commom_words if w in stop_words]
	result = {word: index for index, word in enumerate(commom_words)}
	return result
    
def get_common_words(texts): 
	commom_words = [w.lower() for w in texts[0]]
	for index, i in enumerate(texts):
	    text = [w.lower() for w in i]
	    commom_words = list(set(commom_words) & set(text))
	result = {word: index for index, word in enumerate(commom_words)}
	return result

def get_top_words(texts, top_number="top_1"):
	if top_number.find('top_')!=-1:
	    top_words = int(top_number[top_number.rfind('_') + 1:])
	else:
	    top_words = top_number
	all_words = []
	for text in texts:
	    all_words.extend(list(set(text)))
	counts = Counter(all_words)
	features = counts.most_common(top_words)
	most_commom = dict()
	for index, feat in enumerate(features):
	    most_commom[feat[0]] = index
	return most_commom

def remove_puntuaction(text):
    text = text.translate(text.maketrans('', '', string.punctuation))
    return text
    
def pre_process_text(text, remove_stop_words=False, only_stop_words=False, remove_puntuaction_flag=False):
    if type(text) == list:
    	text = " ".join(text)
    	
    if remove_puntuaction_flag: # remove puntuation
    	text = remove_puntuaction(text)
    	
    tokens = word_tokenize(text) #tokenize
    
    if remove_stop_words:
        filtered_sentence = [w for w in tokens if not w.lower() in stop_words] # remove stopworws
    elif only_stop_words:
        filtered_sentence = [w for w in tokens if w.lower() in stop_words] # only stopworws
    else:
        filtered_sentence = tokens
    return filtered_sentence

def partition_text(texts, labels, length_cut, min_len_book, random_flag=1, step=0):
  step = length_cut if step <= 0 else step
  #df_par = pd.DataFrame(columns = df.columns)
  #texts = df["text"].copy()
  corpus = [word_tokenize(text) for text in texts]
  segmented_corpus = []
  new_texts = []
  new_labels = []
  partitions = int(round(min_len_book/step, 2) + 0.5)
  print("Max partitions: ", partitions)
  for book in corpus:
    segments = [book[int(i*step):int(i*step+length_cut)]+book[0:int(i*step+length_cut-min_len_book)] if i*step+length_cut>min_len_book else book[int(i*step):int(i*step+length_cut)] for i in range(partitions)]
    segmented_corpus.append(segments)

  for label, parts in zip(labels, segmented_corpus):
    if random_flag < partitions:
      for i in range(random_flag):
      	random_index = random.randint(0, len(parts) - 1)
      	new_texts.append(parts[random_index])
      	new_labels.append(label)
    else:
      for p in parts:
        new_texts.append(p)
        new_labels.append(label)
  return corpus, new_texts, new_labels

def get_process_corpus(selected, remove_punctuation=False, lemmatization_flag=False, mode_sequences=True, number_iterations=4, feature_selection = 'common_words'):
  if remove_punctuation:
    selected = [[w for w in sel if w not in string.punctuation] for sel in selected]
  if lemmatization_flag:
    selected = [[lemmatizer.lemmatize(w) for w in sel] for sel in selected]  
  word_index, index_word = get_word_index(selected)
  if feature_selection == 'common_words':
    words_features = get_common_words(selected)
  elif feature_selection == 'stop_words':
    words_features = get_stop_words(selected)
  else:
    words_features = get_top_words(selected, top_number=feature_selection)
  #if len(words_features) == 0:
  #  words_features = get_top_words(selected)
  if mode_sequences:
    selected = get_sequences(selected, word_index)
  return selected, words_features, word_index, index_word
  
def get_min_len_corpus(texts):
  corpus = [len(word_tokenize(text)) for text in texts]
  return min(corpus)
  
def get_corpus(texts, text_partition):
  corpus = [word_tokenize(text) for text in texts]
  #corpus = [i[:self.text_partition] for i in corpus]
  min_size = text_partition
  size_partitions = []
  segmented_corpus = [] # segment corpus
  for book in corpus:
    partitions = int(round(len(book)/min_size,2) + 0.5) #? por que mas 0.5?
    segments = [book[int(round(min_size * i)): int(round(min_size * (i + 1)))] for i in range(partitions)]
    size_partitions.append(len(segments))
    segmented_corpus.append(segments)
    #self.iterations = int(np.mean(size_partitions))
  return corpus, segmented_corpus

# get random partitions of book	
def get_random_corpus(segmented_corpus, remove_punctuation=False, lemmatization_flag=False, mode_sequences=True, number_iterations=4, feature_selection = 'common_words'):
  selected = []
  for partitions in segmented_corpus:
    if number_iterations == 1:
      random_index = 0
    else:
      random_index = random.randint(0, len(partitions) - 1)
    selected.append(partitions[random_index])
  if remove_punctuation:
    selected = [[w for w in sel if w not in string.punctuation] for sel in selected]
  if lemmatization_flag:
    selected = [[lemmatizer.lemmatize(w) for w in sel] for sel in selected]  
  word_index, index_word = get_word_index(selected)
  if feature_selection == 'common_words':
    words_features = get_common_words(selected)
  elif feature_selection == 'stop_words':
    words_features = get_stop_words(selected)
  else:
    words_features = get_top_words(selected, top_number=feature_selection)
  if len(words_features) == 0:
    words_features = get_top_words(selected)
  if mode_sequences:
    selected = get_sequences(selected, word_index)
  return selected, words_features, word_index, index_word
	


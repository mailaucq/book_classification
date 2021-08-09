import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import random
import string
from collections import Counter

stop_words = set(stopwords.words('english'))

def get_word_index(texts, tokenized=True):
	word_index = {}
	index_word = {}
	count = 0
	for text in texts:
	    words = text if tokenized else word_tokenize(text)
	    for word in list(set(words)):
	        if word not in word_index:
                    word_index[word] = str(count)
                    index_word[str(count)] = word
                    count += 1
	return word_index, index_word
    
def get_sequence(text, word_index):
	sequence = []
	for w in text:
	    if w in word_index:
	    	sequence.append(word_index[w])
	    else:
	    	continue
	return sequence

def get_sequences(texts, word_index):
	sequences = []
	for text in texts:
	    sequences.append(get_sequence(text, word_index))
	return sequences

def get_common_words(texts): 
	commom_words = texts[0]
	for index, i in enumerate(texts):
	    commom_words = list(set(commom_words) & set(i))
	result = {word: index for index, word in enumerate(commom_words)}
	return result

def get_top_words(top_number, texts, number=None):
	if top_number.find('top_')!=-1:
	    top_words = int(top_number[top_number.rfind('_') + 1:])
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

def remove_puntuaction(text):
    text = text.translate(text.maketrans('', '', string.punctuation))
    return text
    
def pre_process_text(text, tokenize=True, remove_stop_words=False, only_stop_words=False, remove_puntuaction_flag=False):
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

def partition_text(df_train, step, length_cut, min_len_book, random_flag):
  df_train_par = pd.DataFrame(columns = df_train.columns)
  corpus = df_train["text"].copy()
  segmented_corpus = []
   
  for book in corpus:
    partitions = int(round(min_len_book/step, 2) + 0.5)
    segments = [book[int(i*step):int(i*step+length_cut)]+book[0:int(i*step+length_cut-min_len_book)] if i*step+length_cut>min_len_book else book[int(i*step):int(i*step+length_cut)] for i in range(partitions)]
    segmented_corpus.append(segments)

  for (i, row), partitions in zip(df_train.iterrows(),segmented_corpus):
    if random_flag:
      random_index = random.randint(0, len(partitions) - 1)
      text = " ".join(partitions[random_index])
      row_tmp = row.copy()
      row_tmp["text"] = text
      df_train_par = df_train_par.append(row_tmp, ignore_index=True)
    else:
      for p in partitions:
        text = " ".join(p)
        row_tmp = row.copy()
        row_tmp["text"] = text
        df_train_par = df_train_par.append(row_tmp, ignore_index=True)
  df_train_par.tail()
  return df_train_par

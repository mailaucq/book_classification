import pandas as pd
import glob
import re
from nltk.tokenize import sent_tokenize


df = pd.DataFrame(columns=["label", "text","book"])
dir_books = "books_dataset_1/"
dataset_path = "datasetsv2/"
name_dataset =  "dataset_1.csv"
start_gutenberg = 0
end_gutenberg = 0

txt_files = glob.glob(dir_books + "*.txt")
for txt_file in txt_files:
	infile = open(txt_file,'r',encoding='utf8',errors="ignore")
	id_file = txt_file.replace(dir_books,"").replace(".txt","").split("_")
	authornames = re.findall('[A-Z][^A-Z]*', id_file[0])
	booknames = re.findall('[A-Z][^A-Z]*', id_file[1])
	textlines = infile.readlines()
	# Remove mention of project gutenberg
	for i, textline in enumerate(textlines):
		if "START OF THIS PROJECT GUTENBERG EBOOK" in textline or "START OF THE PROJECT GUTENBERG EBOOK" in textline:
			start_gutenberg = i
		if "END OF THIS PROJECT GUTENBERG EBOOK" in textline or "END OF THE PROJECT GUTENBERG EBOOK" in textline:
			end_gutenberg = i
			break
	print(start_gutenberg, end_gutenberg)
	if start_gutenberg != 0 and end_gutenberg != 0:
		text = "\n".join(textlines[start_gutenberg+1:end_gutenberg])
	else:
		text = infile.read()
	# Remove sentences where is mention the name of authors
	sentences = sent_tokenize(text)
	ans = [an.lower() for an in authornames]
	for an in ans:
		for sentence in sentences:
			if an in sentence.lower():
				print("$$$"+sentence+"$$$")
				text = text.replace(sentence, "")
	authorname = " ".join(authornames)
	bookname = " ".join(booknames)
	df = df.append({"label": authorname, "book":bookname, "text": text}, ignore_index=True)
	infile.close()
	print(authorname)
print(df.head())
print(df.shape)
print(df.label.value_counts())
df.to_csv(dataset_path + name_dataset, index=False)

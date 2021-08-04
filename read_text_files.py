import pandas as pd
import glob

dataset_path = "datasetv2/"
df = pd.DataFrame(columns=["label", "text"])
txt_files = glob.glob("books_dataset_2/*.txt")
for txt_file in txt_files:
	infile = open(txt_file,'r',encoding='utf8',errors="ignore")
	text = infile.read()
	authorname = txt_file.replace("books_dataset_2/","").replace(".txt","").split("_")[0]
	df = df.append({"label": authorname, "text": text}, ignore_index=True)
	infile.close()
	print(authorname)
df.to_csv(dataset_path + "dataset_2.csv", index=False)

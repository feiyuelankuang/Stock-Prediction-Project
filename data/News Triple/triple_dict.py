import csv
import re
from string import digits
filename = 'walmart_news_CNN'




def clean_triple(str):
	return re.sub(':','',str).strip()

def extract_triple(list):
	str = ''
	for element in list:
		str += element
	confidence = float(str[:4])
	#tuples = str[6:-2]
	#tuples.split
	triples = re.findall(r'[(](.*?)[)]', str)
	out_list = []
	for triple in triples:
		triple_list = []
		for str in triple.split(';'):
			#clean all the number
			new_str = str.translate(str.maketrans('', '', digits))
			new_str = re.sub('-|=|"|%','',new_str).strip()
			#clean data
			if new_str[:2] == 'L:':
				triple_list.append(new_str[2:])
			else:
				triple_list.append(new_str)
		if len(triple_list) > 2:
			#print(triple_list[2].strip())
			if triple_list[2][:2] != 'T:' and len(clean_triple(triple_list[2])) > 2 :
				out_list = [confidence]
				for entity in triple_list:
					out_list.append(clean_triple(entity))
				return out_list[:4]

#print(extract_triple(['0.93 (The best sales; to shop; 2021-04-06T15:12:57Z)']))


def select_triple(list_tuple, thres = 0.8):
	select_tuple = None
	best_score = 0
	for tuples in list_tuple:
		if tuples[0] > thres and tuples[0] > best_score:
			select_tuple = tuples
			best_score = tuples[0]
	return select_tuple


with open(filename + '_processed.csv', 'a') as csvfile:
    #writer.writerow('news','date','confidence','agent','predicate','object')
    writer = csv.writer(csvfile)
    with open(filename +'_result.csv', newline='') as csvfile2:
        spamreader = csv.reader(csvfile2, delimiter=',')
        for row in spamreader:
            if len(row) == 2 and row[1][:4].isdigit():
                news = row[0]
                date = row[1]
                #print(news,date)
                triple_list = []
            elif len(row) > 0:
                if extract_triple(row) is not None:
                    triple_list.append(extract_triple(row))
            elif len(row) == 0:
                selected_triple = select_triple(triple_list)
                if selected_triple is not None:
                    write_row = [news,date]
                    for item in selected_triple:
                        write_row.append(item)
                    writer.writerow(write_row)
                

import csv
import re
filename = 'walmart_news_CNN'


def extract_triple(str):
	confidence = float(str[:4])
	#tuples = str[6:-2]
	#tuples.split
	triples = re.findall(r'[(](.*?)[)]', str)[0].split(';')
	out_list = [confidence]
	for entity in triples:
		out_list.append(entity.strip())
	return out_list

def select_triple(list_tuple):
	select_tuple = None
	best_score = 0
	for tuples in list_tuple:
		if tuples[0] > 0.9 and tuples[0] > best_score:
			select_tuple = tuples
			best_score = tuples[0]
	return select_tuple


#print(extract_tuple('1.00 (Amazon; announces; $ 15 minimum wage)'),len(extract_tuple('1.00 (Amazon; announces; $ 15 minimum wage)')))

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
            elif len(row) == 1:
                triple_list.append(extract_triple(row[0]))
            elif len(row) == 0:
                selected_triple = select_triple(triple_list)
                if selected_triple is not None:
                    write_row = [news,date]
                    for item in selected_triple:
                        write_row.append(item)
                    writer.writerow(write_row)
                


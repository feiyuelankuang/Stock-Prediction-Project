import csv

filename = 'apple_reuters'


with open(filename +'_utf8.csv', 'a',encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    with open(filename+'.csv', newline='') as csvfile2:
        spamreader = csv.reader(csvfile2, delimiter=',')
        for row in spamreader:
            if len(row) > 0:
                writer.writerow([row[0],row[1]])
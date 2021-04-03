from newspaper import Article
import urllib.request,sys,time,json
from bs4 import BeautifulSoup
import requests
import pandas as pd
import csv
import io

def get_title_list(pagesToGet,url_init,filename):
    news_dict = []
    time_dict = []
    for page in range(1,pagesToGet+1):
        print('processing page :', page)
        if page > 1:
            url_now = url_init+'from='+ str((page-1)*10) +'&page=' + str(page)
        else:
            url_now = url_init

        print(url_now)
	    
        with urllib.request.urlopen(url_now) as url:
            json_str = url.read().decode()
            data = json.loads(json_str)




        for result in data['result']:
            #art_url = result['url']
            #article = Article(art_url)
            #article.download()
            #article.parse()
                       # print('title: ', result['headline'])
    #print('url: ', result['url'])
    #print('firstPublishDate: ', result['firstPublishDate'])
    #print('body: ', result['body'][:100]) 
            if result['headline'] is None:
            	continue
            news_dict.append(result['headline'].encode("utf-8"))
            #print(result['headline'].encode("utf-8").decode('utf-8', 'ignore'))
            time_dict.append(result['firstPublishDate'])
            #print(article.title)

    with io.open(filename, 'w',encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(news_dict)):
            writer.writerow([news_dict[i], time_dict[i]])


#url =  'https://search.api.cnn.io/content?size=10&q=%27apple%27%20%27tech%27&'
#filename="apple_news_CNN.csv"
#get_title_list(497,url,filename)


#https://edition.cnn.com/search?size=10&q=microsoft&from=10&page=2
url =  'https://search.api.cnn.io/content?size=10&q=microsoft&'
filename="micro_news_CNN.csv"
get_title_list(453,url,filename)

#        print(article.title,article.publish_date)
#article.parse()
 
#print('yes')
# print article text
#print(article.title)

    #an exception might be thrown, so the code should be in a try-except block
#    try:
        #use the browser to get the url. This is suspicious command that might blow up.
 #       page=requests.get(url)                             # this might throw an exception if something goes wrong.
        #print(page.text)
    
 #   except Exception as e:                                   # this describes what to do if an exception is thrown
#        error_type, error_obj, error_info = sys.exc_info()      # get the exception information
#        print ('ERROR FOR LINK:',url)                          #print the link that cause the problem
 #       print (error_type, 'Line:', error_info.tb_lineno)     #print error info and line that threw the exception
 #       continue                                              #ignore this page. Abandon this and go back.
 #   time.sleep(2)   
  #  soup=BeautifulSoup(page.text,'html.parser')
  #  frame=[]
  #  for a in soup.sel
    #links=soup.find_all("a",href=True)
  #  print(len(links))
#    filename="NEWS.csv"
#    f=open(filename,"w", encoding = 'utf-8')
#    headers="Statement,Link,Date, Source, Label\n"
#    f.write(headers)
    
 #   for j in links:
#        Statement = j.find("div",attrs={'class':'m-statement__quote'}).text.strip()
 #       Link = "https://www.politifact.com"
 #       Link += j.find("div",attrs={'class':'m-statement__quote'}).find('a')['href'].strip()
#        Date = j.find('div',attrs={'class':'m-statement__body'}).find('footer').text[-14:-1].strip()
#        Source = j.find('div', attrs={'class':'m-statement__meta'}).find('a').text.strip()
 #       Label = j.find('div', attrs ={'class':'m-statement__content'}).find('img',attrs={'class':'c-image__original'}).get('alt').strip()
  #      frame.append((Statement,Link,Date,Source,Label))
#        f.write(Statement.replace(",","^")+","+Link+","+Date.replace(",","^")+","+Source.replace(",","^")+","+Label.replace(",","^")+"\n")
#    upperframe.extend(frame)




#f.close()
#data=pd.DataFrame(upperframe, columns=['Statement','Link','Date','Source','Label'])
#data.head()



#url = "https://edition.cnn.com/2021/03/24/investing/stocks-value-growth/index.html"
 
# download and parse article
#article = Article(url)
#article.download()
#article.parse()
 
#print('yes')
# print article text
#print(article.title)

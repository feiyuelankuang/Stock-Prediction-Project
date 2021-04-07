from bs4 import BeautifulSoup
import csv
import requests
import re
from retry import retry
from selenium import webdriver
import time
from newspaper import Article
import csv
import io

if __name__ == '__main__':

    company = "apple"

    #url = "https://www.reuters.com/search/news?sortBy=date&dateRange=all&blob=apple"
    url = "https://www.reuters.com/search/news?blob=" + company
    #driver = webdriver.PhantomJS()
    driver = webdriver.Chrome(executable_path="C:\\Users\\fud13\\chromedriver.exe")
    driver.get(url)
    page_num = 0


    try:
        while driver.find_elements_by_css_selector('.search-result-more-txt') and page_num < 80000:
            driver.find_element_by_css_selector('.search-result-more-txt').click()
            page_num += 1
            print("getting page number "+str(page_num))
            #print(driver.page_source.encode('utf-8'))

    except:
        print('search end')
        #time.sleep(1)

    #print(driver.current_url)
    
    html = driver.page_source.encode('utf-8')

    soup = BeautifulSoup(html, 'lxml')
    links = soup.find_all('div', attrs={"class":'search-result-indiv'})
    articles = [a.find('a')['href'] for a in links if a != '']
    news_dict=[]
    time_dict=[]
    print(len(articles))
    for link in articles:
        url = 'https://www.reuters.com'+ link
        #print(url)
        try:
            article = Article(url)
            article.download()
            article.parse()
            if article.title.encode("utf-8") not in news_dict:
                news_dict.append(article.title)#.encode("utf-8"))
                time_dict.append(article.publish_date)

        except:
            pass

    #print(news_dict,time_dict)

    filename = company+ '_reuters.csv'
    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(news_dict)):
            writer.writerow([news_dict[i], time_dict[i]])

    #print(articles)


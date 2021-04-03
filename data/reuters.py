from bs4 import BeautifulSoup
import csv
import requests
import re
from retry import retry
from selenium import webdriver
import time
from newspaper import Article
import csv
# get the link to each news from the home page
def get_linklists(soup):
    # create a list to contain the unselected links
    link_raw = []
    # create a list to contain the selected links
    link_lists = []
    # get all <a> tags
    for k in soup.find_all('a'):
        # get all href in the <a> tags
        link = k.get('href')
        # if the link contains '/article', this is a link to an article
        if '/article' in link:
            # append all links to articles to the unselected list
            link_raw.append(link)
            print(link)
    # because each news has picture links and content links, the links will be repeated, and the duplicate links need to be removed
    for i in range(len(link_raw) - 1):
        # remove the repeated links
        if link_raw[i] != link_raw[i + 1]:
            # append the selected links to link_list
            link_lists.append(link_raw[i])
    return link_lists

# the network is not stable, so use retry() to execute this function again after failure
#@retry()
# get the content of each news
def get_newscontent(link_lists):
    # create a list to contain news of each page
    news_contents = []
    # traverse the links of news to get the content of each news
    for link in link_lists:
        date_html = requests.get('https://www.reuters.com' + link, headers=headers, verify=False)
        subsoup = BeautifulSoup(date_html.text,'lxml')
        # news content is in <p> tag
        contents = subsoup.select('p')
        # change parameter type for the convenience of further operation
        str_contents = "".join([str(x) for x in contents])
        # get news content from <p> tag with regular expression
        contents_text = re.findall('<p>(.*?)</p>', str_contents, re.S)
        # change parameter type for the convenience of further operation
        str_contents_text = "".join([str(x) for x in contents_text])
        # remove the links inside news with regular expression
        selected_text = re.sub(r'<.*?>', "", str_contents_text, count=0, flags=0)
        # append the processed news content to the list of news content
        news_contents.append(selected_text)
    return news_contents


# write the data into CSV
def write_csv(id, time, title, content):
    # write the data of ten news of each page to CSV
    for i in range(len(id)):
        csv_writer.writerow([id[i], time[i].text, title[i].text, content[i]])


if __name__ == '__main__':

    url = "https://www.reuters.com/search/news?sortBy=date&dateRange=all&blob=apple"
    driver = webdriver.PhantomJS()
    driver.get(url)
    html = driver.page_source.encode('utf-8')
    page_num = 0


    try:
        while driver.find_elements_by_css_selector('.search-result-more-txt') and page_num < 10000 :
            driver.find_element_by_css_selector('.search-result-more-txt').click()
            page_num += 1
            print("getting page number "+str(page_num))
            #print(driver.page_source.encode('utf-8'))

    except:
        print('search end')
        #time.sleep(1)

    html = driver.page_source.encode('utf-8')
    soup = BeautifulSoup(html, 'lxml')
    links = soup.find_all('div', attrs={"class":'search-result-indiv'})
    articles = [a.find('a')['href'] for a in links if a != '']
    news_dict=[]
    time_dict=[]
    for link in articles:
        url = 'https://www.reuters.com'+ link
        #print(url)
        article = Article(url)
        article.download()
        article.parse()
        news_dict.append(article.title)
        time_dict.append(article.publish_date)

    #print(news_dict,time_dict)

    filename = 'apple_reuters.csv'
    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(news_dict)):
            writer.writerow([news_dict[i], time_dict[i]])

    #print(articles)


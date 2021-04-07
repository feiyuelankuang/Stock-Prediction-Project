from bs4 import BeautifulSoup
import csv
import requests
import re
from retry import retry
from newspaper import Article

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
    # because each news has picture links and content links, the links will be repeated, and the duplicate links need to be removed
    for i in range(len(link_raw) - 1):
        # remove the repeated links
        if link_raw[i] != link_raw[i + 1]:
            # append the selected links to link_list
            link_lists.append(link_raw[i])
    return link_lists

# the network is not stable, so use retry() to execute this function again after failure
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
def write_csv(id, time, title):
    # write the data of ten news of each page to CSV
    for i in range(10):
        #print(title[i].text.strip())
        csv_writer.writerow([id[i], time[i].span.text.strip(), title[i].text.strip()])


if __name__ == '__main__':
    headers = {
        'Accept': '*/*',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36',
        'Referer': "https://item.jd.com/100000177760.html#comment"}
    # create a CSV to contain data of news
    ff = open('Apple_Reuters_news.csv', 'a+', newline='', encoding='utf-8')
    csv_writer = csv.writer(ff)
    # add header
    csv_writer.writerow(['Id', 'Date', 'Title'])
    # get news data in the given page range
    for page in range(1,2001):
        requests.packages.urllib3.disable_warnings()
        url = 'https://www.reuters.com/news/archive/apple?view=page&page={}&pageSize=10'.format(str(page))
        #url = 'https://www.reuters.com/news/archive/businessnews?view=page&page={}&pageSize=10'.format(str(page))
        r = requests.get(url, headers=headers, verify=False)
        soup = BeautifulSoup(r.content, 'lxml')
        # get title of each news
        title = soup.find_all('h3', class_='story-title')
        # get release time of each news
        time = soup.find_all('time')
        # get 10 links to news of each page
        link_lists = get_linklists(soup)
        # create a list to contain the id of each news
        id_list = []
        # get news id
        for link in link_lists:
            #print(link)
            id_list.append(link)
        # get news content
       # news_lists = get_newscontent(link_lists)
        # write news data to CSV
        write_csv(id_list, time, title)
        print("Page {} is finished".format(page))
ff.close()

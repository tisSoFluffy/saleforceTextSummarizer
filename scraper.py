# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 19:56:38 2022

@author: joshu
"""

import requests
from bs4 import BeautifulSoup
import re
import time
import pandas as pd
import os

URL = "https://www.salesforce.com/solutions/small-business-solutions/keep-customers/"
URL = "https://www.salesforce.com/blog/customer-experience/?sfdc-redirect=282"
URL = "https://www.salesforce.com/content/dam/web/en_us/www/documents/datasheets/digital-engagement-datasheet.pdf"
URL = "https://www.salesforce.com/products/service-cloud/features/service-agent-console/"
URL = "https://www.salesforce.com/products/commerce-cloud/resources/ecommerce-optimization-tips-for-boosting-conversions/"

#read in the urls from Paritosh's spreadsheet
urls = pd.read_csv('urls.csv')

#Iterate over each URL, removing the PDFS. They break Beautifulsoup
soups = []
#TODO fix apply method to series
#urls['url'].apply(lambda x: get_soup(x))
no_pdf_urls = urls[~urls['url'].str.contains('pdf')]['url']


def get_soup(url):
    '''Given a url string, append the url and text of url to the list '''    
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    txt = soup.get_text()
    soups.append([url,txt])

for url in no_pdf_urls:    
    get_soup(url)


df_soups = pd.DataFrame(soups)

#Save to corpus
counter = 0
while os.path.exists(f'corpus_{counter}.csv'):
    counter+=1

df_soups.to_csv(f'corpus_{counter}.csv')
    

a = soup.find_all('a', {'href': re.compile(r'/products/')})


links = []
for link in a:
    links.append(f'https://www.salesforce.com{link.get("href")}')
    
soups = []
for link in links:
    page = requests.get(link)
    soup = BeautifulSoup(page.content,'html.parser')
    txt = soup.get_text()
    soups.append([link,txt])
    time.sleep(5)

for soup in soups:
    name = soup[0][27:] + '.txt'
    name = name.replace(r'/','.')
    with open(name,'w', encoding='utf-8') as f:
        f.write(soup[1])

def saveSoup(soup, name):
    with open(name,'w', encoding='utf-8') as f:
        f.write(soup.get_text())

saveSoup(soup, name = 'digital-engagement-datasheet.txt')

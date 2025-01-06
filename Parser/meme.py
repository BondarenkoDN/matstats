import requests      # Библиотека для отправки запросов
import numpy as np   # Библиотека для матриц, векторов и линала
import pandas as pd  # Библиотека для табличек 
import time          # Библиотека для времени
from fake_useragent import UserAgent
from bs4 import BeautifulSoup


page_link = 'https://opt-opt-opt.ru/'
#print(requests.get(page_link))
## через fake_useragent используя случайно сгенерированные сочетания ОС, версии браузера и тд
response = requests.get(page_link, headers = {'User-Agent': UserAgent().chrome})
#print(response)
html_page = response.content
#print(html_page[:1000])
html = BeautifulSoup(html_page, 'html.parser')
#print(html.html.head.title)
## ссылки с одной страницы
obj = [i.find('a', href = True) for i in html.find_all('div', attrs = {'class':'bx_catalog_item_container'})]
hrefs = [ 'https://opt-opt-opt.ru' + j.get('href') for j in obj]
#print(hrefs[:10])

## соберем ссылки по всем страницам начиная со второй
def getAllLinks(page_number):
    link ='https://opt-opt-opt.ru/?PAGEN_1={}'.format(page_number)
    response = requests.get(link, headers={'User-Agent': UserAgent().chrome})
    
    if not response.ok:
        return [] 
    
    html_page = response.content
    html = BeautifulSoup(html_page, 'html.parser')
    hrefs = [ 'https://opt-opt-opt.ru' + j.get('href') for j in [i.find('a', href = True) for i in html.find_all('div', attrs = {'class':'bx_catalog_item_container'})]]
    
    return hrefs

#print(getAllLinks(2))

## собираем все ссылки
# for i in range(2,65):
#     hrefs.append(getAllLinks(i))


## извлечем данные по позиции
def getStats(html):
    try:
        obj = [i.text.strip() for i in html.find('div', attrs={'class': "bx_catalog_item_articul"}) if i != '\n' and len(i)]
        obj = dict(zip(obj[::2],obj[1::2]))
        obj["Цена"]=html.find ('div', attrs={'class': "bx_catalog_item_price"}).text
    except:
        obj=None
    
    return obj

stats = getStats(html)
print("Артикул: {}\n Торговая марка: {}\n Размеры: {}\n Цена: {}".format(stats["Артикул"], stats["Торговая марка"], stats["Размеры:"], stats["Цена"]))

## описание по всем товарам
# for link in hrefs:
#     stats = getStats(link)
#print("Артикул: {}\n Торговая марка: {}\n Размеры: {}\n Цена: {}".format(stats["Артикул"], stats["Торговая марка"], stats["Размеры:"], stats["Цена"]))
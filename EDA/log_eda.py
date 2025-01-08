import numpy as np         # библиотека для матриц и математики
import pandas as pd        # библиотека для работы с табличками
from scipy import stats    # модуль для работы со статистикой
import matplotlib.pyplot as plt
from IPython.display import display
import seaborn as sns
from datetime import datetime
import squarify
plt.style.use('fivethirtyeight')

df = pd.read_csv('google_log.csv', sep='\t')

'''Описание колонок:
date дата посещения сайта (рассматривается период с 20160801 по 20170801
fullVisitorId уникальный id пользователя
sessionId уникальный id одной пользовательской сессии
channelGrouping откуда произошёл переход
visitStartTime timestamp начала визита
device.browser браузер визита
device.operatingSystem операционная система устройства
device.isMobile является ли устройство мобильным
device.deviceCategory тип устройства (айпад, компьютер, мобильный телефон)
geoNetwork.subContinent часть света пользователя
geoNetwork.country страна пользователя
geoNetwork.region регион пользователя
geoNetwork.city город пользователя
totals.hits похоже что это действия на сайте, но это неточно
totals.pageviews просмотры страниц
totals.transactionRevenue выручка с покупки
trafficSource.source источник трафика
trafficSource.medium более высокоуровневый источник трафика
trafficSource.keyword ключевые слова из поиска
trafficSource.adwordsClickInfo.adNetworkType несколько переменных с дополнительной информацией из adwords
trafficSource.adwordsClickInfo.page
trafficSource.adwordsClickInfo.slot
trafficSource.adwordsClickInfo.isVideoAd
trafficSource.adContent'''


df['totals.transactionRevenue'] = df['totals.transactionRevenue'] /10 ** 6
print(df.head())
print(100*df.isnull().sum()/df.shape[0])

fig, ax = plt.subplots(figsize = (6,6))
heatmap = sns.heatmap(df.sample(1000).isnull(), yticklabels=False, 
                    cbar=False, 
                    cmap='viridis'
                      )
#plt.show()
df['date'] =  pd.to_datetime(df['date'], format = "%Y%m%d")
df['month'] = df['date'].apply(lambda w: w.strftime('%Y-%m'))
df['day'] = df['date'].dt.day
df["month"] = df['date'].dt.month 
df["weekday"] = df['date'].dt.weekday     
df["is_month_start"] = df['date'].dt.is_month_start
df["is_month_end"] = df['date'].dt.is_month_end
df['hour'] = (df['visitStartTime']
              .apply(lambda x: datetime.fromtimestamp(x).hour))
df['t'] = (df['visitStartTime']
           .apply(lambda x: datetime.fromtimestamp(x)))


## число уникальных пользователей за день
df_day = (df.groupby('date')
          .agg({'fullVisitorId':'nunique'})
          .sort_values('date'))
df_day.plot(figsize = (12,6))
#plt.show()
print(f" Среднее число посетителей за день :{df_day.fullVisitorId.mean()}")

##число уникальных пользователей за месяц
df_month = (df.groupby('month')
          .agg({'fullVisitorId':'nunique'})
          .sort_values('month'))
df_month.plot(figsize = (12,6))
#plt.show()
print(f" Среднее число посетителей за день :{df_month.fullVisitorId.mean()}")

##Сколько пользовательских сессий в день
df_session = (df.groupby('date')
          .agg({'fullVisitorId':'count'})
          .sort_values('date'))
df_session.plot(figsize = (12,6))
#plt.show()
print(f" Среднее число сессий за день :{df_session.fullVisitorId.mean()}")

##Как часто люди возвращаются
first_month_session = (df.groupby('fullVisitorId')
                       .agg({
                           'month': 'min',
                           'date' : 'min',
                           't' : 'min'
                       }))
first_month_session.columns = ['first_invate_month', 'first_invate_day', 'first_invate_ts']
df = df.join(first_month_session, on = 'fullVisitorId')
big_month = df[['fullVisitorId', 'month', 'first_invate_month']]
## возврат пользователей на ресурс
comeback = big_month.pivot_table(
    index = 'first_invate_month',
    columns = 'month',
    values = 'fullVisitorId',
    aggfunc='nunique')
sns.heatmap(comeback, annot = True, fmt = '.0f')
#plt.show()

print('Средний процент покупок ', 100 * (1-df['totals.transactionRevenue'].isnull().mean()))
#работаем только с данными про покупки
df_buy = df.dropna(subset = ['totals.transactionRevenue'])
fig, axes = plt.subplots(1, 2, figsize = (5,5))
df_buy['totals.transactionRevenue'].hist(bins = 50, log = True, ax=axes[0])
sns.histplot(np.log(df_buy['totals.transactionRevenue'] + 1),
             ax = axes[1],
             bins = 100,
             kde = True)
#plt.show()

first_day_buy = (df_buy.groupby('fullVisitorId')
                 .agg({
                    'month': 'min',
                    'date' : 'min',
                    't' : 'min'
                 })
)
first_day_buy.columns = ['first_buy_month', 'first_buy_day', 'first_buy_ts']
df_buy = df_buy.join(first_day_buy, on = 'fullVisitorId')

df_buy['delta_t'] =((df_buy['first_buy_ts'] - df_buy['first_invate_ts'])
                    /np.timedelta64(1,'D'))
df_buy['delta_t'].hist(bins = 30)
plt.xlabel('Время между входом на сайт и покупкой')
#plt.show()

df_buy['delta_t'] = ((df_buy['first_buy_ts'] - df_buy['first_invate_ts'])
                           /np.timedelta64(1,'h'))
df_buy['delta_t'].hist(bins=30)
plt.xlabel('Время между входом на сайт и покупкой в часах');
#plt.show()
print( '60 процентный квантиль', df_buy['delta_t'].quantile(0.6), '. Если он больше 0.5, значит первую покупку посетители совершают чаще всего в первый визит')
## распределение покупок по дням и неделям
df_buy.groupby('date')['fullVisitorId'].count().plot(figsize=(6,6))
plt.show()
df_days_hours = (df_buy.pivot_table(index = 'hour', columns = 'weekday',
                         values = 'totals.transactionRevenue',
                         aggfunc = 'sum').style.background_gradient())
sales = df_buy.pivot_table( index = 'first_buy_month', columns = 'month',
                           values ='fullVisitorId', aggfunc = 'nunique')
sns.heatmap(sales, annot = True, linecolor='black', cmap="YlGnBu")
#plt.show()
## средний дневной доход
df_av_day = (df_buy.groupby('date')
             .agg({'totals.transactionRevenue' : 'mean'})
             .plot(figsize = (6,6)))
## суммарно по когортам 
sales_sum = df_buy.pivot_table(index='first_buy_month', columns='month',
             values = 'totals.transactionRevenue', aggfunc='sum')
sns.heatmap(sales_sum/sales, annot = True, linecolor='black', cmap="YlGnBu")
#plt.show()
##  выручка в зависимости от источника
df_buy['totals.transactionRevenue'] = (df_buy['totals.transactionRevenue']
                                       .apply(lambda w: np.log(w + 1)))
## выборка по самым используемым браузерам
df_buy['device.browser'].value_counts().plot(kind = 'bar')
plt.title("TOP 10 самых используемых браузеров", fontsize=20)
plt.xlabel("Браузер", fontsize=16)
plt.ylabel("Число визитов", fontsize=16);
##  выручка с разбивкой по браузерам
most_browsers = (df_buy[df_buy['device.browser']
                        .isin(df_buy['device.browser']
                        .value_counts()[:10].index.values)]
)
plt.figure(figsize=(6,6))
sns.boxplot(x = 'device.browser', y = 'totals.transactionRevenue', data = most_browsers)


## распределение по операционным системам
df_buy['device.operatingSystem'].value_counts().plot(kind = 'bar')
plt.title("Операционная система", fontsize=20)
plt.xlabel("Операционная система", fontsize=16)
plt.ylabel("Число визитов", fontsize=16)
plt.show()
## прибыль в зависимости от операционной сиситемы
most_OS = (df_buy[df_buy['device.operatingSystem']
                        .isin(df_buy['device.operatingSystem']
                        .value_counts()[:6].index.values)])
(sns.FacetGrid(most_OS, hue = 'device.operatingSystem', aspect = 2)
                .map(sns.kdeplot, 'totals.transactionRevenue', shade = True)
                .add_legend())

## по разным типам устройств
sns.countplot(df_buy["device.deviceCategory"], palette = 'hls')
plt.show()
most_device = sns.boxplot(x = "device.deviceCategory", y = "totals.transactionRevenue",
            data = df_buy)
most_device.set_title('Выручка в разбивке по типам устройств', fontsize=20)
most_device.set_xlabel('Устройство', fontsize=18)
most_device.set_ylabel('Распределение выручки', fontsize=18)
most_device = (sns.FacetGrid(df_buy, hue = 'device.deviceCategory',aspect = 2)
               .map(sns.kdeplot, 'totals.transactionRevenue', shade = True)
               .add_legend())
df_geo = df_buy[df_buy["geoNetwork.city"] != 'not available in demo dataset']
cities = df_geo["geoNetwork.city"].value_counts()
cities = round((cities[:30] / len(df_geo["geoNetwork.city"]) * 100), 2)
cities_plot = squarify.plot(sizes = cities.values,
                label = cities.index, value = cities.values, alpha = .3)
plt.show()



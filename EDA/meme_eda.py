import numpy as np
import pandas as pd
import scipy.stats as sts
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
plt.style.use('ggplot')  # стиль для графиков

memes = pd.read_csv('MEMES_new_short.csv', index_col = 0)
'''
name            object
status          object
type            object
origin_year     object
origin_place    object
date_added      object
views            int64
videos           int64
photos           int64
comments         int64
tags            object
about           object
origin          object
other_text      object
'''

print(memes.head())
print(memes.info())# сводная информация - сколько строк, столбцов, типы данных для каждого столбца и колво ненулевых значений
print(memes.describe())# описательная статистика
print(memes.describe(include = 'object'))# описательная статистика по типу объекта
print(memes[memes.views == memes.views.max()])
## создаем колонку с временной меткой
memes['Timestamp'] = pd.to_datetime(memes.date_added, utc = True)
t = memes.Timestamp.loc[0]
print(t.timestamp, t.year, t.month, t.day, t.hour)
## добавим эти столбцы в таблицу
memes['year'] = memes.Timestamp.apply(lambda x: x.year)
memes['month'] = memes.Timestamp.apply(lambda x: x.month)
memes['day'] = memes.Timestamp.apply(lambda x: x.day)
memes['hour'] = memes.Timestamp.apply(lambda x: x.hour)
memes['weekday'] = memes.Timestamp.dt.dayofweek
print(memes.head())
memes['days_from_creation'] = ((memes.Timestamp.max() -  memes.Timestamp)
                               /np.timedelta64(1, 'D'))
#memes[['month', 'day', 'hour', 'weekday']].hist()


index = 0
fig, ax = plt.subplots(2, 2, figsize=(10,5))
bins = [12, 31, 24, 7]
for i in range(2):
    for j in range(2):
        (ax[i, j].hist(memes[['month', 'day', 'hour', 'weekday'][index]],
         bins = bins[index]))
        (ax[i, j]. set_title('Hist of {}'
        .format(['month', 'day', 'hour', 'weekday'][index])))
        index += 1
fig.subplots_adjust(hspace = 0.2)
plt.show()
memes.year.value_counts().sort_index().plot( kind='bar')
plt.title("Hist of year")
plt.show()

# преобразуем тип переменной год создания мема в число и избавимся от обозначения Unknown
memes.loc[memes.origin_year == 'Unknown', 'origin_year'] = None
usl = memes.origin_year.apply(lambda x: str(x).isdigit())
memes.loc[~usl,'origin_year'] = 0
memes.origin_year = memes.origin_year.astype(int)
ancient = memes[ (memes.origin_year < 1500) & (memes.origin_year != 0) ]
for i in ancient.index.tolist():
    print("Name: {}".format(ancient.loc[i, "name"]))
    print("Year: {}".format(ancient.loc[i, "origin_year"]))
    print("About:\n{}".format(ancient.loc[i, "about"]))
    print("==================================================================\n")


memes.views.hist(bins=25, log=True)# тыжелые хвосты и выбросы
plt.show()
memes['av_views']=memes.views/(memes.days_from_creation +1)
print(memes['av_views'].max())
memes['av_comm']=memes.comments/(memes.days_from_creation +1)

pop_mem=memes.sort_values(by = 'av_comm', ascending =False)
for i in pop_mem.index.tolist()[:10]:
    print('Meme name: ', pop_mem.loc[i,'name'] ,'\n',
         "Average comments {}".format(round(pop_mem.loc[i, 'av_comm'])))

## категориальные переменные
print(memes[['status', 'type', 'origin_place']].describe())

memes.status.value_counts().plot(kind = 'barh')
plt.title("Hist of status")
plt.show()

pd.get_dummies(memes['status'], drop_first = True).head()
cnt = memes.type.value_counts()
big_category = set(cnt[cnt >= 30].index)
memes.type = memes.type.apply(lambda w: w if w in big_category else 'another')
memes.type.value_counts().plot(kind='barh');
plt.show()


memes.origin_place.value_counts()
cnt = memes.origin_place.value_counts()
big_category = set(cnt[cnt >= 30].index)
memes['origin_place'] = (memes.origin_place
    .apply(lambda w: w if w in big_category else 'another'))
memes.origin_place.value_counts().plot(kind='barh');
plt.show()

#оценим длину тэгов
memes['tags_len'] = memes.tags.str.len()
memes.tags_len.hist(bins=25)
plt.show()

#заполним пропуски нулями
memes.fillna(0, inplace = True)
#memes.to_csv('memes_prepare.csv', sep='\t')

sns.boxplot(np.log(memes.av_views), memes.status)
plt.show()
memes.boxplot(column =np.log(memes.av_views),
               by = 'hour', figsize = (10,6))
plt.show()

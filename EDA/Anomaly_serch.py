import numpy as np
import pandas as pd

import scipy.stats as sts
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')  # стиль для графиков

df = pd.read_csv('memes_prepare.csv' ,sep = '\t')

'''Описание колонок:

name - название мема
views - число просмотров на сайте
comments - число комментариев
photos - число вкладышей с мемом
days_from_creation - сколько дней прошло от появления мема
average_views - среднее число просмотров за день
average_comments - среднее число комментариев за день
tags_len - длина тегов в описании в числе символов'''


#df.set_index('name', inplace = True)
#print(df.info())
df = df[['views','photos', 'comments', 'days_from_creation', 'av_views', 'av_comm', 'tags_len' ]]
print(df.head())

for var in ['views', 'photos', 'av_views', 'av_comm', 'comments']:
    df[var] = df[var].apply(lambda w: np.log(w+1))

df.hist(figsize=(20, 8))

def isOutlier(data, col, threshold = 3):
    mean = data[col].mean()
    std = data[col].std()
    up_bound = mean + threshold * std
    low_bound = mean - threshold * std
    anom = pd.concat([data[col] > up_bound, data[col] < low_bound], axis=1).any(axis=1)
    return anom, up_bound, low_bound

a,l,r = isOutlier(df, df.columns)

def getColumnOutlier(data, function = isOutlier, threshold = 3):
    outlier = (pd.Series(data=[False]*len(data),
                          index=data.index, name='is_outlier'))# оздаем массив заполняем его предварительно false имя колонки is_outlier
    # табличка для статистики по каждой колонке
    comparison_table = {}
    for col in data.columns:
        anom, upper, lower = function(data, col, threshold=threshold)
        comparison_table[col] = ([upper, lower, sum(anom), 
                                  100*sum(anom)/len(anom)])
        outlier.loc[anom[anom].index] = True# во все колонки outlier записать тру где была найдена аномалия

    comparison_table = pd.DataFrame(comparison_table).T
    comparison_table.columns = ['upper', 'lower','anom_count', 'anom_percent' ]
    return comparison_table, outlier
        
comparison_table, std_outliers = getColumnOutlier(df)
print(comparison_table)
# в табличке std_outliers будут пометки для каждого мема является ли он аномалией

def anom_report(outlier):
    print("Total number of outliers: {}\nPercentage of outliers: {:.2f}%"
           .format(sum(outlier), 100*sum(outlier)/len(outlier)))

anom_report(std_outliers)
anom_df=df.copy()
anom_df['is_outlier'] = std_outliers
sns.pairplot(data=anom_df, vars=df.columns,
             hue='is_outlier', hue_order=[1, 0],
             markers=['X', 'o'],  palette='bright')
plt.show()

def get_iqr(data, col, threshold = 3):
    IQR = data[col].quantile(0.75) - data[col].quantile(0.25)
    up_bound = data[col].quantile(0.75) + IQR * threshold
    low_bound = data[col].quantile(0.75) - IQR * threshold
    anom = (pd.concat([data[col] > up_bound, 
                       data[col] < low_bound], axis=1).any(axis=1))
    return anom, up_bound, low_bound

comparison_table, iqr_outlier = getColumnOutlier(df, function = get_iqr)
anom_report(iqr_outlier)

anom_df['is_outlier'] = iqr_outlier

sns.pairplot(data=anom_df, vars = df.columns,
             hue='is_outlier', hue_order=[1, 0],
             markers=['X', 'o'],  palette='bright');
plt.show()
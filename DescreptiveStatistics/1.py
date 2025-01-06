import numpy as np         # библиотека для матриц и математики
import pandas as pd        # библиотека для работы с табличками
from scipy import stats    # модуль для работы со статистикой
# библиотеки для визуализации
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import plotly.express as px
plt.style.use('ggplot')    # стиль графиков

#df = sns.load_dataset('tips')
#print(df.head())
#df['procent_tip']=100*df.tip / df.total_bill
#print(df.head())


#sns.pairplot(df,hue = 'sex', height = 2, kind = 'scatter')# как между собой ваимосвязаны счёт, чаевые и пол клиента


#sns.catplot(data=df,x="day",y= "total_bill", hue = 'sex', kind="box")#разбиение по дням недели


#sns.jointplot(data = df, x = "total_bill", y = "tip", kind = 'reg' )#  взаимосвязь только двух переменных.

#df.corr(method='pearson')# ковариционные матрицы
#df.corr(method='spearman')

#sns.heatmap(df, square=True, annot = True, cmap="RdBu")# визуализация матриц с помощью тепловой карты
#plt.show()

# print(df.isnull().sum())# проверка на пропуски в данных 

# df = df[df['колонка где есть пропуски'].notnull()]# очистка от пропусков 

# df = df[df['total_bill']>0]# очистка ошибки в данных 


ddf=pd.read_excel("Online Retail.xlsx")
#print(ddf.head())
ddf = ddf[ddf['CustomerID'].notnull()]
ddf.groupby('Country')['CustomerID'].agg('nunique').sort_values(ascending = False)[:10].plot(kind = 'bar')
#plt.show()
uk_data = ddf[ddf.Country == 'United Kingdom']
uk_data = uk_data[uk_data['Quantity']>0]

uk_data['TotalPrice'] = uk_data['Quantity'] * uk_data['UnitPrice']# добавим поле с итоговой суммой при покупке
uk_data['InvoiceDate'] = pd.to_datetime(uk_data['InvoiceDate'])
'''Будем сегментировать клиентов по RFM.
Recency (Свежесть) - число дней насколько давно пользователь что-то у нас покупал
Frequency (Частота) - насколько много заказов пользовать сделал
Monetary (Деньги) - сколько в сумме денег потратил '''

rfm = (uk_data.groupby('CustomerID').
        agg({'InvoiceDate': lambda date : (dt.datetime(2011,12,10) - date.max()).days,
             'InvoiceNo' : 'count',
             'TotalPrice': sum
        })
)
#print(rfm.head())
rfm[['InvoiceDate','InvoiceNo','TotalPrice']].hist(bins=30)
plt.show()
rfm[['InvoiceDate','InvoiceNo','TotalPrice']].apply(lambda w: np.log(w+1)).hist(bins=30)
plt.show()
rfm['InvD_quant']=pd.qcut(rfm['InvoiceDate'], 4, [1,2,3,4])
rfm['InvNo_quant']=pd.qcut(rfm['InvoiceNo'], 4, [4,3,2,1])
rfm['Tot_quant']=pd.qcut(rfm['TotalPrice'], 4, [4,3,2,1])
print(rfm.head())
##найдем самых преданных покупателей:
rfm['RFM_Score'] = rfm.InvD_quant.astype(str) + rfm.InvNo_quant.astype(str) + rfm.Tot_quant.astype(str)
print(rfm[rfm['RFM_Score'] == '111'].sort_values('TotalPrice', ascending=False).head())
fig=px.scatter_3d(
        rfm,
        x = 'InvoiceDate',
        y = 'InvoiceNo',
        z = 'TotalPrice',
        color = 'RFM_Score'
)
fig.show()

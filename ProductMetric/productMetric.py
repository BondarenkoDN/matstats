import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')  # ����� ��� ��������


visits = pd.read_csv('marketing_log.csv', sep = '\t')
visits['date'] = pd.to_datetime(visits["date"], format = "%Y-%m-%d")
visits.sort_values('date', inplace = True)#����������� �� ����
visits.reset_index(drop = True, inplace = True)# ������� ������ �������
day = visits.groupby('date').agg({'fullVisitorId': 'nunique'})# ������� ����� ���������� � ����
day.sort_index().plot(figsize = (10, 5))
plt.show()


first_visit = visits.groupby('fullVisitorId').agg({'date':'min'})
first_visit.columns = ['firstVisit']
visits = visits.join(first_visit, on = 'fullVisitorId')
visits['daysFromFirstVisit'] = ((visits.date - visits.firstVisit)
                                .apply(lambda w: w.days))
visits['daysFromFirstVisit'].hist(bins = 50)#������������� ����� ���� ����� ������� � ������ ����������
plt.show()


#����������� ��� �����, ������� ��������� ����� 20 ����
visits[visits['daysFromFirstVisit'] > 20]['daysFromFirstVisit'].hist(bins = 50)
plt.show()
# �������� ����� ������ � ��������� ����������
firstVisits = visits.groupby(['firstVisit', 'fullVisitorId']
                    .agg({'daysFromFirstVisit': 'max'}))


# �������������� �� 7 ����:
retention = (firstVisits.groupby('firstVisit')['daysFromFirstVisit']
                        .agg([("success", lambda w: sum(w >= 7)),
                        ("total", "count")]))
# ������ ������� retencion, ��� ������ ��������� �������� � ������
retention['retention'] = retention['success'] / retention['total']
retention['retention'][30 : 120].plot(figsize = (10, 5))
plt.show()


# ��� ��� �������������� ��� ����, �� ��� ��� ����� ��������� ������������� �������� � ������� ���
alpha = 0.05
retention['se'] = np.sqrt(retention['retention'] * (1 - retention['retention']) / retention['total'])
q = stats.norm.ppf(1- alpha / 2)
retencion['left'] = retention['retention'] - q * retention['se']
retencion['right'] = retention['retention'] + q * retention['se']
df = retention[30 : 120]
df['retention'].plot(figsize = (10, 5))
plt.fill_between(df.index, df['left'], df['right'], facecolot = 'blue', alpha = 0.2, interpolate = True)
plt.show()


# ������� ������� � ������������

# ���� �������� � % �������:
print("% ������� ����� ���� ��������� �������, ������������ ���������� ���������",
100 * (1 - visits['transactionRevenue'].isnull().sum() / visits.shape[0]))
#������� ����� � ��������:
visits['month'] = visits['date'].apply(lambda w: w.strftime('%Y-%m'))
purchase = visits.dropna(subset = ['transactionRevenue'])# ������� ������ ��� ��������� �������� � ������� transactionRevenue

q99 = purchase['transactionRevenue'].quantile(0.99) # ������ ������� ���� 0,99 ��������
purchases = purchases[purchases['transactionRevenue'] < q99]
purchases['transactionRevenue'].hist(bins = 50)
plt.show()

# ��������� �������, ����������� ����������, � ����� ����������
datePurchase = (visits.groupby(['month'])['transactionRevenue']
                      .agg([('rpu', 'mean'),
                            ('count', 'count'), 
                            ('se', 'std')])
                      .reset_index())
datePurchase['left'] = datePurchase['rpu'] - q * datePurchase['se'] / np.sqrt(datePurchase['count'])
datePurchase['right'] = datePurchase['rpu'] + q * datePurchase['se'] / np.sqrt(datePurchase['count'])
datePurchase['rpu'].plot(figsize = (10, 5))
plt.fill_between(datePurchase['month'], datePurchase['left'], datePurchase['right'],
facecolor = 'blue', alpha = 0.2, interpolate = True)
plt.show()
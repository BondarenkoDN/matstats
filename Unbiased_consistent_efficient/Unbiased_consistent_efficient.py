import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')


'''������ ���������� ���������:
-������������� (��� ������������� ������� ������� �� � ������� �� ���������)
-��������������� (��� ����������� ���������� ������� �� �������� � ��������� ��������)
-������������� (� ����� ������ ����� ��������� ��������� ��� � ������ (��������, ��� ���� ����������� ������))
-������ ��� ��������� ��������� �� �����-�� ���������� ��������������� ������������ ������.'''

norm_rv = stats.norm(loc = 300, scale = 100)
rvs = norm_rv.rvs(size = 10 ** 4)
print("��������� ������� �� �������", np.mean(rvs))

n = 100
# ���������� ��� ������ �������� ��������
x1 = np.zeros(200)
x2 = np.zeros(200)
x3 = np.zeros(200)
for i in range(200):
  rvs_sample = np.random.choice(rvs, size = n, replace = False)
  rvs_mean = np.mean(rvs_sample)
  x1[i] = rvs_mean 
  x2[i] = rvs_mean - 4200 / n
  x3[i] = rvs_mean -5 * (n + 1) / n
print(np.mean(x1), np.mean(x2), np.mean(x3))

plt.figure(figsize = (12, 6))
df = pd.DataFrame(zip(x1, x2, x3), columns = ['x1', 'x2', 'x3'])
sns.boxplot(data = df)
''' ������������� ��� �������� ������ ��� ������������� ������� ������� n. 
��� ��������, ��� ������ "� �������", �� ���� ��� ��������������� ������������� ������, �����������. 
��� �������� ������� ��������, �� �� ������������. 
����������, ����� �������� ������ ����������� � ������ ������ �������. 
����� ������ ���������� �������������� �����������.'''


''' ������ ���� � ������� ��������� ���� ���������� �������������, 
���� ���� � ������� ��������� � ���� �� ����������� ��� ����� ����� ���������'''
rvs = norm_rv.rvs(size = 10 ** 6)
theta_real = np.mean(rvs)
theta_hat = np.zeros((100, 50))
for i in range(100, 10100, 100):
  rvs_sample = np.random.choice(rvs, size = (n, 50), replace = False)
  rvs_mean = np.mean(rvs_sample, axis = 0)
  theta_hat[n//100 - 1] = rvs_mean # �������������
  #theta_hat[n // 100 - 1] = (rvs_mean - 4200 / n)  # �������������
  #theta_hat[n // 100 - 1] = (rvs_mean - 5 * (n + 1) / n) # ���������������

plt.figure(figsize=(14,8))
plt.plot(theta_hat, c='grey', alpha = 0.3)
plt.xlabel('���������� ����������', size = 24)
plt.xlabel('������ ���������', size = 24)
#plt.hlines(theta_real, 0, 100, color = 'blue', lw = 4, label = '�������� ����')
#plt.legend(fontsize = 20)
plt.show()


'''������ ���� � ������� ��������� ���� ���������� ����������� � ��������� ������ ������, 
���� � ��������� � ���� ������ ������ ��� �������� ������ ������� ����������
����� ���� ������������� x1..xn~iid U[0; theta]
�� ������ �������� ���� : Theta = 2*x_mean, 
�� ������ ������������� �������������: Theta = x_mean * (n+2)/n
��� ������ ������������� � �����������. ������� ����� �����������
'''

unoform_rv = stats.uniform(0, 5)
n_obs = 100
theta_1 = 2 * np.mean(x, axis = 0)
theta_2 = (n_obs + 2) / n_obs * np.max(x, axis = 0)
plt.figure(figsize = (12, 6))
plt.hist(theta_1, bins = 100, alpha = 0.5, label = 'Moment method')
plt.hist(theta_2, bins = 100, alpha = 0.5, label = 'ML method')
plt.legend()
plt.show()
'''������� � ����� ������ ������� ������, �� � ����� �����������)'''
print(np.var(theta_1), np.var(theta 2))

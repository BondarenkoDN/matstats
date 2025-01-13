
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as sts
from tqdm.notebook import tqdm
plt.style.use('ggplot')  # стиль для графиков



'''Нормальное распределение'''
plt.figure(figsize = (5,5))
x = np.linspace(-8, 8, 100)
par = [(0,1), (0,2), (0,3), (2,3), (2,1)]

for mu, sigma in par:
    rv = sts.norm(mu, sigma)
    pdf = rv.pdf(x)
    plt.plot(x, pdf, label = "$\mu={},\sigma={}$".format(mu, sigma))
plt.xlabel('$x$', fontsize = 20)
plt.ylabel(r'Probability', fontsize = 20)
plt.legend(fontsize = 20)
plt.show()

'''Распределение кси квадрат, выборка из 1000, 5 степеней свободы 
а)'''
norm_rv = sts.norm(loc = 0, scale = 1)
x = norm_rv.rvs(size = (5,1000))
y = (x ** 2).sum(axis = 0)
plt.hist(y,bins = 50)

'''б)'''
x = np.linspace(0,30,100)
for k in [1, 2, 3, 4, 6, 9]:
    rv = sts.chi2(k)
    pdf = rv.pdf(x)
    plt.plot(x, pdf, label="$k=%s$" % k)
plt.legend()
plt.show()

'''Распределение Стьюдента, выборка из 1000, 4 степени свободы'''
'''а)'''
# norm_rv = sts.norm(loc = 0, scale = 1)
# x0 = sts.norm.rvs(1000)
x = norm_rv.rvs(size = (5,1000))
x = (x ** 2).sum(axis = 0)
y = x / np.sqrt(x / 5)
plt.hist(y,bins = 50)

'''б)'''
x = np.linspace(-5, 5, 100)
for k in [1, 2, 3, 4, 6, 9]:
    rv = sts.t(k)
    pdf = rv.pdf(x)
    plt.plot(x, pdf, label="$k=%s$" % k)
plt.legend()
plt.show()

''' Распределение Фишера, выборка из 1000, 10 и 5 степени свободы'''
'''a)'''
#norm_rv = sts.norm(loc = 0, scale = 1)
x1 = norm_rv.rvs(size = (5,1000))
x2 = norm_rv.rvs(size = (10,1000))
x1 = (x1 ** 2).sum(axis = 0)
x2 = (x2 ** 2).sum(axis = 0)
y = (x1 / 5) / (x2 / 10)
plt.hist(y,bins = 50)

'''б)'''


x = np.linspace(0, 5, 100)
par = [(3,1), (5, 1), (40, 1), (20, 3), (20, 5), (20, 10)]

for k, m in par:
    rv = sts.f(k, m)
    pdf = rv.pdf(x)
    plt.plot(x, pdf, label = "$k={}, m={}".format(k, m))
plt.xlabel('$x$', fontsize = 20)
plt.ylabel(r'Probability', fontsize = 20)
plt.legend(fontsize = 20)
plt.show()


x = np.linspace(-5, 5, 100)
plt.plot(x, sts.t(1).pdf(x), label = "$t(1)$")
plt.plot(x, sts.norm(0, 1).pdf(x), label="$N(0,1)$")
plt.show()



'''Слабая форма ЗБЧ на примере игральной кости'''

n_obs = 100000
eps = 0.01
x = np.random.choice(np.arange(1, 7), size = n_obs)
x_cumsum = np.cumsum(x)
x_mean = x_cumsum / np.arange(1,n_obs +1)
plt.figure(figsize = (10, 5))
plt.plot(x_mean[100:])
plt.axhline(3.5 + eps, color='g', linestyle='dashed', linewidth=2)
plt.axhline(3.5 - eps, color='g', linestyle='dashed', linewidth=2)
plt.xlabel('Число подбрасываний игральной кости')
plt.ylabel('Среднее значение')
plt.show()

n_obs = 10 ** 4
x = np.random.choice(np.arange(1, 7), size = (n_obs, 1000))
x_cumsum = np.cumsum(x, axis = 0)
x_mean = x_cumsum / np.arange(1,n_obs + 1)[: ,None]

eps1 = 0.1
eps2 = 0.01
bad_event1 = np.abs(x_mean - 3.5) > eps1
bad_event2 = np.abs(x_mean - 3.5) > eps2

bad_mean1 = np.mean(bad_event1, axis = 1)
bad_mean2 = np.mean(bad_event2, axis = 1)

plt.figure(figsize = (12,6))
plt.plot(bad_mean1, label = f"Вероятность для коридора {eps1}")
plt.plot(bad_mean2, label = f"Вероятность для коридора {eps2}")
plt.xlabel('Число подбрасываний игральной кости')
plt.ylabel('Иллюстрация сходимости по вероятности')
plt.legend(fontsize=14)
plt.show()


'''Расходимость по вероятности на примере распределения Коши'''

Cauchy = sts.cauchy()
x_с = Cauchy.rvs(size = (n_obs, 1000))
x_cumsum_с= np.cumsum(x_с, axis = 0)
x_mean_с = x_cumsum_с / np.arange(1, n_obs + 1)[: ,None]
plt.figure(figsize=(12,6))
plt.plot(x_mean_с[:,442])
plt.axhline(0, color='b', linestyle='dashed', linewidth=2)
plt.xlabel('Число подбрасываний игральной кости')
plt.ylabel('Среднее значение');
plt.show()

eps1 = 0.1
eps2 = 0.01
bad_event1_с = np.abs(x_mean_с - 3.5) > eps1
bad_event2_с = np.abs(x_mean_с - 3.5) > eps2

bad_mean1_с = np.mean(bad_event1_с, axis = 1)
bad_mean2_с = np.mean(bad_event2_с, axis = 1)
plt.figure(figsize=(12,6))
plt.plot(bad_mean1_с)
plt.plot(bad_mean2_с)
plt.xlabel('Число подбрасываний игральной кости')
plt.ylabel('Вероятность пробить коридор');
plt.show()



'''Генарация распределений 

Из равномерного случайного распределения получить экспоненциальное распредление
функция распределения экспоненциальной случайной величины принимает вид 1-exp(-ах)
Обратная ей функция (-1/а)ln(1-x)

'''
uniform_rv=sts.uniform(0, 1)
n_obs = 10 ** 6
y = uniform_rv.rvs(n_obs)
plt.hist(y)
plt.show()
x = - 0.5 * np.log(1 - y)
plt.hist(x, density = True, bins = 50)
plt.show()


exp_rv = sts.expon(scale = 0.5)
x = exp_rv.rvs(n_obs)
y = 1 - np.exp(-2 * x)
plt.hist(y, density = True, bins = 50)
plt.show()

'''
ЗБЧ разрешает оценить матожидание случайных величин без взятия интегралов
'''
n_obs = 10 ** 6
'''The location (loc) keyword specifies the mean. 
The scale (scale) keyword specifies the standard deviation'''
norm_rv = sts.norm(loc = 5, scale = 3)
x = norm_rv.rvs(n_obs)
print(np.mean(1/x))

'''оценим множественные вероятности 
величина х1+х2+х3**2 > 5'''
uniform_rv = sts.uniform(0, 2)
x_1 = uniform_rv.rvs(n_obs)
x_2 = uniform_rv.rvs(n_obs)
x_3 = uniform_rv.rvs(n_obs)
print(np.sum((x_1 + x_2 + x_3 ** 2) > 5) / n_obs)

''' оценим условную вероятность х1+х2+х3 > 0.8, 
при условии х3<0.1'''

uslovie = x_3 < 0.1
success = x_1[uslovie] + x_2[uslovie] + x_3[uslovie] > 0.8
print(np.sum(success) / np.sum(uslovie))

''' Длина французкого багета 1 метр. За один укус можно сьесть кусок 
распределенный в отрезке [0.1]'''
n_obs = 10 ** 4
uniform_rv = sts.uniform(0, 1)
def bite_baguete():
  length = 1
  bites = 0
  while length > 0:
    length -= uniform_rv.rvs(1)
    bites += 1
  return (bites)
Number_of_bites = [bite_baguete() for i in range(n_obs)]
print(np.mean(Number_of_bites))

''' Проверим теорию: в произвольной группе из 50
человек вероятность того, что хотябы у двоих людей дни рождения 
совпадут, равна 0.97
'''

df = pd.read_csv("data/vk_bdate.tsv", sep = "\t",
                dtype = {'byear' : pd.Int64Dtype(),
                          'bmonth' : pd.Int64Dtype(),
                          'bday' : pd.Int64Dtype()}
                )
print(df.head())
df = df[~(df.bmonth.isnull() | df.bday.isnull())] # удаление строк с пропусками
df['bdate'] = df['bday'].astype(str) + '-' + df['bmonth'].astype(str)

plt.figure(figsize = (10,5))
df.bmonth.value_counts().sort_index().plot(kind = 'bar')
plt.title(' Колво дней рождений в зависимости от месяца')
plt.xlabel('Номер месяца')
plt.ylabel('Колво дней рождений в месяце')



n = 10 ** 4
m = 0
for i in tqdm(range(n)):
  m += df.bdate.sample(50).unique().size < 50
print("вероятность того, что хотябы у двоих людей дни рождения совпадут", m / n)



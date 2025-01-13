'''Центральная предельная теорема'''

import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt

n_obs = 10 ** 6
# в uniform задается левая граница и длина
uni_rv = sts.uniform(-1, 2)
x_1 = uni_rv.rvs(n_obs)
x_2 = uni_rv.rvs(n_obs)
x_3 = uni_rv.rvs(n_obs)
x_4 = uni_rv.rvs(n_obs)
plt.hist(x_1, bins=100) # равномерное распределение
plt.show()
plt.hist(x_1 + x_2, bins=100) # треугольное распределение
plt.show()
plt.hist(x_1 + x_2 + x_3 + x_4, bins=100, density=True)
x = np.linspace(-4, 4, 100)
pdf = sts.norm().pdf(x)
plt.plot(x, pdf)
plt.show()

'''Сходимость по распределению на примере распредлеения Стьюдента'''
fig, ax = plt.subplots(1, 2, figsize = (10,5))
for k in [1, 5, 10, 50]:
    rv = sts.t(df = k)
    pdf = rv.pdf(x)
    cdf = rv.cdf(x)
    ax[0].plot(x, pdf,label = "$t(%s)$" % k, lw = 1.2)
    ax[1].plot(x, cdf,label = "$t(%s)$" % k, lw = 1.2)

rv_schodim = sts.norm()
pdf_limit = rv_schodim.pdf(x)
cdf_limit = rv_schodim.cdf(x)
ax[0].plot(x, pdf_limit, label = 'N(0,1)', linestyle = 'dashed', lw = 2)
ax[0].set_ylim(-0.03, 0.45)
ax[0].set_title("Плотность распределения (PDF)")
ax[0].legend()


ax[1].set_ylim(-0.1, 1.1)
ax[1].plot(x, cdf_limit, label = 'N(0,1)', linestyle = 'dashed', lw = 2)
ax[1].set_title("Функция распределения (CDF)")
ax[1].legend()
plt.show()





''' 50 точек на отрезке от 0 до 100. 
Как они распределены по одномерному отрезку'''
x = np.random.choice(range(0, 100), size = 50, replace = False)
y = np.zeros_like(x)

plt.figure(figsize = (10, 2))
plt.scatter(x, y)
for grid in [20, 40, 60, 80]:
  plt.axvline(x = grid, color = '#D8D8D8')
plt.xlim((0, 100))
plt.xlabel("Размерность 1", fontsize = 14)
plt.ylabel("")
plt.yticks([], [])
plt.title("1D", fontsize = 20)
plt.show()

''' 50 точек на отрезке от 0 до 100. 
Как они распределены по двухмерной плоскости'''
x = np.random.choice(range(0, 100), size = 50, replace = False)
y = np.random.choice(range(0, 100), size = 50, replace = False)

plt.figure(figsize = (8,8))
plt.scatter(x,y)
for grid in [20, 40, 60, 80]:
  plt.axvline(x = grid, color = '#D8D8D8')
  plt.axhline(y = grid, color = '#D8D8D8')
plt.xlim((0, 100))
plt.ylim((0, 100))
plt.xlabel("Размерность 1", fontsize = 14)
plt.ylabel("Размерность 2", fontsize = 14)
plt.title("2D", fontsize = 20)
plt.show()

''' 50 точек на отрезке от 0 до 100. 
Как они распределены в пространстве'''
x = np.random.choice(range(0, 100), size = 50, replace = False)
y = np.random.choice(range(0, 100), size = 50, replace = False)
z = np.random.choice(range(0, 100), size = 50, replace = False)

fig = plt.figure(figsize = (10,6))
ax = fig.add_subplot(1, 1, 1, projection = '3d')
ax.scatter(x, y, z)

for grid in [20, 40, 60, 80]:
  plt.axvline(x = grid, color = '#D8D8D8')
  plt.axhline(y = grid, color = '#D8D8D8')

ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_zlim(0, 100)
ax.set_xlabel("Размерность 1", fontsize = 14)
ax.set_ylabel("Размерность 2", fontsize = 14)
ax.set_zlabel("Размерность 3", fontsize = 14)
plt.title("3D", fontsize = 20)
plt.show()
'''С увеличением размерности число симуляций, 
необходимых для устойчивых выводов экспоненциально увеличивается'''


n_obs = 10 ** 6
x = np.random.uniform(0, 1, size = n_obs)
y = np.random.uniform(0, 1, size = n_obs)
plt.figure(figsize = (7, 7))
p = np.linspace(0, 1, 1000)
plt.plot(p, np.sqrt(1 - p ** 2), color = 'black')
usl = x ** 2 + y ** 2 > 1
plt.scatter(x[usl], y[usl])
plt.scatter(x[~usl], y[~usl])
plt.show()
print("Вычисленное через площадь круга pi = ", 4 * np.mean(x ** 2 + y ** 2 <= 1))

x = np.random.uniform(0, 1, size = n_obs)
y = np.random.uniform(0, 1, size = n_obs)
z = np.random.uniform(0, 1, size = n_obs)
print("Вычисленное через объем шара pi = ", 8 * (3 / 4) * np.sum(x ** 2 + y ** 2 + z ** 2 <= 1) / n_obs)


'''Вычисление через d-мерный шар'''
def findPi(n_dims, n_obs = 10 ** 6):
  x = [np.random.uniform(0, 1, size = n_obs) for _ in range(n_dims)]
  distance = np.zeros(n_obs)
  for item in x:
    distance += item ** 2
  V = np.sum(distance <= 1) / n_obs
  pi = (2 ** n_dims * V * gamma((n_dims / 2) + 1)) ** (2 / n_dims)
  return pi
print(findPi(n_dims=5))

plt.figure(figsize = (7, 5))

n_simulations = [10 ** i for i in range(2, 7)]
dimention = [5,10,15,20]
for d in dimention:
  current_result = []
  for n in n_simulations:
    pi = []
    for _ in range(20):
      pi.append(findPi(d, n))
    current_result.append(np.mean(pi))
  plt.plot(range(2, 7), current_result, label = f'dimention = {d}', lw =2)

plt.axhline(y = np.pi, color = 'black', linestyle = '--')
plt.xticks(np.arange(2, len(n_simulations) + 2), n_simulations)
plt.legend(loc = 'lower left')
plt.show()    
    
'''Для более высоких размерностей требуется 
на порядок больше точек чтобы получить число pi'''

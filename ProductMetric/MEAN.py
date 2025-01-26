
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
plt.style.use('ggplot')  # стиль для графиков


'''Построим функцию для квантиля уровня 2,5%,
то есть 95% асимптотически нормальный доверительный интервал'''
def buildDoveritInterval(alpha, mean_hat = 0, std_hat = 1, margin = 5):
  plt.figure(figsize = (10, 5))
  xs = np.linspace(mean_hat - margin, mean_hat +margin)
  pdf = stats.norm(mean_hat, std_hat).pdf(xs)
  plt.plot(xs, pdf)
  plt.ylabel('$f(x)$', fontsize = 18)
  plt.xlabel('$x$', fontsize = 18)
  left,right = stats.norm.interval(1 - alpha, loc = mean_hat, scale = std_hat)
  for i in [left, right]:
    y_max = plt.ylim()[1]
    plt.axvline(i, color = 'red', linestyle = 'dashed', lw = 2)
    if i == left:
      xq = np.linspace(mean_hat - margin, left)
    else:
      xq = np.linspace(right, mean_hat + margin)
    
    text_margin = 0.05
    plt.text(i + text_margin, 0.8 * y_max, round(i, 2), color = 'blue', fontsize = 14)
    yq = stats.norm(mean_hat, std_hat).pdf(xq)
    plt.fill_between(xq, 0, yq, color = 'blue', alpha = 0.3)
  plt.show()
  return left, right

'''Пусть есть распределение Пуассона. Необходимо построить оценку интенсивности
методом моментов, и доверительный интервал для нее
Распределение имеет вид:'''

x = [5, 7, 8, 2, 3, 1, 2]
norm_rv = stats.norm(loc = 0, scale = 1)
alpha = 0.05 # уровень значимости
z_alpha = norm_rv.ppf(1 - alpha / 2)# квантиль уровня альфа
lambda_hat = np.mean(x)
lambda_se = np.sqrt(lambda_hat / len(x))
lambda_left = lambda_hat - z_alpha * lambda_se
lambda_right = lambda_hat + z_alpha * lambda_se
print(f"Доверительный интервал [{lambda_left:.3}, {lambda_right:.3}] ширины {lambda_right - lambda_left:.3}")

'''То же самое через встроенную функцию'''
print("Доверительный интервал", stats.norm.interval(0.95, loc = lambda_hat, scale = lambda_se))
buildDoveritInterval(alpha, mean_hat = lambda_hat, std_hat = lambda_se)


'''Доверительный интервал для разности'''
x = [5, 7, 8, 2, 3, 1, 2] 
y = [1, 1, 9, 1, 2, 2, 2]
diff = np.mean(x) - np.mean(y)
diff_se = np.sqrt(np.mean(x) / len(x) + np.mean(y) / len(y))
left = diff - z_alpha * diff_se
right = diff + z_alpha * diff_se
print(f"Доверительный интервал [{left:.3}, {right:.3}] ширины {right - left:.3}")
buildDoveritInterval(alpha, mean_hat = diff, std_hat = diff_se)
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random
style.use('ggplot')

def best_fit_slope_and_intercept(xs, ys):
    m_gradient = (((mean(xs) * mean(ys)) - mean(xs * ys)) /
                  ((mean(xs) ** 2) - mean(xs ** 2)))
    c = mean(ys) - m_gradient * mean(xs)
    return m_gradient, c


def squared_error(ys_original, ys_line):
    return sum((ys_line - ys_original) ** 2)


def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr / squared_error_y_mean)


def create_dataset(numberOfDatapoints, variance, step=2, correlation=False):
    val = 1
    ys = []
    for i in range(numberOfDatapoints):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step
    xs = [i for i in range(len(ys))]
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


xs, ys = create_dataset(40, 40, 2, correlation='pos')
#xs = np.array([1, 2, 3, 4, 5], dtype=np.float64)
#ys = np.array([1, 2, 4, 4, 5], dtype=np.float64)

m, b = best_fit_slope_and_intercept(xs, ys)
print(m, b)
regression_line = []
# regression_line = [(m*x + b for x in xs)] # sub in each x into the equation ans store as list
for x in xs:
    regression_line.append(m * x + b)
r_squared = coefficient_of_determination(ys, regression_line)
print("R squared " + str(r_squared))
predict_x = 7
predict_y = (m * predict_x) + b
print(predict_y)
plt.scatter(xs, ys, color='#003F72', label='data')
plt.scatter(predict_x, predict_y)
plt.plot(xs, regression_line, label='regression line')
plt.legend(loc=4)
plt.show()

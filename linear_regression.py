import matplotlib.pyplot as plt
import numpy as np

salary = np.array([20, 30, 40, 50, 60])
eaten_res = np.array([5, 10, 15, 20, 25])

plt.xlabel('Every month income')
plt.ylabel('Going to Restaurant')

plt.title("Income vs Expense")
plt.plot(salary, eaten_res)
plt.show()

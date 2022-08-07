import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


# fpr, tpr
naive_bayes = np.array([0.28, 0.63])
logistic = np.array([0.77, 0.77])
random_forest = np.array([0.88, 0.24])
ann = np.array([0.12, 0.76])

# plotting
plt.scatter(naive_bayes[0], naive_bayes[1], label='Naive Bayes', facecolors='black', edgecolors='orange', s=300)
plt.scatter(logistic[0], logistic[1], label='Logistic Regression', facecolors='orange', edgecolors='orange', s=300)
plt.scatter(random_forest[0], random_forest[1], label='Random Forest', facecolors='blue', edgecolors='black', s=300)
plt.scatter(ann[0], ann[1], label='Artificial Neural Network', facecolors='red', edgecolors='black', s=300)

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc='lower center')

plt.show()

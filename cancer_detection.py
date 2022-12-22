import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data = load_breast_cancer()

x = data['data'] #numerical values only
y = data['target'] #malignant(0) or benign(1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

clf = KNeighborsClassifier()
clf.fit(x_train, y_train)

#print(clf.score(x_test, y_test)) #scores accuracy of model

x_new = np.array(random.sample(range(0,50), len(data['feature_names'])))
#print(data['target_names'][clf.predict([x_new])][0])

#visulaization of data:

column_data = np.concatenate([data['data'], data['target'][:, None]], axis = 1)
column_names = np.concatenate([data['feature_names'], ['class']])

df = pd. DataFrame(column_data, columns = column_names)
print(df.corr())

sns.heatmap(df.corr(), cmap = 'coolwarm', annot = True, annot_kws = {'fontsize': 8})
plt.tight_layout()
plt.show()
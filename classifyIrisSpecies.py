import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
import seaborn as sns
from IPython.display import display
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

#iris_dataset is a Bunch object which contains keys and values like a dictionary
iris_dataset = load_iris()

# Step 1
# print(f"Key's of iris_dataset:\n", iris_dataset.keys())
#
# #The OReilly book presents the max return to 193 characters for this print statement.
# #However, it is import to get as much information as we have to better understand our data and our problem
# #print(iris_dataset['DESCR'][:] + "\n...")
#
# # The value of target_names is an array of strings that contain the species of flowers we want to predict.
# print('Target names: ', iris_dataset['target_names'])
#
# #Frame is none
# print('Frame: \n', iris_dataset['frame'])

#target, feature_names, filename, data_module
# print('Target: ', iris_dataset['target'])
# print('Feature names: ', iris_dataset['feature_names'])
# print('File Name: ', iris_dataset['filename'])
# print('Data Module: ', iris_dataset['data_module'])

#Step 2:
# print('Type of data: ', type(iris_dataset['data']))
# print('Shape of data: ', iris_dataset['data'].shape)
# print('First five rows of data: \n', iris_dataset['data'][:5])
#
# print('Type of Target: ', type(iris_dataset['target']))
# print('Shape of Target: ', iris_dataset['target'].shape)
# print('Target: \n', iris_dataset['target'])

X_train, X_test, y_train,y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)
# print('X - train set shape: ', X_train.shape)
# print('y - train set shape: ', y_train.shape)
#
# print('X- test shape: ', X_test.shape)
# print('y- test shape: ', y_test.shape)

# Create dataframe from data in X_train
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
iris_dataframe['species']= y_train

#Create a scatter matrix from the dataframe, color by y_train
#pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15,15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)


# Create a pairplot with seaborn
g = sns.pairplot(iris_dataframe, hue='species', diag_kind='kde', palette='Set2')


#plt.savefig('scatterplot_seaborn.png')
#plt.show()
#plt.show()

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)

#X_new = np.array([[5,2.9,1,0.2]])
#print("X_new.shape:", X_new.shape)

# prediction = knn.predict(X_new)
# print("Prediction: ", prediction)
# print("Predicted target name: ", iris_dataset['target_names'][prediction])

y_pred = knn.predict(X_test)
print("Test set predictions:/n", y_pred)
print(("Predictions target name: ", iris_dataset['target_names'][y_pred]))

print("Test set score: {:.2f}".format(knn.score(X_test,y_test)))


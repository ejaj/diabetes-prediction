import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

data_frame = pd.read_csv("data/pima-data.csv")

del data_frame['skin']
map_diabetes = {True: 1, False: 0}
data_frame['diabetes'] = data_frame['diabetes'].map(map_diabetes)


# print(data_frame.head())


# Here size means plot-size
def corr_heatmap(data_frame, size=11):
    correlation = data_frame.corr()
    fig, heatmap = plt.subplots(figsize=(size, size))
    heatmap.matshow(correlation)
    plt.xticks(range(len(correlation.columns)), correlation.columns)
    plt.yticks(range(len(correlation.columns)), correlation.columns)
    plt.show()


corr_heatmap(data_frame, 12)

# num_true = 0.0
# num_false = 0.0
# for item in data_frame['diabetes']:
#     if item:
#         num_true += 1
#     else:
#         num_false += 1
# percent_true = (num_true / (num_true + num_false)) * 100
# percent_false = (num_false / (num_true + num_false)) * 100

# num_true = len(data_frame.loc[data_frame['diabetes'] == True])
# num_false = len(data_frame.loc[data_frame['diabetes'] == False])
# print ("Number of True Cases: {0} ({1:2.2f}%)".format(num_true, (num_true / (num_true + num_false)) * 100))
# print ("Number of False Cases: {0} ({1:2.2f}%)".format(num_false, (num_true / (num_true + num_false)) * 100))

# print("Number of True Cases: {0} ({1:2.2f}%)".format(num_true, percent_true))
# print("Number of False Cases: {0} ({1:2.2f}%)".format(num_false, percent_false))

feature_column_names = ['num_preg', 'glucose_conc', 'diastolic_bp', 'thickness', 'insulin', 'bmi', 'diab_pred', 'age']
predicted_class_name = ['diabetes']

x = data_frame[feature_column_names].values
y = data_frame[predicted_class_name].values

split_test_size = 0.30
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=split_test_size, random_state=42)

print("{0:0.2f}% in training set".format((len(X_train) / len(data_frame.index)) * 100))
print("{0:0.2f}% in test set".format((len(X_test) / len(data_frame.index)) * 100))

print("# rows in dataframe {0}".format(len(data_frame)))
print("# rows missing glucose_conc: {0}".format(len(data_frame.loc[data_frame['glucose_conc'] == 0])))
print("# rows missing diastolic_bp: {0}".format(len(data_frame.loc[data_frame['diastolic_bp'] == 0])))
print("# rows missing thickness: {0}".format(len(data_frame.loc[data_frame['thickness'] == 0])))
print("# rows missing insulin: {0}".format(len(data_frame.loc[data_frame['insulin'] == 0])))
print("# rows missing bmi: {0}".format(len(data_frame.loc[data_frame['bmi'] == 0])))
print("# rows missing diab_pred: {0}".format(len(data_frame.loc[data_frame['diab_pred'] == 0])))
print("# rows missing age: {0}".format(len(data_frame.loc[data_frame['age'] == 0])))

fill_0 = SimpleImputer(missing_values=0, strategy="mean")

X_train = fill_0.fit_transform(X_train)
X_test = fill_0.fit_transform(X_test)
nb_model = GaussianNB()
nb_model.fit(X_train, y_train.ravel())

prediction_from_trained_data = nb_model.predict(X_train)
accuracy = metrics.accuracy_score(y_train, prediction_from_trained_data)
print("Accuracy of our naive bayes model is : {0:.4f}".format(accuracy))

prediction_from_test_data = nb_model.predict(X_test)
accuracy = metrics.accuracy_score(y_test, prediction_from_test_data)
print("Accuracy of our naive bayes model is: {0:0.4f}".format(accuracy))

print("Confusion Matrix")

# labels for set 1=True to upper left and 0 = False to lower right
print("{0}".format(metrics.confusion_matrix(y_test, prediction_from_test_data, labels=[1, 0])))

print ("Classification Report")

# labels for set 1=True to upper left and 0 = False to lower right
print ("{0}".format(metrics.classification_report(y_test, prediction_from_test_data, labels=[1, 0])))


rf_model = RandomForestClassifier(random_state=42)

rf_model.fit(X_train, y_train.ravel())

rf_predict_train = rf_model.predict(X_train)

#get accuracy
rf_accuracy = metrics.accuracy_score(y_train, rf_predict_train)

#print accuracy
print ("Accuracy: {0:.4f}".format(rf_accuracy))

rf_predict_test = rf_model.predict(X_test)

#get accuracy
rf_accuracy_testdata = metrics.accuracy_score(y_test, rf_predict_test)

#print accuracy
print ("Accuracy: {0:.4f}".format(rf_accuracy_testdata))

print ("Confusion Matrix for Random Forest")

# labels for set 1=True to upper left and 0 = False to lower right
print ("{0}".format(metrics.confusion_matrix(y_test, rf_predict_test, labels=[1, 0])))

print ("")

print ("Classification Report\n")

# labels for set 1=True to upper left and 0 = False to lower right
print ("{0}".format(metrics.classification_report(y_test, rf_predict_test, labels=[1, 0])))
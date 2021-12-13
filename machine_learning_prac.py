import pandas as pd
import numpy as np
import pickle
# import 
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

# data = pd.read_csv("breast-cancer-wisconsin.data")

# data.columns = ["id", "ClumpThick", "UniSize", "UniShape", "MargAd", "SingEpiCelSize", "BareNuc","BlandChr", "NormalNuc","Mito","Class"]
# data.to_csv("Breast_Cancer_data.csv", index= None, header= True )

df = pd.read_csv("Breast_Cancer_data.csv")

df.drop(["id"], inplace= True, axis = 1)
df.replace("?", -999 , inplace= True)

def return_binary(num):
    if num == 4:
        return 1
    else:
        return 0

# df["Class"] = df["Class"].map(return_binary)
df["Class"] = df["Class"].map(lambda x: 1 if x == 4 else 0)
# print(df)

X = np.array(df.drop(["Class"], axis = 1))
y = np.array(df["Class"])
# print(X,y)

[X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size = 0.1, random_state = 0)
# print(X_train)

##SVM Classifier
# Classifier = svm.SVC(kernel = 'linear')
# ##Train the model using the training sets
# model = Classifier.fit(X_train, y_train)
# ##Check the accuracy of the model
# accuracy = model.score(X_test, y_test)
# print("SVM model",accuracy)

##Logistic Regression classifier
Classifier = LogisticRegression(solver = "liblinear")
model = Classifier.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print("Logistic_Reg", accuracy)


##Save any model
# pickle.dump(model, open("LogisticRegression","wb"))
loaded_model = pickle.load(open("LogisticRegression","rb"))
accuracy = loaded_model.score(X_test, y_test)
print(accuracy)

##Predict the response for test dataset
classes = ["Benign", "Malignant"]
sample = np.array([[5,1,1,1,2,1,3,1,1]])
result = loaded_model.predict(sample)
print(classes[int(result)])
# y_pred = Classifier.predict(X_test)
# print(y_pred)

# print(y_train)

# accuracy = metrics.accuracy_score(y_test, y_pred )
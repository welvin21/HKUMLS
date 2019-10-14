import pandas as pd
import numpy as np
import matplotlib.pyplot as pt
from sklearn.tree import DecisionTreeClassifier

trainData = pd.read_csv('./datasets/train.csv').as_matrix()
testData = pd.read_csv('./datasets/test.csv').as_matrix()
clf = DecisionTreeClassifier()

#training data
xtrain = trainData[0:,1:]
train_label = trainData[0:,0]

clf.fit(xtrain,train_label)

#testing data
xtest = testData[0:,0:]
ImageId,Label = [],[]
for i in range(len(xtest)):
    pred = clf.predict( [xtest[i]] )
    ImageId.append(i+1)
    Label.append(pred[0])
#Create a new pandas dataframe
output = {'ImageId' : ImageId,'Label' : Label}
df = pd.DataFrame(output,columns = ['ImageId','Label'])
export_csv = df.to_csv('./datasets/output.csv',index = None, header = True)


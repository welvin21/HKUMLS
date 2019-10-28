import pandas as pd
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense

def featureEngineer(data):
    data['Sex'] = data['Sex'].map({'male' : 1, 'female' : 0})
    columns = ['Name','Ticket','Fare','Cabin','Embarked','Age']
    return(data.drop(columns,axis=1))

train = pd.read_csv('./datasets/train.csv')
test = pd.read_csv('./datasets/test.csv')
PassengerId = test['PassengerId']

train = featureEngineer(train)
test = featureEngineer(test)

X_train = train.drop('Survived',axis = 1)
y_train = train['Survived']

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(test)

def model():
    model = Sequential()
    model.add(Dense(output_dim = 3, init = 'uniform', activation = 'relu', input_dim = 5))
    model.add(Dense(output_dim = 2, init = 'uniform', activation = 'relu'))
    model.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model

model = model()
model.fit(X_train,y_train, batch_size = 10, nb_epoch = 100)

y_pred = model.predict(X_test)
y_pred = [arr[0] for arr in y_pred]
y_pred = [1 if(i >= 0.5) else(0) for i in y_pred]

output = pd.DataFrame({'PassengerId' : PassengerId, 'Survived' : y_pred}, columns=['PassengerId','Survived'])

output.to_csv('./cnn.csv',index=None,header=True)
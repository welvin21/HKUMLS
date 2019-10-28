import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import re

def data_cleansing(data) : 
    #drop unwanted column
    data = data.drop(['PassengerId'],axis = 1)

    #Convert cabin column to Deck
    #Change N/A value to U0
    deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8} 
    data['Cabin'] = data['Cabin'].fillna('U0') 
    data['Deck'] = data['Cabin'].map(lambda x : re.compile("([a-zA-Z]+)").search(x).group())
    data['Deck'] = data['Deck'].map(deck).fillna(0).astype('int')
    
    #Drop cabin & ticket feature
    data = data.drop(['Cabin','Ticket'],axis = 1)

    #Create random ages data based on mean and stdev
    mean = data['Age'].mean()
    stdev = data['Age'].std()
    nullData = data['Age'].isnull().sum()
    random_ages = np.random.randint(mean - stdev, mean + stdev, size = nullData)
    
    #Fill NaN value in age column with random age generated
    age_slice = data['Age'].copy()
    age_slice[np.isnan(age_slice)] = random_ages
    data["Age"] = age_slice.astype(int)

    #Fill 2 NaN values in embarked column to its top value (S)
    #To find top value : print(data[Embarked].describe())
    #Convert S,Q, or C into numeric
    data['Embarked'] = data['Embarked'].fillna('S')
    ports = {'S' : 0, 'C' : 1, 'Q' : 2}
    data['Embarked'] = data['Embarked'].map(ports)

    #Change fare column data type to int
    data['Fare'] = data['Fare'].fillna(0).astype(int)

    #Convert sex column into numeric (1 or 0)
    sex = {'male' : 0, 'female' : 1}
    data['Sex'] = data['Sex'].map(sex).astype(int)

    #Extract title from name column
    titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    
    #Replace uncommon title value
    data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr','Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    data['Title'] = data['Title'].replace(['Mlle','Ms'],'Miss')
    data['Title'] = data['Title'].replace(['Mme'],'Mr')

    #Convert title into numeric
    data['Title'] = data['Title'].map(titles).fillna(0).astype(int)
    data = data.drop(['Name'],axis = 1)

    return data

def create_category(data) : 
    #Age category
    data.loc[ data['Age'] <= 11, 'Age'] = 0
    data.loc[(data['Age'] > 11) & (data['Age'] <= 18), 'Age'] = 1
    data.loc[(data['Age'] > 18) & (data['Age'] <= 22), 'Age'] = 2
    data.loc[(data['Age'] > 22) & (data['Age'] <= 27), 'Age'] = 3
    data.loc[(data['Age'] > 27) & (data['Age'] <= 33), 'Age'] = 4
    data.loc[(data['Age'] > 33) & (data['Age'] <= 40), 'Age'] = 5
    data.loc[(data['Age'] > 40) & (data['Age'] <= 66), 'Age'] = 6
    data.loc[ data['Age'] > 66, 'Age'] = 6
    data['Age'] = data['Age'].astype(int)

    #Fare category
    data.loc[ data['Fare'] <= 7.91, 'Fare'] = 0
    data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Fare'] = 1
    data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31), 'Fare']   = 2
    data.loc[(data['Fare'] > 31) & (data['Fare'] <= 99), 'Fare']   = 3
    data.loc[(data['Fare'] > 99) & (data['Fare'] <= 250), 'Fare']   = 4
    data.loc[ data['Fare'] > 250, 'Fare'] = 5
    data['Fare'] = data['Fare'].astype(int)

    #New_feature : Relative
    data['Relative'] = (data['SibSp'] + data['Parch']).astype(int)

    #New_feature : Not_Alone
    data['Not_Alone'] = (data['Relative']==0).astype(int)
    
    #New feature : Age_Class
    data['Age_Class'] = data['Age'] * data['Pclass']

    #New feature : Fare_Per_Person
    data['Fare_Per_Person'] = (data['Fare']/(data['Relative']+1)).astype(int)

    return data

train_data = pd.read_csv('./datasets/train.csv')
test_data = pd.read_csv('./datasets/test.csv')

#Double the dataset size
train_data = train_data.append(train_data)
train_data = train_data.append(train_data)
train_data = train_data.append(train_data)
# print(train_data.info())

#Get PassengerId from test data before data cleansing
PassengerId = test_data['PassengerId']

#Apply data cleansing function
train_data = data_cleansing(train_data)
test_data = data_cleansing(test_data)
#Add more features
train_data = create_category(train_data)
test_data = create_category(test_data)

X_train = train_data.drop(['Survived'],axis = 1)
y_train = train_data['Survived']

#ML model (Random Forest)
# random_forest_model = RandomForestClassifier(n_estimators=100)
# random_forest_model.fit(X_train,y_train)
# print(random_forest_model.score(X_train, y_train))
model = DecisionTreeClassifier()
model.fit(X_train,y_train)
print(model.score(X_train,y_train))

y_pred = model.predict(test_data)
output = pd.DataFrame({'PassengerId' : PassengerId, 'Survived' : y_pred}, columns=['PassengerId','Survived'])

output.to_csv('./output.csv',index=None,header=True)


# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

# machine learning

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split



train_df = pd.read_csv('input/train.csv')
test_df = pd.read_csv('input/test.csv')
combine = [train_df, test_df]

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

# Extract Titles
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

# Group titles with identical meaning, or rare if uncommon in dataset
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', \
                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')


train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]



# Fill missing ages based off Title
avg_age = train_df[['Title', 'Age']].groupby(['Title'], as_index=False).mean()
age_dict = dict(zip(avg_age['Title'], avg_age['Age']))
for dataset in combine:
    dataset.loc[dataset['Age'].isnull(), 'Age'] = dataset['Title'].map(age_dict)

# Group Ages
for dataset in combine:
    dataset['Age_bin'] = pd.cut(dataset['Age'], bins=[0, 12, 20, 40, 120], labels=['Child', 'Teen', 'Adult', 'Elder'])
    dataset.drop(['Age'],axis=1,inplace=True)

# Ordinal values for departure port
freq_port = train_df.Embarked.dropna().mode()[0]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

# Fill NA Fare values
test_df['Fare'].fillna(test_df['Fare'].dropna().mean(), inplace=True)

"""
pyplot.boxplot(dataset['Fare'])
pyplot.show()
"""

# Group Fares into Buckets
for dataset in combine:
    dataset['Fare_bin'] = pd.cut(dataset['Fare'], bins=[0, 7.91, 14.454, 31, 600], labels=['Low', 'Med', 'High', 'VeryHigh'])
    dataset.drop(['Fare'], axis=1, inplace=True)

# Create FamilySize
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    dataset.drop(['SibSp', 'Parch'], axis=1, inplace=True)

combine = [train_df, test_df]

train_df = pd.get_dummies(train_df, columns=["Title", "Sex", "Age_bin", "Embarked", "Fare_bin"], prefix=["Title", "Sex", "Age", "Embark", "Fare"])
test_df = pd.get_dummies(test_df, columns=["Title", "Sex", "Age_bin", "Embarked", "Fare_bin"], prefix=["Title", "Sex", "Age", "Embark", "Fare"])

train_df = train_df.dropna()
test_df = test_df.dropna()
combine = [train_df, test_df]

X_train = train_df.drop("Survived", axis=1)
X_test = test_df.drop("PassengerId", axis=1).copy()
Y_train = train_df["Survived"]

# Decision Tree
print("Decision Tree:")
decision_tree = DecisionTreeClassifier(max_depth=10, min_samples_split=.4)
decision_tree.fit(X_train, Y_train)
# Y_pred = decision_tree.predict(X_test
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
fig = plt.figure(figsize=(25,20))
features = X_train.columns
tree.plot_tree(decision_tree, feature_names=features,filled=True)
fig.savefig("decision_tree.png")
scores = cross_val_score(decision_tree, X_train, Y_train, cv=5)
score = round(scores.mean() * 100, 2)
print(f"Average classification accuracy: {score}")
print(f"Training Accuracy: {acc_decision_tree}")

print()


# Random Forest
print("Random Forest Tree:")
random_forest = RandomForestClassifier(n_estimators=100, max_depth=80, max_features=3, min_samples_leaf=5, min_samples_split=12)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
scores = cross_val_score(random_forest, X_train, Y_train, cv=5)
score = round(scores.mean() * 100, 2)
print(f"Average Classification Accuracy: {score}")
print(f"Training Accuracy {acc_random_forest}")



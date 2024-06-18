import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print(train.head())

train.info()

print(train.describe())

print(train.isnull().sum())

train['Age'].fillna(train['Age'].median(), inplace=True)

train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)

train.drop(columns=['Cabin'], inplace=True)

print(train.isnull().sum())

sns.set_style('whitegrid')

plt.figure()
sns.countplot(x='Survived', data=train)
plt.title('Distribution of Survival')
plt.show(block=False)

#distribution of 'Age'
plt.figure()
sns.histplot(train['Age'], bins=30, kde=True)
plt.title('Age Distribution of Passengers')
plt.show(block=False)

#distribution of 'Pclass'
plt.figure()
sns.countplot(x='Pclass', data=train)
plt.title('Passenger Distribution according to Class')
plt.show(block=False)


#Showing the graphs of survival rates according to:

#Passenger class
plt.figure()
sns.barplot(x='Pclass', y='Survived', data=train)
plt.title('Survival Rate by Passenger Class')
plt.show(block=False)

#Gender
plt.figure()
sns.barplot(x='Sex', y='Survived', data=train)
plt.title('Survival Rate by Gender')
plt.show(block=False)

#Age
plt.figure()
sns.histplot(train[train['Survived'] == 'Yes']['Age'], bins=30, kde=True, color='red', label='Survived')
sns.histplot(train[train['Survived'] == 'No']['Age'], bins=30, kde=True, color='blue', label='Did Not Survive')
plt.legend()
plt.title('Number of survivors with respect to age')
plt.show(block=False)
#End survival rate graph

# Pairplot for selected features
sns.pairplot(train[['Survived', 'Pclass', 'Age', 'Fare', 'SibSp', 'Parch']], hue='Survived')
plt.suptitle('Pairplot of Selected Features', y=1.02)
plt.show(block=False)

# Heatmap of correlations
plt.figure(figsize=(10, 8))
numeric_columns = train.select_dtypes(include=[np.number])
correlation_matrix = numeric_columns.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show(block=False)

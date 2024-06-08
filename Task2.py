import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('Titanic.csv')

# Display the first few rows of the dataset
print("Original Data:")
print(data.head())

# Basic Information
print("\nDataset Information:")
print(data.info())

print("\nSummary Statistics:")
print(data.describe())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Data Cleaning
# Fill missing 'Age' values with the mean age
data['Age'].fillna(data['Age'].mean(), inplace=True)

# Fill missing 'Embarked' values with the most frequent value (mode)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# Drop the 'Cabin' column due to too many missing values
data.drop(columns=['Cabin'], inplace=True)

# Drop rows with missing 'Fare' values (if any)
data.dropna(subset=['Fare'], inplace=True)

# Verify the cleaned data
print("\nCleaned Data Missing Values:")
print(data.isnull().sum())

# Exploratory Data Analysis (EDA)
# Distribution of Survived
plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', data=data, palette='pastel')
plt.title('Distribution of Survival')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.show()

# Distribution of Age
plt.figure(figsize=(10, 6))
sns.histplot(data['Age'], bins=30, kde=True, color='blue')
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Survival by Gender
plt.figure(figsize=(8, 6))
sns.countplot(x='Survived', hue='Sex', data=data, palette='viridis')
plt.title('Survival by Gender')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.show()

# Survival by Class
plt.figure(figsize=(8, 6))
sns.countplot(x='Survived', hue='Pclass', data=data, palette='coolwarm')
plt.title('Survival by Class')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.show()

# Age distribution by Survival status
plt.figure(figsize=(10, 6))
sns.histplot(data=data, x='Age', hue='Survived', multiple='stack', palette='rocket_r', bins=30)
plt.title('Age Distribution by Survival Status')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Save the cleaned data to a new CSV file
data.to_csv('cleaned_titanic_data.csv', index=False)

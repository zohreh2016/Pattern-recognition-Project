
# coding: utf-8

# In[21]:


# Import libraries
import pandas as pd
import numpy as np
import sklearn as skl
import copy
import scipy
import matplotlib


# In[3]:


# Load Data
stroke_df = pd.read_csv("data/train_2v.csv",sep=",")
stroke_df.head()


# In[4]:


# Dataframe information
print(stroke_df.info())
# missing data in bmi and smoking_status columns


# In[7]:


#to see missing values
missing = stroke_df.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar()


# In[ ]:


# Handle missing data 

#For  Smoking_status column, we can fill in the value with the "never smoked",for the sake of simplicity,(since that is the most common value):(since that is the most common value):
# another way that we can do is in the following:
    


# In[22]:


# Handle missing data 

# Mean value for nan data in bmi column
stroke_df = stroke_df.fillna({'bmi': stroke_df['bmi'].mean()})

# Consider children as non smokers
stroke_df.loc[(stroke_df['smoking_status'].isnull()) & (stroke_df['age'] <=15), "smoking_status"] = "never smoked"

# Fill nan data in smoking_status column with the mode value in this column
stroke_df = stroke_df.fillna({'smoking_status' : stroke_df['smoking_status'].value_counts().index[0]})

# Check if there is no nan value (print 0 if there is no nan value)
print(stroke_df.isnull().values.sum())


# In[23]:


# Make a dataframe for categorical features
cat_df = stroke_df.select_dtypes(include=['object']).copy()
cat_df.head()


# In[24]:


stroke_df['ever_married'].value_counts()


# In[25]:


stroke_df['work_type'].value_counts()


# In[26]:


# A template to plot a barplot of the frequency distribution of a categorical feature
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
column_name = 'work_type'
feature_count = cat_df[column_name].value_counts()
sns.set(style="darkgrid")
sns.barplot(feature_count.index, feature_count.values, alpha=0.9)
plt.title('Frequency Distribution of' + column_name)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel(column_name, fontsize=12)
plt.show()


# In[27]:


# Practice different ways of handling categorical data


# method #1: replace values

cat_df_replace = cat_df.copy()
for column in cat_df.columns:
    labels = cat_df[column].astype('category').cat.categories.tolist()
    replace_map_comp = {column : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}

    print(replace_map_comp)

    cat_df_replace.replace(replace_map_comp, inplace=True)

cat_df_replace.head()


# In[28]:


# Practice different ways of handling categorical data


# method #2: label encoding

cat_df_lc = cat_df.copy()
for column in cat_df.columns:    
    cat_df_lc[column] = cat_df_lc[column].astype('category')
    cat_df_lc[column] = cat_df_lc[column].cat.codes
cat_df_lc.head()
# It seems faster than previous method


# In[29]:


# Practice different ways of handling categorical data


# method #3: One-Hot encoding

cat_df_onehot = cat_df.copy()
for column in cat_df.columns:    
    cat_df_onehot = pd.get_dummies(cat_df_onehot, columns=[column], prefix = [column])

cat_df_onehot.head()
# The problem is many new columns!


# In[30]:


# Replace dataframe

df_cat_no = stroke_df.copy()
df_cat_no = df_cat_no.drop(columns = cat_df.columns)
df_cat_no = df_cat_no.join(cat_df_lc)
df_cat_no.head()


# In[31]:


# New dataframe information
df_cat_no.info()


# In[35]:


#Checking of Linearity between features
from pandas import DataFrame


# In[32]:


# relation of bmi and stroke
#relation of gloucoze level and bmi


# In[36]:


df = DataFrame(df_cat_no,columns=['bmi','avg_glucose_level','stroke', 'age','heart_disease','hypertension'])
 
plt.scatter(df['bmi'], df['stroke'], color='red')
plt.title('stroke Vs bmi', fontsize=14)
plt.xlabel('bmi', fontsize=14)
plt.ylabel('stroke', fontsize=14)
plt.grid(True)
plt.show()
 
plt.scatter(df['avg_glucose_level'], df['bmi'], color='green')
plt.title('bmi Vs avg_glucose_level', fontsize=14)
plt.xlabel('avg_glucose_level', fontsize=14)
plt.ylabel('bmi', fontsize=14)
plt.grid(True)
plt.show()


plt.scatter(df['avg_glucose_level'], df['stroke'], color='blue')
plt.title('stroke Vs avg_glucose_level', fontsize=14)
plt.xlabel('avg_glucose_level', fontsize=14)
plt.ylabel('stroke', fontsize=14)
plt.grid(True)
plt.show()


plt.scatter(df['age'], df['stroke'], color='yellow')
plt.title('age Vs Stroke', fontsize=14)
plt.xlabel('age', fontsize=14)
plt.ylabel('stroke', fontsize=14)
plt.grid(True)
plt.show()


plt.scatter(df['heart_disease'], df['stroke'], color='orange')
plt.title('heart_disease Vs Stroke', fontsize=14)
plt.xlabel('heart_disease', fontsize=14)
plt.ylabel('stroke', fontsize=14)
plt.grid(True)
plt.show()


# In[37]:


# check for outliers
import seaborn as sns
sns.boxplot(x=df_cat_no['bmi'])


# In[41]:


# check for outliers

sns.boxplot(x=df_cat_no['avg_glucose_level'])


# In[47]:


# to see where stroke is 1
df_cat_no[df_cat_no.stroke==1]


# In[48]:


# only 783 out of 43400 entries


# In[87]:


# Apply PCA

# Extract features from dataframe
y = df_cat_no.loc[:, 'stroke'].values
features = df_cat_no.columns.tolist()
features.remove('stroke')
features.remove('id')
x = df_cat_no.loc[:, features].values

# Normalize data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(x)
x = scaler.transform(x)

# PCA projection
from sklearn.decomposition import PCA

# See all components
pca = PCA()
principal_components = pca.fit_transform(x)
print(pca.explained_variance_ratio_)

pca = PCA(n_components = 10)
principal_components = pca.fit_transform(x)

columns_list = ['principal component '+str(i) for i in range(1,principal_components.shape[1]+1)]
principal_df = pd.DataFrame(data = principal_components
             , columns = columns_list)

principal_df.head()


# In[90]:


# Plot result of pca for no_comp=2

principal_df_plot = principal_df.copy()
principal_df_plot['stroke'] = y
principal_df_plot.head()

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 5', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

targets = [0, 1]
colors = ['r', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = principal_df_plot['stroke'] == target
    ax.scatter(principal_df_plot.loc[indicesToKeep, 'principal component 1']
               , principal_df_plot.loc[indicesToKeep, 'principal component 5']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()


# In[57]:


#1
# SVM Classifier
from sklearn import svm

x_data = principal_df.values
clf_svm = svm.SVC(gamma='scale')
clf_svm.fit(x_data, y)



# In[58]:


# it takes only some seconds to show the result of SVM


# In[59]:


# Cross validation on training data

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer

score_func = make_scorer(accuracy_score)
score = cross_val_score(clf_svm, x_data, y, scoring=score_func, cv=5)
print(score)


# In[60]:


# it also took some seconds to show the result but more than SVM classifer.


# In[61]:


#2
# Logistic Regression
from sklearn.linear_model import LogisticRegression

clf_lr = LogisticRegression(random_state=0, solver='lbfgs')
clf_lr.fit(x_data, y)



# In[62]:


#it was too fast


# In[63]:


# Cross validation on training data

score = cross_val_score(clf_lr, x_data, y, scoring=score_func, cv=5)
print(score)


# In[64]:


#3
# Decision Tree 
from sklearn.tree import DecisionTreeClassifier

clf_dt = DecisionTreeClassifier(random_state=0)
clf_dt.fit(x_data, y)


# In[65]:


# Cross validation on training data

score = cross_val_score(clf_dt, x_data, y, scoring=score_func, cv=5)
print(score)


# In[66]:


# Evaluate models on train set
from sklearn.metrics import accuracy_score
y_pred_svm = clf_svm.predict(x_data)
y_pred_lr = clf_lr.predict(x_data)
y_pred_dt = clf_dt.predict(x_data)

acc_svm = accuracy_score(y, y_pred_svm)
acc_lr = accuracy_score(y, y_pred_lr)
acc_dt = accuracy_score(y, y_pred_dt)

print("svm: ", acc_svm)
print("logistic regression: ", acc_lr)
print("decision tree: ", acc_dt)


# In[91]:


# Test models on test data
# we can split our train data in order to have train and test data, or use the test data that we have without split.


# split data to test and train data
x = principal_df.iloc[:, :-1].values
y = principal_df.iloc[:, -1].values

# Spliting
# the final preprocessing step is to divide data into training and test sets. 
from sklearn.model_selection import train_test_split
xtrn, xtes, ytrn, ytes = train_test_split(x, y, test_size=0.2, random_state=123)


# In[92]:


# but we use the test data that we have


# In[95]:


# Test models on test data

# Load test data
test_df = pd.read_csv("data/test_2v.csv")
test_df.head()


# In[96]:


# Handle missing data 

# Mean value for nan data in bmi column
test_df = test_df.fillna({'bmi': test_df['bmi'].mean()})

# Consider children as non smokers
test_df.loc[(test_df['smoking_status'].isnull()) & (test_df['age'] <=15), "smoking_status"] = "never smoked"

# Fill nan data in smoking_status column with the mode value in this column
test_df = test_df.fillna({'smoking_status' : test_df['smoking_status'].value_counts().index[0]})

# Check if there is no nan value (print 0 if there is no nan value)
print(test_df.isnull().values.sum())


# In[97]:


# Make a dataframe for categorical features
cat_test_df = test_df.select_dtypes(include=['object']).copy()
cat_test_df.head()


# In[98]:


# method #2 for test dataset

cat_df_lc = cat_test_df.copy()
for column in cat_df.columns:    
    cat_df_lc[column] = cat_df_lc[column].astype('category')
    cat_df_lc[column] = cat_df_lc[column].cat.codes
cat_df_lc.head()


# In[100]:


# Replace dataframe

test_df_cat_no = test_df.copy()
test_df_cat_no = test_df_cat_no.drop(columns = cat_test_df.columns)
test_df_cat_no = test_df_cat_no.join(cat_df_lc)
test_df_cat_no.head()


# In[101]:


# Apply PCA

# Extract features from dataframe
features = test_df_cat_no.columns.tolist()
features.remove('id')
x_test = test_df_cat_no.loc[:, features].values

# Normalize data
x_test = scaler.transform(x_test)

# PCA
test_principal_components = pca.fit_transform(x_test)

columns_list = ['principal component '+str(i) for i in range(1,principal_components.shape[1]+1)]
test_principal_df = pd.DataFrame(data = principal_components
                                 , columns = columns_list)

test_principal_df.head()


# In[113]:


#1
#Predict

# SVM Classifier

x_test_data = test_principal_df.values

y_test_pred_svm = clf_svm.predict(x_test_data)
print(y_test_pred_svm)


# In[108]:


#2
# Logistic Regression

y_test_pred_lr = clf_lr.predict(x_test_data)
print(y_test_pred_lr)


# In[109]:


#3
# Decision Tree

y_test_pred_dt = clf_dt.predict(x_test_data)
print(y_test_pred_dt)


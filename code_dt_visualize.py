# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 23:16:19 2023
@author: Ammar Ahmed Siddiqui, MS DS Bahria University

"""

# all imports belows
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing libraries for decision Tree model implementation
from sklearn.model_selection import train_test_split
from sklearn import tree

# Prprocessors
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Importing library for decision tree visualization
#from dtreeviz.trees import *

# installed package "dtreeviz.trees" via following line
# pip install dtreeviz

# loading dataset named "dataset1"

# Change your working folder accordingly
working_folder = "D:/IMPORTANT/MASTERS/BAHRIA UNIV/Spring2023 Semester/Assigments/TDS/DistanceTree/"

dataset_orig = pd.read_csv(working_folder + "dataset1.csv",
                           index_col=False)


# data-pre-processing

# Removing duplicate entries
dataset_orig.drop_duplicates()


# Splitting dataset
# and removing the unrequired column

selected_features = [ 'Weather','Parents','Money']
decision_feature = ['Decision']
target_class_values = ['Cinema','Shopping','StayIn','Tennis']

df_f = dataset_orig[selected_features]
df_d = dataset_orig[decision_feature]


# Sorting by Index
df_f.sort_index()                     
df_d.sort_index() 


# Prep for applying decision tree

# Data Encoding

encode = LabelEncoder()
df_f['Weather']     = encode.fit_transform(df_f['Weather'])
df_f['Parents']     = encode.fit_transform(df_f['Parents'])
df_f['Money']       = encode.fit_transform(df_f['Money'])
df_d['Decision']    = encode.fit_transform(df_d['Decision'])

                    

# Spliting data into training/test at 80-20 ration                       
# X_train, X_test, y_train, y_test = train_test_split(df_f, df_d, test_size=0.2, random_state=42)
# we are not splitting, so that our entire dataset is used for making of DT
# and thus we can compare it with our classwork
               
# fit the classifier
clf = tree.DecisionTreeClassifier(max_depth=5, random_state=42)


#clf.fit(X_train, y_train)
# Not splitting training data for above mentioned reason


clf.fit(df_f,df_d)

# Plotting the decision Tree

fig = plt.figure(figsize=(25,20))

tree.plot_tree(clf,
               feature_names = selected_features, 
               class_names = target_class_values,
               fontsize=25,
               precision=3,
               rounded=True, 
               filled=True)

# Saving figure to disk
fig.savefig(working_folder+"decistion_tree.png")

# Generating text description
text_representation = tree.export_text(clf)
print(text_representation)








""" __________________________________________End of Code
"""



"""
print(explain_prediction_path(clf, X_test[0], 
                              feature_names=selected_features, 
                              explanation_type="plain_english"))
"""




""" ____________________________________________________

(1) 
Encountered following error
when a component was not installed
"ImportError: cannot import name 'dtreeviz' from 'dtreeviz.trees' (C:\ProgramData\Anaconda3\lib\site-packages\dtreeviz\trees.py)"

Following was solution
pip install dtreeviz==1.3

(2)
restart kernel when needed

"""
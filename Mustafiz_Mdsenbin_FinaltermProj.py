#!/usr/bin/env python
# coding: utf-8

# <p style="text-align: center; font-size: 18px; line-height: .7;">Md Sen Bin Mustafiz</p>
# <p style="text-align: center; line-height: 1.2;">
# mbm52@njit.edu<br>
# NJIT ID: 31690921<br>
# 24 Nov, 2024<br>
# Professor Yasser Abdullah<br>
# CS 634: Data Mining
# </p>
# <p style="text-align: center; font-size: 18px; line-height: .7;">Final Project Report</p>
# 

# A machine learning classifier is an algorithm used to determine the category or class of a data point. 
# It is a supervised learning technique where the model is trained on labeled data, consisting of input features and their corresponding output labels. 
# The classifier identifies patterns in the training data and uses this understanding to classify new data.
# 
# Main Components of a Classifier:
# - Input Features: Characteristics or attributes of the data.  
# - Labeled Data: Data with known categories for training.  
# - Classification Model: The algorithm (e.g., Decision Tree, SVM, Neural Networks) that learns from the data.  
# - Output Class: The predicted category for the input data.

# A machine learning classifier relies on structured data to make accurate predictions, with **input features**, **labeled data**, 
# and **output classes** playing crucial roles in its functioning. In this project I use Car Evaluation Database. It is based on a hierarchical decision model for evaluating car acceptability. It simplifies the decision structure by linking car acceptability directly to six input attributes: 
# 
# 1. buying (v-high, high, med, low)
# 2. maint (v-high, high, med, low)
# 3. doors (2, 3, 4, 5-more)
# 4. persons (2, 4, more)
# 5. lug_boot (small, med, big)
# 6. safety (low, med, high)
#  
# The dataset contains 1,728 instances with no missing values and classifies the data into four categories:  
# 
# 1. unacceptable
# 2. acceptable 
# 3. good 
# 4. very good
# 
# This dataset is widely used for testing machine learning methods such as structure discovery and constructive induction.

# **Classification Model:** In this project I used 3 different classification algorithms in Python. They are:
# 1. Random Forest
# 2. Naïve Bayes
# 3. Bidirectional-LSTM
# 
# In evaluating classification performance, I also used the 10-fold cross validation 
# metho in every classification model.d

# ### Importing the package
# 
# Remove the # and import the pacage when you run it.

# In[6]:


#pip install tensorflow


# ### Importing the libraries that are required for the project
# 

# In[8]:


# Import libraries
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
import warnings


# ### Data reading

# In[10]:

import os
import pandas as pd

# Dynamically set the working directory to the script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)



# Load the dataset
data = pd.read_csv('car.csv')  # csv file

# Encode catagory
label_encoders = {}
for column in data.columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# divide 
X = data.drop(columns='class')
y = data['class']


# ### 10 fold cross validation

# In[12]:


# k = 10 fold
kfold = KFold(n_splits=10, shuffle=True, random_state=42)


# ## 1. Random Forest Classifier

# In[14]:


# Random Forest Classifier
rf_mod = RandomForestClassifier(random_state=42)


# Here I used Random Forest classifier to calculate values like Confusion matrix, Sensitivity, Specificity, False Positive Rate,
# False Negative Rate, precision, F1 score, Balanced Accuracy, True Skill Statistic, Heidke Skill Score and AUC.
# The results for each fold are stored for overall evaluation.

# In[16]:


# empty list to store values for each fold
fold_values = []

for i, (train_index, test_index) in enumerate(kfold.split(X), start=1):
    # Splitting the data
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]    # Train 
    rf_mod.fit(X_train, y_train)
    y_pred = rf_mod.predict(X_test)
    
    # Confusion matrix 
    cm = confusion_matrix(y_test, y_pred)
    tp = cm.diagonal()  # True Positives 
    fn = cm.sum(axis=1) - tp  # False Negatives 
    fp = cm.sum(axis=0) - tp  # False Positives
    tn = cm.sum() - (fp + fn + tp)  # True Negatives 

 
    p = tp + fn
    n = tn + fp
    TPR = tp / (tp + fn)  # Sensitivity
    TNR = tn / (tn + fp)  # Specificity 
    FPR = fp / (fp + tn)  # False Positive Rate 
    FNR = fn / (fn + tp)  # False Negative Rate
    Precision = tp / (tp + fp)  # Precision 
    F1_measure = 2 * (Precision * TPR) / (Precision + TPR)  # F1 Score
    Accuracy = accuracy_score(y_test, y_pred)
    Error_rate = 1 - Accuracy
    BACC = (TPR + TNR) / 2  # Balanced Accuracy
    TSS = TPR - FPR  # True Skill Statistic 
    HSS = (2 * (tp * tn - fp * fn)) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))  # Heidke Skill Score 
    
    #  Brier Score 
    y_proba = rf_mod.predict_proba(X_test)  # Probabilities
    brier_score = np.mean([(y_proba[:, i] - (y_test == i).astype(int)) ** 2 for i in range(y_proba.shape[1])])
    
    #  AUC 
    try:
        auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
    except ValueError:
        auc = np.nan  #  NaN if calculation not meet
    
    # Store averaged values
    fold_values.append([
        tp.mean(), tn.mean(), fp.mean(), fn.mean(), p.mean(), n.mean(),
        TPR.mean(), TNR.mean(), FPR.mean(), FNR.mean(),
        Precision.mean(), F1_measure.mean(),
        Accuracy, Error_rate, BACC.mean(), TSS.mean(), HSS.mean(),
        brier_score, auc, Accuracy  # Acc_by_package_fn 
    ])


# ### Printing Output

# In[18]:


# values to DataFrame
values_df = pd.DataFrame(fold_values, columns=[
    "TP", "TN", "FP", "FN", "P", "N", "TPR", "TNR", "FPR", "FNR", "Precision", "F1 measure",
    "Accuracy", "Error_rate", "BACC", "TSS", "HSS", "Brier score", "AUC", "Acc_by_package_fn"
])

# Transpose 
value_df_rf = values_df.T
value_df_rf.columns = [f"Fold : {i+1}" for i in range(value_df_rf.shape[1])]

# Display
value_df_rf


# ## 2. Naive Bayes Model

# Here I used Naive Bayes classifier to calculate values like Confusion matrix, Sensitivity, Specificity, False Positive Rate, False Negative Rate, precision, F1 score, Balanced Accuracy, True Skill Statistic, Heidke Skill Score and AUC. 
# The results for each fold are stored for overall evaluation.

# In[21]:


# Initialize Naive Bayes classifier

nb_model = GaussianNB()


# In[22]:


#  empty list 
fold_value = []

# Loop through each fold
for i, (train_index, test_index) in enumerate(kfold.split(X), start=1):
    # Splitting data
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Train 
    nb_model.fit(X_train, y_train)
    y_pred = nb_model.predict(X_test)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tp = cm.diagonal()  # True Positives
    fn = cm.sum(axis=1) - tp  # False Negatives
    fp = cm.sum(axis=0) - tp  # False Positives 
    tn = cm.sum() - (fp + fn + tp)  # True Negatives
    p = tp + fn
    n = tn + fp

   
    TPR = tp / (tp + fn)  # Sensitivity (Recall) 
    TNR = tn / (tn + fp)  # Specificity
    FPR = fp / (fp + tn)  # False Positive Rate
    FNR = fn / (fn + tp)  # False Negative Rate 
    
    #  Precision and F1_measure
    Precision = np.divide(tp, (tp + fp), out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
    F1_measure = np.divide(2 * (Precision * TPR), (Precision + TPR), out=np.zeros_like(TPR, dtype=float), where=(Precision + TPR) != 0)
    
    Accuracy = accuracy_score(y_test, y_pred)
    Error_rate = 1 - Accuracy
    BACC = (TPR + TNR) / 2  # Balanced Accuracy 
    TSS = TPR - FPR  # True Skill Statistic
    HSS = (2 * (tp * tn - fp * fn)) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))  # Heidke Skill Score
    
    #  Brier Score
    y_proba = nb_model.predict_proba(X_test)  # Probabilities 
    brier_score = np.mean([(y_proba[:, i] - (y_test == i).astype(int)) ** 2 for i in range(y_proba.shape[1])])
    
    # AUC
    try:
        auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
    except ValueError:
        auc = np.nan  #  NaN 
    
    # averaged values 
    fold_value.append([
        tp.mean(), tn.mean(), fp.mean(), fn.mean(),p.mean(),n.mean(),
        TPR.mean(), TNR.mean(), FPR.mean(), FNR.mean(),
        Precision.mean(), F1_measure.mean(),
        Accuracy, Error_rate, BACC.mean(), TSS.mean(), HSS.mean(),
        brier_score, auc, Accuracy  # Acc_by_package_fn 
    ])


# ### Printing Output

# In[24]:


# values to DataFrame 
value_df = pd.DataFrame(fold_value, columns=[
    "TP", "TN", "FP", "FN","P","N", "TPR", "TNR", "FPR", "FNR", "Precision", "F1_measure",
    "Accuracy", "Error_rate", "BACC", "TSS", "HSS", "Brier_score", "AUC", "Acc_by_package_fn"
])
 #transpose
value_df_nb = value_df.T
value_df_nb.columns = [f"Fold {i+1}" for i in range(value_df_nb.shape[1])]

# display 
value_df_nb


# ## 3. Bidirectional-LSTM 

# Here I used Bidirectional-LSTM classifier to calculate values like Confusion matrix, Sensitivity, Specificity, False Positive Rate, False Negative Rate, precision, F1 score, Balanced Accuracy, True Skill Statistic, Heidke Skill Score and AUC.
# The results for each fold are stored for overall evaluation.

# In[27]:


# Standardize features 
scaler = StandardScaler()
X = scaler.fit_transform(X)

# target variable to categorical 
y = to_categorical(y)


# In[28]:


# Function for Bidirectional-LSTM model
def create_bidirectional_lstm(input_shape, num_classes):
    model = Sequential()
    model.add(Input(shape=input_shape))  
    model.add(Bidirectional(LSTM(64)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# In[29]:


# Initialize an empty list
warnings.filterwarnings("ignore")
fold_value = []

# Reshape input data to be compatible with LSTM
X = X.reshape(X.shape[0], X.shape[1], 1)
input_shape = (X.shape[1], 1)
num_classes = y.shape[1]

# Loop through each fold
for i, (train_index, test_index) in enumerate(kfold.split(X), start=1):
    # Splitting the data
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Create and train the Bidirectional-LSTM model
    model = create_bidirectional_lstm(input_shape, num_classes)
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    

    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_test_class = np.argmax(y_test, axis=1)  
    
    # Confusion matrix 
    cm = confusion_matrix(y_test_class, y_pred)
    tp = cm.diagonal()  # True Positives 
    fn = cm.sum(axis=1) - tp  # False Negatives
    fp = cm.sum(axis=0) - tp  # False Positives 
    tn = cm.sum() - (fp + fn + tp)  # True Negatives 

    p = tp + fn
    n = tn + fp

    
    TPR = tp / (tp + fn)  # Sensitivity 
    TNR = tn / (tn + fp)  # Specificity 
    FPR = fp / (fp + tn)  # False Positive Rate 
    FNR = fn / (fn + tp)  # False Negative Rate s
    
    Precision = np.divide(tp, (tp + fp), out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
    F1_measure = np.divide(2 * (Precision * TPR), (Precision + TPR), out=np.zeros_like(TPR, dtype=float), where=(Precision + TPR) != 0)
    
    Accuracy = accuracy_score(y_test_class, y_pred)
    Error_rate = 1 - Accuracy
    BACC = (TPR + TNR) / 2  # Balanced Accuracy 
    TSS = TPR - FPR  # True Skill Statistic
    HSS = (2 * (tp * tn - fp * fn)) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))  # Heidke Skill Score 
    
    #  Brier Score 
    brier_score = np.mean([(y_pred_proba[:, i] - (y_test_class == i).astype(int)) ** 2 for i in range(y_pred_proba.shape[1])])
    
    # AUC
    try:
        auc = roc_auc_score(y_test_class, y_pred_proba, multi_class='ovr')
    except ValueError:
        auc = np.nan  # NaN
    
    # averaged 
    fold_value.append([
        tp.mean(), tn.mean(), fp.mean(), fn.mean(),p.mean(),n.mean(),
        TPR.mean(), TNR.mean(), FPR.mean(), FNR.mean(),
        Precision.mean(), F1_measure.mean(),
        Accuracy, Error_rate, BACC.mean(), TSS.mean(), HSS.mean(),
        brier_score, auc, Accuracy  # Acc_by_package_fn 
    ])


# ### Printing Output

# In[31]:


# value to DataFrame
value_df = pd.DataFrame(fold_value, columns=[
    "TP", "TN", "FP", "FN","P","N", "TPR", "TNR", "FPR", "FNR", "Precision", "F1_measure",
    "Accuracy", "Error_rate", "BACC", "TSS", "HSS", "Brier_score", "AUC", "Acc_by_package_fn"
])

# Transpose
value_df_bilstm = value_df.T
value_df_bilstm.columns = [f"Fold {i+1}" for i in range(value_df_bilstm.shape[1])]

# Display
value_df_bilstm


# ### Average Output
# In this section I calculae the average of each calculation criteria and show them in a table for easy comparison.

# In[33]:


values = [
    "TP", "TN", "FP", "FN","P","N", "TPR", "TNR", "FPR", "FNR", 
    "Precision", "F1_measure", "Accuracy", "Error_rate", 
    "BACC", "TSS", "HSS", "Brier_score", "AUC", "Acc_by_package_fn"
]

# names
value_df_rf.index = values
value_df_nb.index = values
value_df_bilstm.index = values

# Calculate the mean
avg_value_rf = value_df_rf.mean(axis=1)  # Average Random Forest
avg_value_nb = value_df_nb.mean(axis=1)  # Average  Naive Bayes
avg_value_bilstm = value_df_bilstm.mean(axis=1)  # Average Bidirectional LSTM 

# averages to DataFrame
avg_values_combined = pd.DataFrame({
    "Random Forest": avg_value_rf,
    "Naive Bayes": avg_value_nb,
    "Bidirectional-LSTM": avg_value_bilstm
})

#index name 
avg_values_combined.index.name = "Values"

# Display 

print(avg_values_combined)


# ### Conclusion: 
# The Random Forest model is the best performer among the three, 
# with the highest accuracy (98.15%), precision (96.77%), True positive rate (TPR) (94.65%), and F1-measure (95.28%), 
# as well as the lowest error rate (1.82%). It consistently delivers the most reliable results across all metrics. 
# Bidirectional-LSTM performs moderately well, with an accuracy of 83.56%, but falls short compared to Random Forest. 
# Naive Bayes, however, performs poorly, with a low accuracy of 62.67% and high error rate (37.33%), making it the least suitable option. 
# Therefore, Random Forest is the best choice for this task, while Bidirectional-LSTM may be considered for sequential data, 
# and Naive Bayes should be avoided.

# In[ ]:





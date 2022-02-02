#!/usr/bin/env python
# coding: utf-8

# 
# # Breast Cancer Detection Using Machine Learning
# 

# -----------------------------------------------------------------------------------------------------------------------------------------

# # Import Essential Libraries 

# In[1]:


import pandas as pd # for data manupulation or analysis
import numpy as np # for numeric calculation
import matplotlib.pyplot as plt # for data visualization
import seaborn as sns # for data visualization


# In[2]:


# Load breast cancer data into dataframe
cancer_data = pd.read_csv("data.csv")

# head (first 5 row) of the cancer data frame, diagnosis = 'malignant', 'benign'
cancer_data.head()  


# In[3]:


# check the shape of the data
cancer_data.shape 


# In[4]:


# in this data 212 patents have breast cancer and 357 don't have cancer
cancer_data["diagnosis"].value_counts() 


# In[5]:


# check null data
cancer_data.isnull().sum() 


# In[6]:


# remove the unnecessary data
cancer_data =  cancer_data.drop(["Unnamed: 32"], axis=1) 
cancer_data = cancer_data.drop(["id"], axis=1)


# In[7]:


# featurs of each cells in numeric format 
from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder()


# In[8]:


# change diagnosis into a numeric form, diagnosis = 'malignant = 1', 'benign = 0'
cancer_data["diagnosis"] = le.fit_transform(cancer_data["diagnosis"]) 


# In[9]:


cancer_data


# In[10]:


cancer_data.columns


# In[11]:


# Information of cancer Dataframe
cancer_data.info


# In[12]:


# Numerical distribution of data
cancer_data.describe()


# In[13]:


# Paiplot of cancer dataframe
#sns.pairplot(cancer_data,hue="diagnosis")


# In[14]:


# pair plot of sample feature
sns.pairplot(cancer_data , hue = 'diagnosis', 
             vars = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean'] ) # ****** img 5 ***


# In[15]:


# Count the target class
sns.countplot(cancer_data['diagnosis'])


# In[16]:


# counter plot of feature mean radius
plt.figure(figsize = (20,8))
sns.countplot(cancer_data["radius_mean"])


# In[17]:


# heatmap of DataFrame
plt.figure(figsize=(16,9))
sns.heatmap(cancer_data)


# In[18]:


# Heatmap of a correlation matrix 
cancer_data.corr()


# In[19]:


# Heatmap of Correlation matrix of breast cancer DataFrame
plt.figure(figsize=(20,20))
sns.heatmap(cancer_data.corr(), annot = True, cmap ='coolwarm', linewidths=4)


# # Split DatFrame in Train and Test

# In[20]:


x = cancer_data.iloc[:,1:] 
y = cancer_data["diagnosis"] 


# In[21]:


# input variable
x


# In[22]:


# output variable
y


# In[23]:


# lets see important features but work on all data
from sklearn.feature_selection import SelectKBest
k_best = SelectKBest(k=20)


# In[24]:


k_best.fit(x,y)
k_best.scores_


# In[25]:


sc = pd.concat([pd.DataFrame(x.columns),pd.DataFrame(k_best.scores_)],axis=1)


# In[26]:


# these are the 20 important features
sc.columns = ["columns","score"]
sc.sort_values("score",ascending=False)


# In[27]:


# split the dataset into train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=2)
print(x.shape, x_train.shape, x_test.shape)


# # Feature scaling 

# In[28]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train_sc = sc.fit_transform(x_train)
x_test_sc = sc.transform(x_test)


# # Machine Learning Model Building

# In[29]:


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# # Support vector classifier

# In[30]:


# Support vector classifier
from sklearn.svm import SVC
svc_classifier = SVC()
svc_classifier.fit(x_train, y_train)
y_pred_scv = svc_classifier.predict(x_test)
accuracy_score(y_test, y_pred_scv)


# In[31]:


# Train with Standard scaled Data
svc_classifier2 = SVC()
svc_classifier2.fit(x_train_sc, y_train)
y_pred_svc_sc = svc_classifier2.predict(x_test_sc)
accuracy_score(y_test, y_pred_svc_sc)


# # Logistic Regression

# In[32]:


# Logistic Regression
from sklearn.linear_model import LogisticRegression
lr_classifier = LogisticRegression(random_state=5)
lr_classifier.fit(x_train, y_train)
y_pred_lr = lr_classifier.predict(x_test)
accuracy_score(y_test, y_pred_lr)


# In[33]:


# Train with Standard scaled Data
lr_classifier2 = LogisticRegression(random_state = 51)
lr_classifier2.fit(x_train_sc, y_train)
y_pred_lr_sc = lr_classifier.predict(x_test_sc)
accuracy_score(y_test, y_pred_lr_sc)


# # K – Nearest Neighbor Classifier

# In[34]:


# K – Nearest Neighbor Classifier
from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_neighbors = 5)
knn_classifier.fit(x_train, y_train)
y_pred_knn = knn_classifier.predict(x_test)
accuracy_score(y_test, y_pred_knn)


# In[35]:


# Train with Standard scaled Data
knn_classifier2 = KNeighborsClassifier(n_neighbors = 5)
knn_classifier2.fit(x_train_sc, y_train)
y_pred_knn_sc = knn_classifier.predict(x_test_sc)
accuracy_score(y_test, y_pred_knn_sc)


# # Naive Bayes Classifier

# In[36]:


# Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
nb_classifier = GaussianNB()
nb_classifier.fit(x_train, y_train)
y_pred_nb = nb_classifier.predict(x_test)
accuracy_score(y_test, y_pred_nb)


# In[37]:


# Train with Standard scaled Data
nb_classifier2 = GaussianNB()
nb_classifier2.fit(x_train_sc, y_train)
y_pred_nb_sc = nb_classifier2.predict(x_test_sc)
accuracy_score(y_test, y_pred_nb_sc)


# # Decision Tree Classifier

# In[38]:


# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 51)
dt_classifier.fit(x_train, y_train)
y_pred_dt = dt_classifier.predict(x_test)
accuracy_score(y_test, y_pred_dt)


# In[39]:


# Train with Standard scaled Data
dt_classifier2 = DecisionTreeClassifier(criterion = 'entropy', random_state = 51)
dt_classifier2.fit(x_train_sc, y_train)
y_pred_dt_sc = dt_classifier.predict(x_test_sc)
accuracy_score(y_test, y_pred_dt_sc)


# # Random Forest Classifier

# In[40]:


# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 51)
rf_classifier.fit(x_train, y_train)
y_pred_rf = rf_classifier.predict(x_test)
accuracy_score(y_test, y_pred_rf)


# In[41]:


# Train with Standard scaled Data
rf_classifier2 = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 51)
rf_classifier2.fit(x_train_sc, y_train)
y_pred_rf_sc = rf_classifier.predict(x_test_sc)
accuracy_score(y_test, y_pred_rf_sc)


# # Adaboost Classifier

# In[42]:


# Adaboost Classifier
from sklearn.ensemble import AdaBoostClassifier
adb_classifier = AdaBoostClassifier(DecisionTreeClassifier(criterion = 'entropy', random_state = 200),
                                    n_estimators=2000,
                                    learning_rate=0.1,
                                    random_state=1,)
adb_classifier.fit(x_train, y_train)
y_pred_adb = adb_classifier.predict(x_test)
accuracy_score(y_test, y_pred_adb)


# In[43]:


# Train with Standard scaled Data
adb_classifier2 = AdaBoostClassifier(DecisionTreeClassifier(criterion = 'entropy', random_state = 200),
                                    n_estimators=2000,
                                    learning_rate=0.1,
                                    random_state=1,)
adb_classifier2.fit(x_train_sc, y_train)
y_pred_adb_sc = adb_classifier2.predict(x_test_sc)
accuracy_score(y_test, y_pred_adb_sc)


# # XGBoost Classifier

# In[44]:


# XGBoost Classifier
from xgboost import XGBClassifier
xgb_classifier = XGBClassifier()
xgb_classifier.fit(x_train, y_train)
y_pred_xgb = xgb_classifier.predict(x_test)
accuracy_score(y_test, y_pred_xgb)


# In[45]:


# Train with Standard scaled Data
xgb_classifier2 = XGBClassifier()
xgb_classifier2.fit(x_train_sc, y_train)
y_pred_xgb_sc = xgb_classifier2.predict(x_test_sc)
accuracy_score(y_test, y_pred_xgb_sc)


# # Grid Search

# In[46]:


#from sklearn.model_selection import GridSearchCV 
import warnings
warnings.filterwarnings('ignore')
forest_params = [{'max_depth': list(range(10, 15)), 'max_features': list(range(0,14))}]
grid_search = GridSearchCV(rf_classifier,forest_params, cv = 10, scoring='accuracy')
grid_search.fit(x_train, y_train)
grid_search.best_estimator_


# In[47]:


print(grid_search.best_params_)
print(grid_search.best_score_)


# #  Confusion Matrix

# In[48]:


cm = confusion_matrix(y_test, y_pred_rf)
plt.title('Heatmap of Confusion Matrix', fontsize = 20)
sns.heatmap(cm, annot = True)
plt.show()


# The model is giving 0 type III error and it is best

# # Classification Report Of model

# In[49]:


print(classification_report(y_test,y_pred_rf))


# # Cross-validation of the ML model

# In[50]:


# Cross validation
from sklearn.model_selection import cross_val_score
cross_validation = cross_val_score(estimator = rf_classifier, X = x_train_sc,y = y_train, cv = 10)
print("Cross validation accuracy of Random Forest model = ", cross_validation)
print("\nCross validation mean accuracy of Random Forest model = ", cross_validation.mean())


# # Save Random Forest Classifier model using Pickel

# In[51]:


# Pickle
import pickle

# save model
pickle.dump(rf_classifier, open('breast_cancer_detector.pickle', 'wb'))

# load model
breast_cancer_detector_model = pickle.load(open('breast_cancer_detector.pickle', 'rb'))

# predict the output
y_pred = breast_cancer_detector_model.predict(x_test)

# confusion matrix
print('Confusion matrix of Random Forest model: \n',confusion_matrix(y_test, y_pred),'\n')

# show the accuracy
print('Accuracy of Random Forest model = ',accuracy_score(y_test, y_pred))


# # Building a Predictive System

# In[52]:


# Enter the value of tumor patent 
input_data = (13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259)

# change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for one datapoint
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = rf_classifier.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 1):
    print('The Breast cancer is Malignant')

else:
    print('The Breast Cancer is Benign')


# # Using Flask For Deployment 

# In[53]:


# import flask
from flask import Flask,render_template,request


# In[ ]:


app = Flask(__name__)
@app.route("/")
def web():
    return render_template("breast cancer.html")
@app.route("/info",methods=['POST','GET'])
def xyz():
    if request.method=='POST':
        
        radius_mean=float(request.form.get('radius_mean'))
        texture_mean=float(request.form.get('texture_mean'))
        perimeter_mean=float(request.form.get('perimeter_mean'))
        area_mean=float(request.form.get('area_mean'))
        smoothness_mean=float(request.form.get('smoothness_mean'))
        compactness_mean=float(request.form.get('compactness_mean'))
        concavity_mean=float(request.form.get('concavity_mean'))
        concave_points_mean=float(request.form.get('concave points_mean'))
        symmetry_mean=float(request.form.get('symmetry_mean'))
        fractal_dimension_mean=float(request.form.get('fractal_dimension_mean'))
        radius_se=float(request.form.get('radius_se'))
        texture_se=float(request.form.get('texture_se'))
        perimeter_se=float(request.form.get('perimeter_se'))
        area_se=float(request.form.get('area_se'))
        smoothness_se=float(request.form.get('smoothness_se'))
        compactness_se=float(request.form.get('compactness_se'))
        concavity_se=float(request.form.get('concavity_se'))
        concave_points_se=float(request.form.get('concave points_se'))
        symmetry_se=float(request.form.get('symmetry_se'))
        fractal_dimension_se=float(request.form.get('fractal_dimension_se'))
        radius_worst=float(request.form.get('radius_worst'))
        texture_worst=float(request.form.get('texture_worst'))
        perimeter_worst=float(request.form.get('perimeter_worst'))
        area_worst=float(request.form.get('area_worst'))
        smoothness_worst=float(request.form.get('smoothness_worst'))
        compactness_worst=float(request.form.get('compactness_worst'))
        concavity_worst=float(request.form.get('concavity_worst'))
        concave_points_worst=float(request.form.get('concave points_worst'))
        symmetry_worst=float(request.form.get('symmetry_worst'))
        fractal_dimension_worst=float(request.form.get('fractal_dimension_worst'))
        
        
        output = rf_classifier.predict([[radius_mean, texture_mean, perimeter_mean,
       area_mean, smoothness_mean, compactness_mean, concavity_mean,
       concave_points_mean, symmetry_mean, fractal_dimension_mean,
       radius_se, texture_se, perimeter_se, area_se, smoothness_se,
       compactness_se, concavity_se, concave_points_se,symmetry_se,
       fractal_dimension_se, radius_worst, texture_worst,
       perimeter_worst, area_worst, smoothness_worst,
       compactness_worst, concavity_worst, concave_points_worst,
       symmetry_worst, fractal_dimension_worst]])
        
        if output==1:
            result = " Patient Has Breast Cancer "
        else:
            result = " Patient Has No Breast Cancer "
        
        return render_template("breast cancer.html",prediction_text = result )

if __name__=='__main__':
    app.run()


# In[ ]:





# In[ ]:





# In[ ]:





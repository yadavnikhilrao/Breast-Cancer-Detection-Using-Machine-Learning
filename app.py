#!/usr/bin/env python
# coding: utf-8

# 
# # Breast Cancer Detection Using Machine Learning
# 

# -----------------------------------------------------------------------------------------------------------------------------------------

# In[1]:


import pandas as pd
cancer_data = pd.read_csv("data.csv")  


# In[2]:


cancer_data =  cancer_data.drop(["Unnamed: 32"], axis=1) 
cancer_data = cancer_data.drop(["id"], axis=1)


# In[3]:


from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder()
cancer_data["diagnosis"] = le.fit_transform(cancer_data["diagnosis"]) 


# In[4]:


x = cancer_data.iloc[:,1:] 
y = cancer_data["diagnosis"] 


# In[5]:


# split the dataset into train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=2)


# # Machine Learning Model Building

# In[6]:


from sklearn.metrics import accuracy_score


# In[7]:


# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 51)
rf_classifier.fit(x_train, y_train)
y_pred_rf = rf_classifier.predict(x_test)
accuracy_score(y_test, y_pred_rf)


# # Using Flask For Deployment 

# In[8]:


# import flask
from flask import Flask,render_template,request


# In[9]:


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





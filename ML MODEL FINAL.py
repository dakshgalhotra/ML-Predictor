#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report


# In[2]:


# Load the dataset
df_predictor = pd.read_csv('exp2.csv')


# In[3]:


df_predictor


# In[4]:


df_predictor.describe()


# In[7]:


df_predictor.isnull().value_counts()


# In[10]:


df_predictor['Activity'].value_counts()


# In[13]:


X = df_predictor[['Sleeping time', 'Which day is it', 'Last_Activity', 'Hours slept', 'Schedule']]
y = df_predictor['Activity']


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[19]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")


# In[32]:


import pandas as pd

# Assuming you have already trained the RandomForestClassifier and stored it in rf_classifier

# Create a new DataFrame with the user's input
new_data = pd.DataFrame(columns=['Sleeping time', 'Which day is it', 'Last_Activity', 'Hours slept', 'Schedule'])

# Assuming user inputs a comma-separated string
enter_resp = input('Enter Sleeping time [1-12], Day [0-1], Last activity[1-5], Hours slept[1-8], Schedule[1-6] (comma-separated):')
user_inputs = [int(x) for x in enter_resp.split(',')]
new_data.loc[0] = user_inputs

# Use the trained model to make predictions
prediction = rf_classifier.predict(new_data)
label_mapping = {
    1: 'Phone',
    2: 'Brush/Bath',
    3: 'Read',
    4: 'Drink water',
    5: 'Switch off AC',
    6: 'Make your bed',
    # Add more mappings as needed
}

# Use the trained model to make predictions
prediction = rf_classifier.predict(new_data)

# Map the numerical prediction to a meaningful label
predicted_label = label_mapping.get(prediction[0], 'Unknown Label')

print(f"The predicted result is: {predicted_label}")
print(f"The predicted result is: {prediction[0]}")


# In[ ]:





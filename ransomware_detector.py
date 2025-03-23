#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd

# Generate benign data
num_benign = 700
benign_data = {
    "cpu_usage": np.random.uniform(5, 50, num_benign),
    "disk_io": np.random.uniform(100, 500, num_benign),
    "network_activity": np.random.uniform(100, 2000, num_benign),
    "label": 0
}

# Generate ransomware data
num_ransomware = 300
ransomware_data = {
    "cpu_usage": np.random.uniform(70, 100, num_ransomware),
    "disk_io": np.random.uniform(5000, 10000, num_ransomware),
    "network_activity": np.random.uniform(3000, 5000, num_ransomware),
    "label": 1
}

# Combine datasets
benign_df = pd.DataFrame(benign_data)
ransomware_df = pd.DataFrame(ransomware_data)
df = pd.concat([benign_df, ransomware_df], ignore_index=True)

# Save dataset to CSV
df.to_csv("ransomware_dataset.csv", index=False)


# In[3]:


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[4]:


# Load dataset
data = pd.read_csv('ransomware_dataset.csv')

# Separate features and labels
X = data.drop('label', axis=1)  # All columns except the label
y = data['label']  # The label column

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[5]:


print(X_train, X_test, y_train, y_test)


# In[6]:


# Create the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)


# In[7]:


# Predict on the test set
y_pred = rf_model.predict(X_test)

# Evaluation Metrics
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))


# In[8]:


# Feature importance analysis
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("Feature Importances:\n", feature_importances)


# In[9]:


import joblib
joblib.dump(rf_model, "ransomware_detector.pkl")


# In[ ]:





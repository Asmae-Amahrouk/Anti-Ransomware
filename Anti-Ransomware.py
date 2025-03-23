#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import psutil
import joblib
import numpy as np
import time

# Load model
model = joblib.load('ransomware_detector.pkl')

while True:
    # Collect hardware metrics
    cpu_usage = psutil.cpu_percent(interval=1)
    disk_io = psutil.disk_io_counters().write_bytes
    network_out = psutil.net_io_counters().bytes_sent

    # Prepare data for the model
    features = np.array([cpu_usage, disk_io, network_out]).reshape(1, -1)

    # Predict
    prediction = model.predict(features)

    if prediction[0] == 1:
        print("ALERT: Potential ransomware detected!")
    else:
        print("System is normal.")

    # Sleep before next check
    time.sleep(5)


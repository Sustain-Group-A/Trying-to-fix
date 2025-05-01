import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn import metrics
import seaborn as sns


df = pd.read_csv('full_grain_data.csv')
df1 = pd.read_csv('predictions.csv')
true = df1['True']
predicted = df1['Predicted']
    
    # Extract features (band_1 to band_256) and labels (grain_type)
feature_cols = [f'band_{i+1}' for i in range(256)]
feature_cols1 = df.iloc[:,2:]
X = df[feature_cols].values  # Shape: (n_samples, 256)
X1 = feature_cols1.values
    #y = df['grain_type'].values  # Shape: (n_samples,)
    #dflabel = df['grain_type'].apply(lambda x:Path(x).stem.split("-")[0])
dflabel1 = df['grain_type'].str.extract(r"([^/]+)(?=-\d+\.hdr$)", expand=False)
print(X.shape)
print(dflabel1.shape)

    # Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(dflabel1)
print(len(y_encoded))
predicted = label_encoder.fit_transform(predicted)
true = label_encoder.fit_transform(true)
    
    # Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X1)
    #print(X_scaled)

    # Split data: 80% train, 10% validation, 10% test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)


model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(5731,90,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (2,2), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))

#model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
#histroy = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

clf = SVC(gamma=0.001, C=10000, kernel='rbf')
clf.fit(X_train, y_train)

pred = clf.predict(X_test)
print("Evaulating the Model:")
print('SVC Accuracy:', metrics.accuracy_score(y_test,pred))

figure = plt.figure()
plt.title("SVC Scatter Graph")
plt.xlabel("Actual Labels")
plt.ylabel("Predicted Lables")
plt.scatter(y_test, pred)
plt.show()

fig = plt.figure()
plt.title("SVC Pie chart")
plt.subplot(1,2,1)
plt.title("Acutal values")
plt.pie(y_test)
plt.subplot(1,2,2)
plt.title("Predicted Values")
plt.pie(pred)
plt.show()

fig3 = plt.figure()
plt.title("SVC Confustion Matrix")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
matrix = metrics.confusion_matrix(y_test, pred)
sns.heatmap(matrix, annot=True)
plt.show()
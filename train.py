import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

df=pd.read_csv("heart_disease.csv")
df["target"] = (df["target"] > 0).astype(int)

x = df.drop(columns="target")
y = df["target"]
x_train, x_test, y_train, y_test = train_test_split(x, y, 
    test_size=0.2, stratify=y, random_state=42)

model = LogisticRegression(max_iter=1000)

rows=[]
confusion=[]
    
crossValidation_auc = cross_val_score(model, x_train, y_train, cv=5, scoring="roc_auc")
model.fit(x_train,y_train)
pred = model.predict(x_test)
accuracy = (pred == y_test).mean()

true_positive = ((pred ==1)&(y_test==1)).sum()
false_negative = ((pred ==0)&(y_test==1)).sum()
false_positive = ((pred ==1)&(y_test==0)).sum()
true_negative = ((pred ==0)&(y_test==0)).sum()








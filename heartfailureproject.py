#sex-Gender of patient male=1,female=0
#age-age of patient
#Diabates-0=no,1=yes
#anemia-0=no,1=yes
#high blood pressure-0=no,1=yes
#smoking-0=no,1=yes
#death event-0=no,1=yes

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from plotly import graph_objects
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

heart=pd.read_csv("heartdataset.csv")
print(heart)
print(heart.head())
print(heart.describe())
print(heart.isnull())

#piechart
names=["No Diabetes","Diabetes"]
diabetesyes=heart[heart['diabetes']==1]
diabetesno=heart[heart['diabetes']==0]
data=[len(diabetesyes), len(diabetesno)]          
figure=go.Figure(data=[go.Pie(labels=names,values=data,hole=.6)])#hole denote the middle hole of the piechart
figure.update_layout(title_text="Analysis on Diabeters" )
figure.show()


figure=px.pie(heart,values="diabetes",names="DEATH_EVENT",title="death analytics")
figure.show()


names=["heart attack", "no heart attack risk"]
heartattackyes=heart[heart['DEATH_EVENT']==1]
heartattackno=heart[heart['DEATH_EVENT']==0]
data=[len(heartattackyes),len(heartattackno)]
figure=go.Figure(data=[go.Pie(labels=names,values=data,hole=.1)])
figure.update_layout(
    title_text="Heart attack assumptions"
)
figure.show()


#heat map

plt.figure(figsize=(10,10))
sns.heatmap(heart.corr(),vmin=-1,cmap="coolwarm",annot=True);
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
Feature=['time','ejection_fraction','serum_creatinine']
x=heart[Feature]
y=heart["DEATH_EVENT"]
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.linear_model import LogisticRegression
log_re=LogisticRegression()
log_re.fit(xtrain,ytrain)
log_re_pred=log_re.predict(xtest)
log_acc=accuracy_score(ytest,log_re_pred)
print("Logistic Accuracy Score: ","{:.2f}%".format(100*log_acc))
from mlxtend.plotting import plot_confusion_matrix
cm = confusion_matrix(ytest, log_re_pred)
plt.figure()
plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
plt.title("Logistic Regerssion  - Confusion Matrix")
plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.show()








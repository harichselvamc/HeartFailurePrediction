# #sex-Gender of patient male=1,female=0
# #age-age of patient
# #Diabates-0=no,1=yes
# #anemia-0=no,1=yes
# #high blood pressure-0=no,1=yes
# #smoking-0=no,1=yes
# #death event-0=no,1=yes

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.graph_objs as go
# from plotly import graph_objects
# import plotly.express as px
# import warnings
# warnings.filterwarnings("ignore")

# heart=pd.read_csv("heartdataset.csv")
# print(heart)
# print(heart.head())
# print(heart.describe())
# print(heart.isnull())

# #piechart
# names=["No Diabetes","Diabetes"]
# diabetesyes=heart[heart['diabetes']==1]
# diabetesno=heart[heart['diabetes']==0]
# data=[len(diabetesyes), len(diabetesno)]          
# figure=go.Figure(data=[go.Pie(labels=names,values=data,hole=.6)])#hole denote the middle hole of the piechart
# figure.update_layout(title_text="Analysis on Diabeters" )
# figure.show()


# figure=px.pie(heart,values="diabetes",names="DEATH_EVENT",title="death analytics")
# figure.show()


# names=["heart attack", "no heart attack risk"]
# heartattackyes=heart[heart['DEATH_EVENT']==1]
# heartattackno=heart[heart['DEATH_EVENT']==0]
# data=[len(heartattackyes),len(heartattackno)]
# figure=go.Figure(data=[go.Pie(labels=names,values=data,hole=.1)])
# figure.update_layout(
#     title_text="Heart attack assumptions"
# )
# figure.show()


# #heat map

# plt.figure(figsize=(10,10))
# sns.heatmap(heart.corr(),vmin=-1,cmap="coolwarm",annot=True);
# plt.show()


# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix,accuracy_score
# Feature=['time','ejection_fraction','serum_creatinine']
# x=heart[Feature]
# y=heart["DEATH_EVENT"]
# xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=2)
# from sklearn.linear_model import LogisticRegression
# log_re=LogisticRegression()
# log_re.fit(xtrain,ytrain)
# log_re_pred=log_re.predict(xtest)
# log_acc=accuracy_score(ytest,log_re_pred)
# print("Logistic Accuracy Score: ","{:.2f}%".format(100*log_acc))
# from mlxtend.plotting import plot_confusion_matrix
# cm = confusion_matrix(ytest, log_re_pred)
# plt.figure()
# plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
# plt.title("Logistic Regerssion  - Confusion Matrix")
# plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
# plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
# plt.show()




# #sex-Gender of patient male=1,female=0
# #age-age of patient
# #Diabates-0=no,1=yes
# #anemia-0=no,1=yes
# #high blood pressure-0=no,1=yes
# #smoking-0=no,1=yes
# #death event-0=no,1=yes

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.graph_objs as go
# from plotly import graph_objects
# import plotly.express as px
# import warnings
# warnings.filterwarnings("ignore")

# heart=pd.read_csv("heartdataset.csv")
# print(heart)
# print(heart.head())
# print(heart.describe())
# print(heart.isnull())

# #piechart
# names=["No Diabetes","Diabetes"]
# diabetesyes=heart[heart['diabetes']==1]
# diabetesno=heart[heart['diabetes']==0]
# data=[len(diabetesyes), len(diabetesno)]          
# figure=go.Figure(data=[go.Pie(labels=names,values=data,hole=.6)])#hole denote the middle hole of the piechart
# figure.update_layout(title_text="Analysis on Diabeters" )
# figure.show()


# figure=px.pie(heart,values="diabetes",names="DEATH_EVENT",title="death analytics")
# figure.show()


# names=["heart attack", "no heart attack risk"]
# heartattackyes=heart[heart['DEATH_EVENT']==1]
# heartattackno=heart[heart['DEATH_EVENT']==0]
# data=[len(heartattackyes),len(heartattackno)]
# figure=go.Figure(data=[go.Pie(labels=names,values=data,hole=.1)])
# figure.update_layout(
#     title_text="Heart attack assumptions"
# )
# figure.show()


# #heat map

# plt.figure(figsize=(10,10))
# sns.heatmap(heart.corr(),vmin=-1,cmap="coolwarm",annot=True);
# plt.show()


# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix,accuracy_score
# Feature=['time','ejection_fraction','serum_creatinine']
# x=heart[Feature]
# y=heart["DEATH_EVENT"]
# xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=2)
# from sklearn.linear_model import LogisticRegression
# log_re=LogisticRegression()
# log_re.fit(xtrain,ytrain)
# log_re_pred=log_re.predict(xtest)
# log_acc=accuracy_score(ytest,log_re_pred)
# print("Logistic Accuracy Score: ","{:.2f}%".format(100*log_acc))
# from mlxtend.plotting import plot_confusion_matrix
# cm = confusion_matrix(ytest, log_re_pred)
# plt.figure()
# plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
# plt.title("Logistic Regerssion  - Confusion Matrix")
# plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
# plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
# plt.show()







# import streamlit as st
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.graph_objs as go
# from plotly import graph_objects
# import plotly.express as px
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix, accuracy_score
# from sklearn.linear_model import LogisticRegression
# from mlxtend.plotting import plot_confusion_matrix
# st.set_option('deprecation.showPyplotGlobalUse', False)
# # Load dataset
# heart = pd.read_csv("heartdataset.csv")

# # Streamlit app
# st.title("Heart Disease Analysis")

# # Display the dataset
# st.subheader("Dataset")
# st.write(heart)

# # Display basic statistics
# st.subheader("Basic Statistics")
# st.write(heart.describe())

# # Check for missing values
# st.subheader("Missing Values")
# st.write(heart.isnull())

# # Pie chart for Diabetes analysis
# st.subheader("Diabetes Analysis")
# names = ["No Diabetes", "Diabetes"]
# diabetes_yes = heart[heart['diabetes'] == 1]
# diabetes_no = heart[heart['diabetes'] == 0]
# data = [len(diabetes_yes), len(diabetes_no)]
# fig = go.Figure(data=[go.Pie(labels=names, values=data, hole=.6)])
# fig.update_layout(title_text="Analysis on Diabetes")
# st.plotly_chart(fig)

# # Pie chart for Death analytics
# st.subheader("Death Analytics")
# fig = px.pie(heart, values="diabetes", names="DEATH_EVENT", title="Death Analytics")
# st.plotly_chart(fig)

# # Pie chart for Heart attack assumptions
# st.subheader("Heart Attack Assumptions")
# names = ["Heart Attack", "No Heart Attack Risk"]
# heart_attack_yes = heart[heart['DEATH_EVENT'] == 1]
# heart_attack_no = heart[heart['DEATH_EVENT'] == 0]
# data = [len(heart_attack_yes), len(heart_attack_no)]
# fig = go.Figure(data=[go.Pie(labels=names, values=data, hole=.1)])
# fig.update_layout(title_text="Heart Attack Assumptions")
# st.plotly_chart(fig)

# # Heatmap
# st.subheader("Correlation Heatmap")
# plt.figure(figsize=(10, 10))
# sns.heatmap(heart.corr(), vmin=-1, cmap="coolwarm", annot=True)
# st.pyplot()

# # Machine Learning Model
# st.subheader("Machine Learning Model - Logistic Regression")
# feature = ['time', 'ejection_fraction', 'serum_creatinine']
# X = heart[feature]
# y = heart["DEATH_EVENT"]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# log_re = LogisticRegression()
# log_re.fit(X_train, y_train)
# log_re_pred = log_re.predict(X_test)
# log_acc = accuracy_score(y_test, log_re_pred)

# st.write("Logistic Accuracy Score: {:.2f}%".format(100 * log_acc))









import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# Load dataset
heart = pd.read_csv("heartdataset.csv")

# Streamlit app
st.title("Heart Disease Analysis")

# Display the dataset
st.subheader("Dataset")
st.write(heart)

# Display basic statistics
st.subheader("Basic Statistics")
st.write(heart.describe())

# Check for missing values
st.subheader("Missing Values")
st.write(heart.isnull())

# Pie chart for Diabetes analysis
st.subheader("Diabetes Analysis")
names = ["No Diabetes", "Diabetes"]
diabetes_yes = heart[heart['diabetes'] == 1]
diabetes_no = heart[heart['diabetes'] == 0]
data = [len(diabetes_yes), len(diabetes_no)]
fig = go.Figure(data=[go.Pie(labels=names, values=data, hole=.6)])
fig.update_layout(title_text="Analysis on Diabetes")
st.plotly_chart(fig)

# Pie chart for Death analytics
st.subheader("Death Analytics")
fig = px.pie(heart, values="diabetes", names="DEATH_EVENT", title="Death Analytics")
st.plotly_chart(fig)

# Pie chart for Heart attack assumptions
st.subheader("Heart Attack Assumptions")
names = ["Heart Attack", "No Heart Attack Risk"]
heart_attack_yes = heart[heart['DEATH_EVENT'] == 1]
heart_attack_no = heart[heart['DEATH_EVENT'] == 0]
data = [len(heart_attack_yes), len(heart_attack_no)]
fig = go.Figure(data=[go.Pie(labels=names, values=data, hole=.1)])
fig.update_layout(title_text="Heart Attack Assumptions")
st.plotly_chart(fig)

# Heatmap
st.subheader("Correlation Heatmap")
heatmap_fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(heart.corr(), vmin=-1, cmap="coolwarm", annot=True, ax=ax)
st.write(heatmap_fig)

# Machine Learning Model
st.subheader("Machine Learning Model - Logistic Regression")
feature = ['time', 'ejection_fraction', 'serum_creatinine']
X = heart[feature]
y = heart["DEATH_EVENT"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

log_re = LogisticRegression()
log_re.fit(X_train, y_train)
log_re_pred = log_re.predict(X_test)
log_acc = accuracy_score(y_test, log_re_pred)

st.write("Logistic Accuracy Score: {:.2f}%".format(100 * log_acc))

# User Input for Prediction
st.sidebar.subheader("Predictions")
st.sidebar.text("Enter values for prediction:")

# Input for time
time = st.sidebar.slider("Time", float(heart['time'].min()), float(heart['time'].max()), float(heart['time'].mean()))

# Input for ejection_fraction
ejection_fraction = st.sidebar.slider("Ejection Fraction", float(heart['ejection_fraction'].min()), float(heart['ejection_fraction'].max()), float(heart['ejection_fraction'].mean()))

# Input for serum_creatinine
serum_creatinine = st.sidebar.slider("Serum Creatinine", float(heart['serum_creatinine'].min()), float(heart['serum_creatinine'].max()), float(heart['serum_creatinine'].mean()))

# Button to trigger prediction
if st.sidebar.button("Predict"):
    # Make prediction using the model
    input_data = np.array([[time, ejection_fraction, serum_creatinine]])
    prediction = log_re.predict(input_data)

    # Display the prediction
    st.sidebar.subheader("Prediction Result:")
    result_text = "Heart Attack Risk: {}".format("Yes" if prediction[0] == 1 else "No")
    st.sidebar.text(result_text)



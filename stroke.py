import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import operator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.metrics import mean_squared_error,accuracy_score
#from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.preprocessing import StandardScaler

#cross_entropy
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))
def cross_entropy(x,y):
    x_softmax = [softmax(x[i]) for i in range(len(x))]
    x_log = [np.log(x_softmax[i][y[i]]) for i in range(len(y))]
    loss = -np.sum(x_log)/len(y)
    return loss
#main
url = 'd:\AI\FinalAssignment\healthcare-dataset-stroke-data.csv'
Data = pd.read_csv(url)
Data = Data.sort_values(['id'],ascending=True)
All_Data = []
All_Data = pd.DataFrame(All_Data)
#print(Data)

#答案
answer = Data.iloc[:,[11]]

#gender
Data_gender = Data.iloc[:,[1]]
Data_gender_OHE = pd.get_dummies(Data_gender)
#print(Data_gender.describe())
#print(Data['gender'].unique())
#print(Data_gender_OHE)
All_Data = pd.concat([All_Data,Data_gender_OHE],axis=1,sort=False)

#age
Data_age = Data.iloc[:,[2]]
#print(Data_age)
Data_age_LE = []
num = pd.to_numeric(Data_age.squeeze(),errors='coerce')
Q1 = num.quantile(0.25)
Q2 = num.quantile(0.5)
Q3 = num.quantile(0.75)
for j in range(Data_age.size):
    match_num = num[j]
    match match_num:
        case match_num if match_num < Q1:
            Data_age_LE.append(1)
        case match_num if operator.and_(match_num >= Q1,match_num < Q2) : 
            Data_age_LE.append(2)
        case match_num if operator.and_(match_num >= Q2,match_num < Q3): 
            Data_age_LE.append(3)
        case numatch_numm if match_num >= Q3:
            Data_age_LE.append(4)
Data_age_LE = pd.DataFrame(Data_age_LE)
#print(Data_age_LE)
All_Data = pd.concat([All_Data,Data_age_LE],axis=1,sort=False)

#ever_merry
Data_ever_merry = Data.iloc[:,[5]]
Data_ever_merry_convert = []
#print(Data_ever_merry.iloc[0,0])
for j in range(Data_ever_merry.size):
    ju = Data_ever_merry.iloc[j,0]
    if ju =='Yes':
        Data_ever_merry_convert.append(1)
    else:
        Data_ever_merry_convert.append(0)
Data_ever_merry_convert = pd.DataFrame(Data_ever_merry_convert)
All_Data = pd.concat([All_Data,Data_ever_merry_convert],axis=1,sort=False)

#work_type
Data_work_type = Data.iloc[:,[6]]
#print(Data_work_type['work_type'].unique())
Data_work_type_OHE = pd.get_dummies(Data_work_type)
#print(Data_work_type_OHE)
All_Data = pd.concat([All_Data,Data_work_type_OHE],axis=1,sort=False)

#Residence
Data_residence = Data.iloc[:,[7]]
Data_residence_convert = []
print(Data['Residence_type'].unique())
for j in range(Data_residence.size):
    ju = Data_residence.iloc[j,0]
    if ju == 'Urban':
        Data_residence_convert.append(1)
    elif ju =='Rural':
        Data_residence_convert.append(0)
Data_residence_convert = pd.DataFrame(Data_residence_convert)
All_Data = pd.concat([All_Data,Data_residence_convert],axis=1,sort=False)

#avg_glucose_level
Data_avg_glucose_level = Data.iloc[:,[8]]
Data_avg_glucose_level_LE = []
num = pd.to_numeric(Data_avg_glucose_level.squeeze(),errors='coerce')
Q1 = num.quantile(0.25)
Q2 = num.quantile(0.5)
Q3 = num.quantile(0.75)
for j in range(Data_avg_glucose_level.size):
    match_num = num[j]
    match match_num:
        case match_num if match_num < Q1:
            Data_avg_glucose_level_LE.append(1)
        case match_num if operator.and_(match_num >= Q1,match_num < Q2) : 
            Data_avg_glucose_level_LE.append(2)
        case match_num if operator.and_(match_num >= Q2,match_num < Q3): 
            Data_avg_glucose_level_LE.append(3)
        case numatch_numm if match_num >= Q3:
            Data_avg_glucose_level_LE.append(4)
Data_avg_glucose_level_LE = pd.DataFrame(Data_avg_glucose_level_LE)
All_Data = pd.concat([All_Data,Data_avg_glucose_level_LE],axis=1,sort=False)

#bmi
Data_bmi = Data.iloc[:,[9]]
#print(Data['bmi'].describe())
#print(Data['bmi'].isnull())
Data_mean_bmi = Data.iloc[:,[1,9]]
#print(Data_mean_bmi)
for i in range(len(Data_mean_bmi)):
    if Data_mean_bmi['bmi'][i] > 50:
        Data_mean_bmi['bmi'][i] = 0

mean_bmi = Data_mean_bmi[['gender','bmi']].groupby('gender').mean()

mapping = {'Female':27.943562,'Male':28.051914,'Other':22.40000}

for i in range(Data_bmi.iloc[:,0].size):
    if Data_bmi['bmi'].isnull()[i]:
        Data_bmi.loc[i,'bmi'] = mapping.get(Data_mean_bmi['gender'][i])
All_Data = pd.concat([All_Data,Data_bmi],axis=1,sort=False)

#All_Data = pd.concat([All_Data,answer],axis=1,sort=False)
answer = answer.to_numpy()
All_Data = All_Data.to_numpy()
#print(All_Data)
X_train, X_test, y_train, y_test = train_test_split(All_Data, answer, test_size=0.3, random_state=42)

LR = LogisticRegression()
LR_train = LR.fit(X_train,y_train)
print(LR_train.score(X_train,y_train))

LR_test = LR.fit(X_test,y_test)
print(LR_test.score(X_test,y_test))

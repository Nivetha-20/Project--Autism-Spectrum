import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

import pickle
import warnings
warnings.filterwarnings('ignore')
data=pd.read_csv("autism-screening.csv")
print(data)

data['gender'] = data['gender'].replace(np.nan, 0)
data['jundice'] = data['jundice'].replace(np.nan, 0)
data['austim'] = data['austim'].replace(np.nan, 0)
data['used_app_before'] = data['used_app_before'].replace(np.nan, 0)
data['relation'] = data['relation'].replace(np.nan, 0)
data['ethnicity'] = data['ethnicity'].replace(np.nan, 0)
data['contry_of_res'] = data['contry_of_res'].replace(np.nan, 0)



check_nan1 = data['gender'].isnull().values.any()
check_nan2 = data['jundice'].isnull().values.any()
check_nan3 = data['austim'].isnull().values.any()
check_nan4 = data['used_app_before'].isnull().values.any()
check_nan5 = data['relation'].isnull().values.any()
check_nan6 = data['Class/ASD'].isnull().values.any()


print(check_nan1)
print(check_nan2)
print(check_nan3)
print(check_nan4)
print(check_nan5)
print(check_nan6)


data.dropna(inplace=True)


le=LabelEncoder()
data['gender'] = le.fit_transform(data['gender'])
##data['ethnicity'] = le.fit_transform(data['ethnicity'])
data['jundice'] = le.fit_transform(data['jundice'])
data['austim'] = le.fit_transform(data['austim'])
data['used_app_before'] = le.fit_transform(data['used_app_before'])
data['relation'] = data['relation'].astype(str)
data['relation'] = le.fit_transform(data['relation'])
data['Class/ASD'] = le.fit_transform(data['Class/ASD'])
data['ethnicity'] = data['ethnicity'].astype(str)
data['ethnicity'] = le.fit_transform(data['ethnicity'])
data['contry_of_res'] = data['contry_of_res'].astype(str)
data['contry_of_res'] = le.fit_transform(data['contry_of_res'])


X=data.drop(['Class/ASD','A1_Score','A2_Score','A3_Score','A4_Score','A5_Score','A6_Score','A7_Score','A8_Score','age_desc','A9_Score','A10_Score','age_desc','sum_missing_rowWise'],axis=1)
print(X)
Y=data['Class/ASD']
print(Y)

##To check the number of samples and features in your data
print("X shape:", X.shape)
print("y shape:", Y.shape)

##To check the mean, standard deviation, minimum, maximum, and quartile values of each feature.
print("Describe",X.describe())

x_train,x_test,y_train,y_test = train_test_split(X,Y,shuffle=True,test_size=0.3, random_state=0)


NB = GaussianNB()
NB.fit(x_train, y_train)  #train the data
y_pred=NB.predict(x_test)
##print(y_pred)
##print(y_test)
print('Naive Bayes ACCURACY is', accuracy_score(y_test,y_pred)*100)


##filename = 'model.pkl'
##pickle.dump(NB, open(filename, 'wb'))



#1/15/18 
from keras.models import Sequential 
from keras.layers import Dense,Activation
import numpy as np 
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd 
#vorpy muscles
vorpy=pd.read_csv("Vorp final 2017 (version 1).csv")
vorpy.info() #miami beach rentals?
vorpy.shape 
#covert pandas to numpy
vorpy_np=vorpy.values 
vorpy_np.shape #277,34
vorpy.describe() #1.46 
#create a classification variable (custy returns)
def vorp_classify(y):
    if y>1.46:
        return 1
    if y<1.46:
        return 0

def power_conf(z):
    if z=="Big East":
        return 1
    if z=="SEC":
        return 2
    if z=="Big Ten":
        return 3
    if z=="Pac-12":
        return 4
    if z=="Big 12":
        return 5
    if z=="Pac-10":
        return 4
    if z=="ACC":
        return 6
    else:
        return 0 

vorpy['vorp_grade']=vorpy['VORP'].apply(vorp_classify)
vorpy['conf_grade']=vorpy['Conf'].apply(power_conf)
#keras setup (miami beach, north shore)
#donovan mitchell (lighting up from the outside?)
vorpy_sub=vorpy[["conf_grade","Body.Fat..","Hand.Length","Hand.Width",
"Height.wo.shoes","Standing.Reach.Height.w.shoes","Weight","Wingspan",
"Lane.Agiility","three.quarter.sprint","standing.vertical","max.vertical.leap",
"G","MP","FGA","X2P","X2PA","X3P","X3PA","FT","FTA","TRB","AST","TOV","PF","PTS","vorp_grade"]]
vorpy_sub1=vorpy_sub.dropna(how="any")
vorpy_sub2=vorpy_sub1.values 
type(vorpy_sub2)
vorpy_sub2.shape 
#split into features and response variables
X_predictor=vorpy_sub2[:,0:25].astype(float)
y_response=vorpy_sub2[:,26]
#split the data into test and train sets
import sklearn
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_predictor,y_response,test_size=0.3,random_state=73)
X_train.shape 
X_test.shape
## reshape X and y 
X_train=X_train.reshape(193,25)
X_test=X_test.reshape(84,25)
#convert class vectors to binary class matrices
from keras.utils import np_utils 
y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)
y_test.shape 
y_train.shape 

def model_keras():
    model=Sequential()
    model.add(Dense(25,input_dim=193,kernel_initializer='normal',activation='relu'))
    model.add(Dense(1,kernel_initializer='normal',activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model 

#model evaluation
est=KerasClassifier(build_fn=model_keras,nb_epoch=100,batch_size=5)
kfold_x=StratifiedKFold(n_splits=10,shuffle=True)
results_model=cross_val_score(est,X_predictor,y_response,cv=kfold_x)

#build another model?************
model=Sequential()
model.add(Dense(25,input_dim=193))
model.add(Dense(1,input_dim=193))
model.add(Activation('relu'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(X_test,y_test,epochs=2,batch_size=5)

####SVC playground?***********************
X=np.array([[1,2],[3,4],[4,3],[5,7]])
X.shape #(4,2)
y=[1,1,1,0]
#define classifier
from sklearn import svm
clf=svm.SVC(kernel='linear',C=1.0)
clf.fit(X,y)
#empeza predicting
clf.predict([[0.99,0.99]])
#visualize data
w=clf.coef_[0]
a=-w[0]/w[1]

##svm on the nba draft? 
from sklearn import svm
C=1.0 #svm regularization parameter 
svc=svm.SVC(kernel='linear',C=C,gamma=1).fit(X_train,y_train)
svc_pred=svc.predict(y_test)

##random forest model***** 1/15/17***
X_train,X_test,y_train,y_test=train_test_split(X_predictor,y_response,test_size=0.3,random_state=735)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score 
rf_vorp=RandomForestClassifier(n_estimators=145,oob_score=True,random_state=4224)
rf_vorp.fit(X_train,y_train)
predict_rf_vorp=rf_vorp.predict(X_test)
acc_score_rf=accuracy_score(y_test,predict_rf_vorp)
acc_score_rf #64.3% accuracy
#feature importance?
feature_importance=rf_vorp.feature_importances_
feature_importance 
indices=np.argsort(feature_importance)
indices

#vorpy_sub (just how good is cauley stein?)
vorpy_sub.info()
vorpy.groupby(['conf_grade']).mean()
vorpy.groupby(['Power']).mean()
vorpy.groupby(['Power']).var()
vorpy.groupby(['Power']).count()
vorpy_sub.columns[21]
vorpy.ix[5]




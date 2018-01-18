vorpy=pd.read_csv("Vorp final 2017 (version 1).csv")
vorpy.info() #miami beach rentals?
vorpy.shape 
#convert pandas to numpy
vorpy_np=vorpy.values 
vorpy_np.shape #277,34
vorpy.describe() #1.46 
#create a classification variable =
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
    model.add(Dense(25,input_dim=25,kernel_initializer='normal',activation='relu'))
    model.add(Dense(1,kernel_initializer='normal',activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model 

#model evaluation
est=KerasClassifier(build_fn=model_keras,nb_epoch=100,batch_size=5)
kfold_x=StratifiedKFold(n_splits=10,shuffle=True)
results_model=cross_val_score(est,X_predictor,y_response,cv=kfold_x)
results_model.mean() #63.1% accuracy 

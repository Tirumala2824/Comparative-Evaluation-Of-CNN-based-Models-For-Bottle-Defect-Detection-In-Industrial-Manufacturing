from PIL import Image 
import numpy as np 
import os 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


data='D://python//Flask-2//Bottle_Fault_Detection//images'

filepath=[]
labels=[]

for folder in os.listdir(data):
    file_path=os.path.join(data,folder)
    if os.path.exists(file_path):
        for file in os.listdir(file_path):
            # print(os.path.join(file_path,file))

            filepath.append(os.path.join(file_path,file))
            labels.append(folder)



def dataprocessing(filepath):
    features=[]
    for file in filepath:
        image=Image.open(file,'r')
        image=image.resize((128,128))
        img=np.array(image)
        img=img.flatten()
        features.append(img)
    return features
X=dataprocessing(filepath)
labelss=LabelEncoder()
y=labelss.fit_transform(labels)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

print('='*50)
print('\t Random Forest Classification ')
#Random Forest Accuracy 
n_esti=[i*10 for i in range(1,11)]
rand_accuracy=[]
rand_loss=[]
for i in n_esti:
    random1=RandomForestClassifier(n_estimators=i)
    random1.fit(X_train,y_train)
    y_pred=random1.predict(X_test)
    accuracy=accuracy_score(y_pred,y_test)
    rand_accuracy.append(accuracy)
    mse_Random=mean_squared_error(y_pred,y_test)
    rand_loss.append(mse_Random)
    print(f'Accuracy of a Random Forest Classifier in {i} estimators : {accuracy}')
    print(f'MSE of a Random Forest Classifier in {i} estimators :{mse_Random}')
random_output={
    'N_estimators':n_esti,
    'accuracy':rand_accuracy,
    'losses':rand_loss,
}
print('='*50)
#Decision Tree Accuracy 
print('\t Decison Tree Classifier ')
max_depth=[3,5,7,9]
decision_accuracy=[]
decision_loss=[]
for i in max_depth:
    decision=DecisionTreeClassifier(max_depth=i)
    decision.fit(X_train,y_train)
    y_pred1=decision.predict(X_test)
    accuracy1=accuracy_score(y_pred1,y_test)
    decision_accuracy.append(accuracy1)
    mse_Decision=mean_squared_error(y_pred1,y_test)
    decision_loss.append(mse_Decision)
    print(f'Accuracy of a Decision Tree Classifier in {i} max depth : {accuracy1}')
    print(f'Mse of a Decision Tree Classifier  in {i} max depth: {mse_Decision}')
decision_output={
    'max_depth':max_depth,
    'accuracy':decision_accuracy,
    'loss':decision_loss,
}
print('='*50)
print(f'\t Logistic Regression')
# Logistic regression accuracy
max_iteration=[i*10 for i in range(1,11)]
log_accuracy=[]
log_loss=[]
for i in max_iteration:
    log1=LogisticRegression(max_iter=i)
    log1.fit(X_train,y_train)
    y_pred2=log1.predict(X_test)
    accuracy2=accuracy_score(y_pred2,y_test)
    log_accuracy.append(accuracy2)
    print(f'Accuracy for a Logistic Regression in {i} iterations: {accuracy2}')
    mse_Logistic=mean_squared_error(y_pred2,y_test)
    log_loss.append(mse_Logistic)
    print(f'Mean Squared Error for a Logistic Regression in {i} iterations: {mse_Logistic}')
logistic_output={
    'iterations':max_iteration,
    'accuracy':log_accuracy,
    'loss':log_loss
}
print('='*50)
print('\t Support Vector Machine')
# SVM for Accuracy 
svm_accuracy=[]
svm_loss=[]
for i in max_iteration:
    log2=SVC(max_iter=i)
    log2.fit(X_train,y_train)
    y_pred3=log2.predict(X_test)
    accuracy3=accuracy_score(y_pred3,y_test)
    svm_accuracy.append(accuracy3)
    print(f'Accuracy for a SVM in {i} iterations : {accuracy3}')
    mse_Svm=mean_squared_error(y_pred2,y_test)
    svm_loss.append(mse_Svm)
    print(f'Mean Squared Error for a SVM in {i} iterations: {mse_Svm}')

svm_output={
    'max_iteration':max_iteration,
    'accuracy':svm_accuracy,
    'loss':svm_loss
}


from icecream import ic

ic(random_output)
ic(decision_output)
ic(logistic_output)
ic(svm_output)
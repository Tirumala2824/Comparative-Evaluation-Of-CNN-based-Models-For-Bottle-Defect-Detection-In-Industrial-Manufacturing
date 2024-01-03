from Algorithm_accuracy import random_output,decision_output,logistic_output,svm_output
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

cnn_history=pd.read_csv('D://python//Flask-2//Ganesh_Fault_Detection//Comparsion_Accuracy_Train_and_Test//CNN_metrics.csv')

print(cnn_history.loc[24::25])

# CNN and Random Forest accuracy visualiztion 

plt.plot(cnn_history.loc[24::25,['Epoch']],cnn_history.loc[24::25,['Val_Accuracy']],label='CNN Accuracy')
plt.plot(random_output['N_estimators'],random_output['accuracy'],label='Random Forest Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy comparison between Cnn and Random Forest Algorithm')
plt.legend()
plt.show()

# CNN and Random Forest Loss visualization 

plt.plot(cnn_history.loc[24::25,['Epoch']],cnn_history.loc[24::25,['Val_Loss']],label='CNN Loss')
plt.plot(random_output['N_estimators'],random_output['losses'],label='Random Forest Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss comparison between Cnn and Random Forest Algorithm')
plt.legend()
plt.show()

# Cnn and Logistic Regression accuracy algorithm 

plt.plot(cnn_history.loc[24::25,['Epoch']],cnn_history.loc[24::25,['Val_Accuracy']],label='CNN Accuracy')
plt.plot(logistic_output['iterations'],logistic_output['accuracy'],label='Logistic Regression Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy comparison between Cnn and Logistic Regression Algorithm')
plt.legend()
plt.show()

# Cnn and Logistic Regression loss algorithm 

plt.plot(cnn_history.loc[24::25,['Epoch']],cnn_history.loc[24::25,['Val_Loss']],label='CNN Loss')
plt.plot(logistic_output['iterations'],logistic_output['loss'],label='Logistic Regression Loss')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Loss comparison between Cnn and Logistic Regression Algorithm')
plt.legend()
plt.show()

# Cnn and SVM accuracy algorithm 

plt.plot(cnn_history.loc[24::25,['Epoch']],cnn_history.loc[24::25,['Val_Accuracy']],label='CNN Accuracy')
plt.plot(svm_output['max_iteration'],svm_output['accuracy'],label='SVM Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy comparison between Cnn and SVM Algorithm')
plt.legend()
plt.show()

# Cnn and SVM loss algorithm 

plt.plot(cnn_history.loc[24::25,['Epoch']],cnn_history.loc[24::25,['Val_Loss']],label='CNN Loss')
plt.plot(svm_output['max_iteration'],svm_output['loss'],label='SVM Loss')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Loss comparison between Cnn and SVM Algorithm')
plt.legend()
plt.show()
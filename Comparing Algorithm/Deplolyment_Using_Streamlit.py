import streamlit as slt
from Algorithm_accuracy import random_output,decision_output,logistic_output,svm_output
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

cnn_history=pd.read_csv('D://python//Flask-2//Ganesh_Fault_Detection//Comparsion_Accuracy_Train_and_Test//CNN_metrics.csv')
ann_history=pd.read_csv('D://python//Flask-2//Ganesh_Fault_Detection//Comparsion_Accuracy_Train_and_Test//CNN_metrics.csv')

print(cnn_history.loc[9::10])

# # CNN and Random Forest accuracy visualiztion 

# plt.plot(cnn_history.loc[9::10,['Epoch']],cnn_history.loc[9::10,['Val_Accuracy']],label='CNN Accuracy')
# plt.plot(random_output['N_estimators'],random_output['accuracy'],label='Random Forest Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.title('Accuracy comparison between Cnn and Random Forest Algorithm')
# plt.legend()
# plt.show()

# # CNN and Random Forest Loss visualization 

# plt.plot(cnn_history.loc[9::10,['Epoch']],cnn_history.loc[9::10,['Val_Loss']],label='CNN Loss')
# plt.plot(random_output['N_estimators'],random_output['losses'],label='Random Forest Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Loss comparison between Cnn and Random Forest Algorithm')
# plt.legend()
# plt.show()

# # Cnn and Logistic Regression accuracy algorithm 

# plt.plot(cnn_history.loc[24::25,['Epoch']],cnn_history.loc[24::25,['Val_Accuracy']],label='CNN Accuracy')
# plt.plot(logistic_output['iterations'],logistic_output['accuracy'],label='Logistic Regression Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.title('Accuracy comparison between Cnn and Logistic Regression Algorithm')
# plt.legend()
# plt.show()

# # Cnn and Logistic Regression loss algorithm 

# plt.plot(cnn_history.loc[9::10,['Epoch']],cnn_history.loc[9::10,['Val_Loss']],label='CNN Loss')
# plt.plot(logistic_output['iterations'],logistic_output['loss'],label='Logistic Regression Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.title('Loss comparison between Cnn and Logistic Regression Algorithm')
# plt.legend()
# plt.show()

# # Cnn and SVM accuracy algorithm 

# plt.plot(cnn_history.loc[9::10,['Epoch']],cnn_history.loc[9::10,['Val_Accuracy']],label='CNN Accuracy')
# plt.plot(svm_output['max_iteration'],svm_output['accuracy'],label='SVM Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.title('Accuracy comparison between Cnn and SVM Algorithm')
# plt.legend()
# plt.show()

# # Cnn and SVM loss algorithm 

# plt.plot(cnn_history.loc[9::10,['Epoch']],cnn_history.loc[24::25,['Val_Loss']],label='CNN Loss')
# plt.plot(svm_output['max_iteration'],svm_output['loss'],label='SVM Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.title('Loss comparison between Cnn and SVM Algorithm')
# plt.legend()
# plt.show()


def home_page():
    slt.title("Automated Fault Detection System")
    slt.header("Introduction:")
    slt.write("""
    The detection of faults in bottle caps and precise alignment of labels in manufacturing processes are crucial tasks in ensuring product quality and integrity. 
    To address these critical aspects, this project introduces an innovative approach leveraging cutting-edge Machine Learning (ML) algorithms, specifically Convolutional Neural Networks (CNN), Logistic Regression, Random Forest, Artificial Neural Networks (ANN), and Support Vector Machines (SVM), to enhance accuracy in bottle cap fault detection and label alignment.
    """)

    slt.header("Objective:")
    slt.write("""
    The primary objective of this project is to develop an advanced automated system that significantly improves the accuracy and efficiency of bottle cap fault detection and label alignment in manufacturing processes.
    """)

    slt.header("Purpose:")
    slt.write("""
    The purpose of this project is to create a robust and reliable automated system that addresses the challenges associated with bottle cap fault detection and label alignment. 
    By harnessing the power of machine learning, the system intends to streamline manufacturing operations, minimize defects, and uphold quality standards, thereby contributing to increased customer satisfaction and reduced manufacturing costs.
    """)

    slt.header("Scope:")
    slt.write("""
    The scope of this project encompasses the development and implementation of machine learning-based models utilizing CNN, Logistic Regression, Random Forest, ANN, and SVM algorithms. 
    These models will be trained and validated using datasets comprising images of bottle caps and labels with various defects and misalignments. 
    The system will focus on accurately detecting different types of faults in bottle caps, such as dents, deformities, or improper seals, while also ensuring precise alignment of labels.
    """)

    slt.write("""
    The project's scope extends to the evaluation and comparison of these machine learning algorithms' performance in terms of accuracy, speed, and scalability in the context of bottle cap fault detection and label alignment.
    """)

    slt.write("""
    Additionally, the system's usability and practicality in real manufacturing environments will be considered for potential deployment and integration into industrial processes.
    """)

    slt.write("""
    By achieving these objectives, the project aims to introduce an advanced automated system that significantly enhances the accuracy, efficiency, and reliability of bottle cap fault detection and label alignment processes, thereby positively impacting the manufacturing industry.
    """)
    slt.button("Next",on_click=data_visualization)

def data_visualization():
    slt.title("Data Visualization")
    slt.write("This is the next page!")
    if slt.button("Next"):
        comparison_algorithm_CNN_and_ANN()
def comparison_algorithm_CNN_and_ANN():
    slt.title("Next")
    slt.write("This is the next page!")
    if slt.button("Next"):
        comparison_random_forest_and_CNN
def comparison_random_forest_and_CNN():
    slt.title("Next Page")
    slt.write("This is the next page!")
    if slt.button("Next"):
        comparison_logistic_regression_and_CNN
def comparison_logistic_regression_and_CNN():
    slt.title("Next Page")
    slt.write("This is the next page!")
    if slt.button("Next"):
        comparison_svm_algorithm_and_CNN
def comparison_svm_algorithm_and_CNN():
    slt.title("Next Page")
    slt.write("This is the next page!")
    if slt.button("Next"):
        final_results()
def final_results():
    slt.title("Next Page")
    slt.write("This is the next page!")
    if slt.button("Back to Home"):
        home_page()
def main():
    slt.title('Automated  Fault Bottle Detection System')
    pages={
        'Home Page': home_page,
        'Data Visualization': data_visualization,
        'Comparison of CNN and ANN ALgorithm': comparison_algorithm_CNN_and_ANN,
        'Comparison of CNN and RandomForest Algorithms': comparison_random_forest_and_CNN,
        'Comparison of CNN and Logistic Regression Algorithm': comparison_logistic_regression_and_CNN,
        'Comparison of CNN and SVM ALgorithm': comparison_svm_algorithm_and_CNN,
        'Final Results': final_results
    }

    slt.sidebar.header('Pages')
    choice = slt.sidebar.radio('Select an option',list(pages.keys()))

    if choice in list(pages.keys()):
        pages[choice]()
if __name__=="__main__":
    main()
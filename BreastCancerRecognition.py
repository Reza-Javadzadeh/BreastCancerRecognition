'''--- Reza Javadzadeh
In this project we're up to predict the situation of Patient. whether she has a breast cancer or not; she just has a benign tumor.

More Projects and Information:
--- --- Github: https://www.github.com/Reza-Javadzadeh
--- --- LinkedIn: https://www.linkedin.com/in/reza-javadzadeh
'''

#let's import some useful module:

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#####################################
## Reading Data:
#####################################
from sklearn.datasets import load_breast_cancer

#For better visualization in Python Console's output, We should change the default option of dataframe visualiztion
pd.set_option('display.width',None)
pd.set_option('display.max_column',None)

# Method I : Reading,Cleaning and Showing the Data Frame by Pandas module:
# df=pd.read_csv(r'D:\Koolac\06- Machine Learning\P00-01-Datasets\06-Cancer.csv')
# x=df.iloc[:,:-1].values #input as a NumPy array
# y=df.iloc[:,-1].values #output as a NumPy array
# print(df.info(),end='\n\n') #Some useful information about our BreastCancer datset.
# print('Cancer DataFrame Head: \n\n',df.head(),end='\n\n')

# Method II : load dataset (or dataframe) by scikit-learn! this module already has this dataframe:
x,y=load_breast_cancer(return_X_y=True,as_frame=True) #input and output as an DataFrame and Series
print('Features of Breast Cancer Dataset:\n\n',x,end='\n\n')
print('Target Value of Breast Cancer Dataset: (0:Benignant , 1:Malignant)\n\n',y,end='\n\n')

x=x.values #input as an NumPy array
y=y.values #output as an NumPy array


#####################################
## Preprocessing:
#####################################


# Train-Test-Splitting: # We should distinguish between our training data and test data. both input and output.
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,stratify=y,random_state=None) #it's a binary classification , it's good to stratifing the choices

## Scaling: # Better to scale our train and test data both input and output
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)


## Feature Extraction  -> PCA :
from sklearn.decomposition import PCA ## PCA has a linear approach, we can use KernelPCA instead of it.
pca=PCA(n_components=None)         ## First, we find out how many Principle Components are enough , then we change the argument.So we first assign the argument to <<n_components=None>>
x_train=pca.fit_transform(x_train)
x_test=pca.transform(x_test)
print('PCA Variance Ratio :\n',pca.explained_variance_ratio_,end='\n\n') #For indicating Variance and Importancy of each new feature (i.e.: PC_i)
# compared to another, we use this command. Those features which have bigger  Variance, mean they are more vulnerable on the whole data and should
# consider in our silulation carefully, and those features which has few variance, can be ruduct to increase the performance of algorithms and may
# better evaluarion scores later.

print('PCA Cumulative Sum Variance Ratio : \n',np.cumsum(pca.explained_variance_ratio_),end='\n\n')#For indicating a better look of variances,we took
#CumSum of them.

# With ruuning the code till this line,and noticing to "PCA Cumulative Sum Variance Ratio" or 'PCA Variance Ratio', we'll find out
# that the proper value for PCA's argument (i.e.: <<n_components>>) is 15. (or another number like 16,17 ,etc..; it's actually desirable, I chose 15)
# So we change <<n_components=None>> to <<n_components=15>>


####################################
# Building The Model:
####################################
# Now we make our model with Support Vector Machine Classifier , and RBF kernel
from sklearn.svm import SVC
model=SVC(probability=True,kernel='rbf')
model.fit(x_train,y_train)



#####################################
## Prediction and Evaluation:
#####################################

##Prediction:
y_pred=model.predict(x_test)
y_pred_prob=model.predict_proba(x_test)


##Evaluation:
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,roc_auc_score,RocCurveDisplay,classification_report
label_order=[0,1]

## Accuracy:
acc=accuracy_score(y_true=y_test,y_pred=y_pred)

## Recall:
recall=recall_score(y_true=y_test,y_pred=y_pred,labels=label_order,average='binary')

## Precision:
precision=precision_score(y_true=y_test,y_pred=y_pred,labels=label_order,average='binary')

## F1:
f1=f1_score(y_true=y_test,y_pred=y_pred,labels=label_order,average='binary')

## AUC:
auc=roc_auc_score(y_true=y_test,y_score=y_pred_prob[:,1])

list=[acc,recall,precision,f1,auc]
printable_list=['Accuracy','Recall','Precision','F1','AUC']

for i,j in enumerate(printable_list):                       ## Printing the evaluation Scores
    print(f'The {j} Score: {list[i]}',end='\n\n')


# ROC Curve:

RocCurveDisplay.from_estimator(estimator=model,X=x_test,y=y_test)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC for Cancer Dataset')


## Confusion Matrix:
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_true=y_test,y_pred=y_pred,labels=label_order)

## Specificity:
specificity=cm[0,0]/np.sum(cm[0,:])
print(f'The Specificity Score: {specificity}',end='\n\n')


# Classification Report:
report=classification_report(y_true=y_test,y_pred=y_pred,labels=label_order,target_names=['Benign','Malignant'])
print('===========================================\n\nClassification Report: \n\n',report,end='\n\n')
#
## Plotting Confusion Matrix:
import seaborn as sns
plt.figure('Confusion Matrix')
sns.heatmap(cm,cmap='Greens',annot=True,fmt='0.2f',cbar_kws={'orientation':'vertical','label':'Color Bar'},xticklabels=['Benign','Malignant'],yticklabels=
            ['Benign','Malignant'])

plt.title('Confusion Matrix For Cancer Dataset')
plt.xlabel('Predicted')
plt.ylabel('Actual')


## Normalized Confusion Matrix:

normalized_cm=cm/(np.sum(cm,axis=1).reshape(-1,1))

# Plotting Normalized Confusion Matrix:

plt.figure('Normalized Confusion Matrix')
sns.heatmap(normalized_cm,cmap='Greens',annot=True,fmt='0.2f',cbar_kws={'orientation':'vertical','label':'Color Bar'},xticklabels=['Benign','Malignant'],yticklabels=
            ['Benign','Malignant'])

plt.title('Normalized Confusion Matrix For Cancer Dataset')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

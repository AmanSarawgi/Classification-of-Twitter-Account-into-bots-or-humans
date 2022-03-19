import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score,classification,roc_curve
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import classification_report
from yellowbrick.classifier import ClassificationReport
from sklearn.metrics import log_loss
from matplotlib import pyplot
from numpy import array
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest,SelectPercentile
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib



file='training_data_2_csv_UTF.csv'

training_data = pd.read_csv(file,encoding='latin-1')

data = pd.read_csv("features.csv")
print(data.head(9))
col = data.columns
data.isnull().sum()

X = data.drop(['bot'],1)
y=data['bot']

X_train, X_test, y_train, y_test = tts(
    X,
    y,
    test_size=0.3,
    random_state=0)
 
print(X_train.shape, X_test.shape)

##Forwards Feature Selection
sfs1=SFS(RandomForestClassifier(n_jobs=-1,n_estimators=100),
         k_features=8,
         forward=True,
         floating=False,
         verbose=2,
         scoring='roc_auc',
         cv=3
         )

sfs1=sfs1.fit(X_train,y_train)

select_feat_forward= X_train.columns[list(sfs1.k_feature_idx_)]
print("Feature Selection Method - Forward Feature Selection : ",select_feat_forward)

## Backward Feature Selection
sfs2=SFS(RandomForestClassifier(n_jobs=1,n_estimators=100),
         k_features=8,
         forward=False,
         floating=False,
         verbose=2,
         scoring='roc_auc',
         cv=3
         )

sfs2=sfs2.fit(np.array(X_train),y_train)

select_feat_backward= X_train.columns[list(sfs1.k_feature_idx_)]
print("Frature Selection Method - Backward Feature Selection : ", select_feat_backward)

def plot_roc_curve(fpr, tpr):  
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()



# Decission Tree
def Decissiontree(X_train, y_train, X_test, y_test):
    global acc1
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    print("decission Tree:train set")
    y_pred = dt.predict(X_train)
    pred = dt.predict_proba(X_test)
    print("Decission Tree:Confusion Matrix: ",
          confusion_matrix(y_train, y_pred))
    print("Decission Tree:Accuracy : ", accuracy_score(y_train, y_pred)*100)
    print("Decission Tree:Test set")
    y_pred = dt.predict(X_test)
    print("Decission Tree:Confusion Matrix: ",
          confusion_matrix(y_test, y_pred))
    print("Decission Tree:Accuracy : ", accuracy_score(y_test, y_pred)*100)
    acc1 = accuracy_score(y_test, y_pred)*100
    # confusion Matrix
    matrix = confusion_matrix(y_test, y_pred)
    class_names = [0, 1]
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    sns.heatmap(pd.DataFrame(matrix), annot=True, cmap="YlGnBu", fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()
    # ROC_AUC curve
    probs = dt.predict_proba(X_test)
    probs = probs[:, 1]
    auc = roc_auc_score(y_test, probs)
    print('AUC: %.2f' % auc)
    le = preprocessing.LabelEncoder()
    y_test1 = le.fit_transform(y_test)
    fpr, tpr, thresholds = roc_curve(y_test1, probs)
    plot_roc_curve(fpr, tpr)
    # Classification Report
    target_names = ['No','Yes']
    prediction = dt.predict(X_test)
    print(classification_report(y_test, prediction, target_names=target_names))
    classes = ['No','Yes']
    visualizer = ClassificationReport(dt, classes=classes, support=True)
    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)
    g = visualizer.poof()


Decissiontree(X_train[select_feat_forward],y_train,X_test[select_feat_forward],y_test)



#KNearest Neighbors
def KNearest(X_train,y_train,X_test,y_test):
  global acc2
  knn = KNeighborsClassifier()
  knn.fit(X_train,y_train)
  print("KNearest Neighbors:train set")
  y_pred = knn.predict(X_train)
  pred=knn.predict_proba(X_test)   
  print("KNearest Neighbors:Confusion Matrix: ", confusion_matrix(y_train, y_pred))
  print ("KNearest Neighbors:Accuracy : ", accuracy_score(y_train,y_pred)*100)
  print("KNearest Neighbors:Test set")
  y_pred = knn.predict(X_test)
  print("KNearest Neighbors:Confusion Matrix: ", confusion_matrix(y_test, y_pred))
  print ("KNearest Neighbors:Accuracy : ", accuracy_score(y_test,y_pred)*100)
  acc2=accuracy_score(y_test,y_pred)*100
  #confusion Matrix
  matrix =confusion_matrix(y_test, y_pred)
  class_names=[0,1] 
  fig, ax = plt.subplots()
  tick_marks = np.arange(len(class_names))
  plt.xticks(tick_marks, class_names)
  plt.yticks(tick_marks, class_names)
  sns.heatmap(pd.DataFrame(matrix), annot=True, cmap="YlGnBu" ,fmt='g')
  ax.xaxis.set_label_position("top")
  plt.tight_layout()
  plt.title('Confusion matrix', y=1.1)
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label')
  plt.show()
  #ROC_AUC curve
  probs = knn.predict_proba(X_test) 
  probs = probs[:, 1]  
  auc = roc_auc_score(y_test, probs)  
  print('AUC: %.2f' % auc)
  le = preprocessing.LabelEncoder()
  y_test1=le.fit_transform(y_test)
  fpr, tpr, thresholds = roc_curve(y_test1, probs)
  plot_roc_curve(fpr, tpr)
  #Classification Report
  target_names = ['Yes', 'No']
  prediction=knn.predict(X_test)
  print(classification_report(y_test, prediction, target_names=target_names))
  classes = ["Yes", "No"]
  visualizer = ClassificationReport(knn, classes=classes, support=True)
  visualizer.fit(X_train, y_train)  
  visualizer.score(X_test, y_test)  
  g = visualizer.poof()

KNearest(X_train[select_feat_forward],y_train,X_test[select_feat_forward],y_test)




#SVM
def SupportVector(X_train, y_train, X_test, y_test):
    global acc4
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    print("SVM:train set")
    y_pred = clf.predict(X_train)
    pred = clf.predict_proba(X_test)
    print("SVM:Confusion Matrix: ",
          confusion_matrix(y_train, y_pred))
    print("SVM:Accuracy : ", accuracy_score(y_train, y_pred)*100)
    print("SVM:Test set")
    y_pred = clf.predict(X_test)
    print("SVM:Confusion Matrix: ",
          confusion_matrix(y_test, y_pred))
    print("SVM:Accuracy : ", accuracy_score(y_test, y_pred)*100)
    acc4 = accuracy_score(y_test, y_pred)*100
    # confusion Matrix
    matrix = confusion_matrix(y_test, y_pred)
    class_names = [0, 1]
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    sns.heatmap(pd.DataFrame(matrix), annot=True, cmap="YlGnBu", fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()
    # ROC_AUC curve
    probs = clf.predict_proba(X_test)
    probs = probs[:, 1]
    auc = roc_auc_score(y_test, probs)
    print('AUC: %.2f' % auc)
    le = preprocessing.LabelEncoder()
    y_test1 = le.fit_transform(y_test)
    fpr, tpr, thresholds = roc_curve(y_test1, probs)
    plot_roc_curve(fpr, tpr)
    # Classification Report
    target_names = ['No','Yes']
    prediction = clf.predict(X_test)
    print(classification_report(y_test, prediction, target_names=target_names))
    classes = ['No','Yes']
    visualizer = ClassificationReport(clf, classes=classes, support=True)
    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)
    g = visualizer.poof()


SupportVector(X_train[select_feat_forward],y_train,X_test[select_feat_forward],y_test)


#RandomForestClassifier 
def RandomForest(X_train,y_train,X_test,y_test):
  global acc3
  rf = RandomForestClassifier()
  rf.fit(X_train,y_train)
  joblib.dump(rf,'RandomForest.pkl')
  print("RandomForest:train set")
  y_pred = rf.predict(X_train)
  pred=rf.predict_proba(X_test)   
  print("RandomForest:Confusion Matrix: ", confusion_matrix(y_train, y_pred))
  print ("RandomForest:Accuracy : ", accuracy_score(y_train,y_pred)*100)
  print("RandomForest:Test set")
  y_pred = rf.predict(X_test)
  print("RandomForest:Confusion Matrix: ", confusion_matrix(y_test, y_pred))
  print ("RandomForest:Accuracy : ", accuracy_score(y_test,y_pred)*100)
  acc3=accuracy_score(y_test,y_pred)*100
  #confusion Matrix
  matrix =confusion_matrix(y_test, y_pred)
  class_names=[0,1] 
  fig, ax = plt.subplots()
  tick_marks = np.arange(len(class_names))
  plt.xticks(tick_marks, class_names)
  plt.yticks(tick_marks, class_names)
  sns.heatmap(pd.DataFrame(matrix), annot=True, cmap="YlGnBu" ,fmt='g')
  ax.xaxis.set_label_position("top")
  plt.tight_layout()
  plt.title('Confusion matrix', y=1.1)
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label')
  plt.show()
  #ROC_AUC curve
  probs = rf.predict_proba(X_test) 
  probs = probs[:, 1]  
  auc = roc_auc_score(y_test, probs)  
  print('AUC: %.2f' % auc)
  le = preprocessing.LabelEncoder()
  y_test1=le.fit_transform(y_test)
  fpr, tpr, thresholds = roc_curve(y_test1, probs)
  plot_roc_curve(fpr, tpr)
  #Classification Report
  target_names = ['Yes', 'No']
  prediction=rf.predict(X_test)
  print(classification_report(y_test, prediction, target_names=target_names))
  classes = ["Yes", "No"]
  visualizer = ClassificationReport(rf, classes=classes, support=True)
  visualizer.fit(X_train, y_train)  
  visualizer.score(X_test, y_test)  
  g = visualizer.poof()

RandomForest(X_train[select_feat_forward],y_train,X_test[select_feat_forward],y_test)


labels = ['Decision Tree','KNN','SVM','RandomForest']
#sizes = [5, neg_per, neu_per]
sizes = [acc1,acc2,acc4,acc3]
index = np.arange(len(labels))
plt.bar(index, sizes)
plt.xlabel('Algorithm', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)
plt.xticks(index, labels, fontsize=10, rotation=0)
plt.title('comparative study')
plt.show()



# GUI
from tkinter import *
from tkinter import messagebox
#from openpyxl import load_workbook
#import xlrd
import pandas as pd
#from auto_tqdm import tqdm

root = Tk()  # Main window 
f = Frame(root)
frame1 = Frame(root)
frame2 = Frame(root)
frame3 = Frame(root)
root.title("Twitter Bot detection")
root.geometry("1080x720")

canvas = Canvas(width=1080, height=250)
canvas.pack()
photo = PhotoImage(file='landscape.png')
canvas.create_image(-80, -80, image=photo, anchor=NW)

root.configure(background='white')
scrollbar = Scrollbar(root)
scrollbar.pack(side=RIGHT, fill=Y)

firstname = StringVar()  # Declaration of all variables
lastname = StringVar()
id = StringVar()
dept = StringVar()
designation = StringVar()
remove_firstname = StringVar()
remove_lastname = StringVar()
searchfirstname = StringVar()
searchlastname = StringVar()
sheet_data = []
row_data = []





def add_entries():  # to append all data and add entries on click the button
    a = " "
    f = firstname.get()
    f1 = f.lower()
    l = lastname.get()
    l1 = l.lower()
    d = dept.get()
    d1 = d.lower()
    de = designation.get()
    de1 = de.lower()
    list1 = list(a)
    list1.append(f1)
    list1.append(l1)
    list1.append(d1)
    list1.append(de1)


def click( ):
    screen_name_binary =  str(e1.get())
    description_binary =  str(e2.get())
    status_binary =  str(e3.get())
    verified =  str(e4.get())
    followers_count =  float(e5.get())
    friends_count =  float(e6.get())
    favourites_count =  str(e7.get())
    statuses_count =  float(e8.get())
    bag_of_words_bot = r'bot|b0t|cannabis|tweet me|mishear|follow me|updates every|gorilla|yes_ofc|forget' \
                    r'expos|kill|clit|bbb|butt|fuck|XXX|sex|truthe|fake|anony|free|virus|funky|RNA|kuck|jargon' \
                    r'nerd|swag|jack|bang|bonsai|chick|prison|paper|pokem|xx|freak|ffd|dunia|clone|genie|bbb' \
                    r'ffd|onlyman|emoji|joke|troll|droop|free|every|wow|cheese|yeah|bio|magic|wizard|face'


    if(screen_name_binary in (bag_of_words_bot)):
      screen_name_binary=False
    else:
      screen_name_binary=True

    if(description_binary in (bag_of_words_bot)):
      description_binary=False
    else:
      description_binary=True

    if(status_binary in (bag_of_words_bot)):
      status_binary=False
    else:
      status_binary=True

    if(favourites_count in (bag_of_words_bot)):
      favourites_count=False
    else:
      favourites_count=True

    if(verified =="False"):
      verified=False
    else:
      verified=True


    col = ['screen_name_binary', 'description_binary', 'status_binary','verified','followers_count', 'friends_count', 'favourites_count','statuses_count']

    

    output_data=[]
    output_data.append(screen_name_binary)
    output_data.append(description_binary)
    output_data.append(status_binary)
    output_data.append(verified)
    output_data.append(followers_count)
    output_data.append(friends_count)
    output_data.append(favourites_count)
    output_data.append(statuses_count)


    output_data=pd.DataFrame([output_data],columns = col)


    rf=joblib.load('RandomForest.pkl')
    pred=rf.predict(output_data)
    print("Prediction for newly added data : ",pred)

    if(pred==1):
        e10.delete(0, END) #deletes the current value
        e10.insert(0, "BOT")
    else:
        e10.delete(0, END) #deletes the current value
        e10.insert(0, "NONBOT")



def clear_all():  # for clearing the entry widgets
    frame1.pack_forget()
    frame2.pack_forget()
    frame3.pack_forget()




label1 = Label(root, text="TWITTER BOT DETECTION")
label1.config(font=('Italic', 18, 'bold'), justify=CENTER, background="Yellow", fg="Red", anchor="center")
label1.pack(fill=X)



frame2.pack_forget()
frame3.pack_forget()
screen_name_binary = Label(frame1, text="screen_name_binary: ", bg="red", fg="white")
screen_name_binary.grid(row=1, column=1, padx=10,pady=10)
e1 = Entry(frame1 )
e1.grid(row=1, column=2, padx=10,pady=10)
e1.focus()

description_binary = Label(frame1, text="description_binary: ", bg="red", fg="white")
description_binary.grid(row=2, column=1, padx=10,pady=10)
e2 = Entry(frame1 )
e2.grid(row=2, column=2, padx=10,pady=10)

status_binary = Label(frame1, text="status_binary: ", bg="red", fg="white")
status_binary.grid(row=3, column=1, padx=10,pady=10)
e3 = Entry(frame1)
e3.grid(row=3, column=2, padx=10,pady=10)


verified = Label(frame1, text="verified: ", bg="red", fg="white")
verified.grid(row=4, column=1, padx=10,pady=10)
e4 = Entry(frame1)
e4.grid(row=4, column=2, padx=10,pady=10)


followers_count = Label(frame1, text="followers_count: ", bg="red", fg="white")
followers_count.grid(row=5, column=1, padx=10,pady=10)
e5 = Entry(frame1)
e5.grid(row=5, column=2, padx=10,pady=10)


friends_count = Label(frame1, text="friends_count: ", bg="red", fg="white")
friends_count.grid(row=6, column=1, padx=10,pady=10)
e6 = Entry(frame1)
e6.grid(row=6, column=2, padx=10,pady=10)

favourites_count = Label(frame1, text="favourites_count: ", bg="red", fg="white")
favourites_count.grid(row=7, column=1, padx=10,pady=10)
e7 = Entry(frame1)
e7.grid(row=7, column=2, padx=10,pady=10)


statuses_count = Label(frame1, text="statuses_count: ", bg="red", fg="white")
statuses_count.grid(row=8, column=1, padx=10,pady=10)
e8 = Entry(frame1)
e8.grid(row=8, column=2, padx=10,pady=10)







button5 = Button(frame1, text="Predict", command=click)
button5.grid(row=9, column=1, pady=10,padx=10)


output = Label(frame1, text="output: ", bg="red", fg="white")
output.grid(row=10, column=1, padx=10)
e10 = Entry(frame1)
e10.grid(row=10, column=2, padx=10)

frame1.configure(background="Red")
frame1.pack(pady=10)



# f.configure(background="Submit")
# f.pack()

root.mainloop()





from io import open

file1= open('test_data_4_students.csv', mode='r', encoding='utf-8', errors='ignore')

test_data = pd.read_csv(file1)
col = test_data.columns
bag_of_words_bot = r'bot|b0t|cannabis|tweet me|mishear|follow me|updates every|gorilla|yes_ofc|forget' \
                    r'expos|kill|clit|bbb|butt|fuck|XXX|sex|truthe|fake|anony|free|virus|funky|RNA|kuck|jargon' \
                    r'nerd|swag|jack|bang|bonsai|chick|prison|paper|pokem|xx|freak|ffd|dunia|clone|genie|bbb' \
                    r'ffd|onlyman|emoji|joke|troll|droop|free|every|wow|cheese|yeah|bio|magic|wizard|face'
            
test_data['screen_name_binary'] = test_data.screen_name.str.contains(bag_of_words_bot, case=False, na=False)
test_data['name_binary'] = test_data.name.str.contains(bag_of_words_bot, case=False, na=False)
test_data['description_binary'] = test_data.description.str.contains(bag_of_words_bot, case=False, na=False)
test_data['status_binary'] = test_data.status.str.contains(bag_of_words_bot, case=False, na=False)
test_data['listed_count_binary'] = (test_data.listed_count>20000)==False
test_data['favourites_count'] = (test_data.favourites_count==0)==False
features = ['screen_name_binary', 'description_binary', 'status_binary','verified','followers_count', 'friends_count', 'favourites_count','statuses_count']

df=test_data[features]

X_test1 = df
rf=joblib.load('RandomForest.pkl')
pred=rf.predict(X_test1)
id_df=test_data['id_str']
pred_df=pred


dataframe=pd.DataFrame(pred_df, columns=['Bot']) 
dataframe1=pd.DataFrame(test_data,columns=col)
dframe= pd.concat([dataframe1, dataframe], axis=1, sort=False)
dframe.to_csv('Output.csv') 
print('Done')











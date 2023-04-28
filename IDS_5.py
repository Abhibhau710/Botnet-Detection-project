import tkinter as tk
from tkinter import *
import pandas as pd
import numpy as np
#from hmmlearn import hmm
from sklearn.preprocessing import LabelEncoder                  #for preprocessing
from sklearn.model_selection import train_test_split                      #split data into traning and testing
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score               # for calculating accuracy and generating classification report
from sklearn.ensemble import VotingClassifier                       #inbuild library for voting algorithm
from sklearn.svm import SVC                      #inbuild library for SVM algorithm


from sklearn import model_selection
le = LabelEncoder()     #object creation

root = tk.Tk()                    #rrot object of tkinter

w,h = root.winfo_screenwidth() ,root.winfo_screenheight()                 #calculating screen width and height
root.geometry("%dx%d+0+0"%(w,h))
root.title("BOTNET DETECTION SYSTEM")
root.configure(background="cyan2")








new_data = pd.read_csv(r'D:\test\IDS_5_CLASSIFIERS2\KDDTrain.csv') #, sep='COLUMN_SEPARATOR', dtype=np.float64)
new_data['protocol_type']=le.fit_transform(new_data['protocol_type'])
new_data['service']=le.fit_transform(new_data['service'])
new_data['flag']=le.fit_transform(new_data['flag'])
new_data['label']=le.fit_transform(new_data['label'])
x_train = new_data.drop(["label"],axis=1)
# x_train = new_data.drop(["label","duration","src_bytes","dst_bytes", "land","urgent","hot","num_failed_logins","lnum_compromised","lroot_shell","lsu_attempted","lnum_root","lnum_file_creations","lnum_shells","lnum_access_files","is_host_login","is_guest_login","srv_count","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate"],axis=1)
y_train = new_data["label"]


new_data = pd.read_csv(r'D:\test\IDS_5_CLASSIFIERS2\KDDTest.csv') #, sep='COLUMN_SEPARATOR', dtype=np.float64)
new_data['protocol_type']=le.fit_transform(new_data['protocol_type'])
new_data['service']=le.fit_transform(new_data['service'])
new_data['flag']=le.fit_transform(new_data['flag'])
new_data['label']=le.fit_transform(new_data['label'])
# x_test = new_data.drop(["label","duration","src_bytes","dst_bytes", "land","urgent","hot","num_failed_logins","lnum_compromised","lroot_shell","lsu_attempted","lnum_root","lnum_file_creations","lnum_shells","lnum_access_files","is_host_login","is_guest_login","srv_count","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate"],axis=1)
x_test = new_data.drop(["label"],axis=1)
y_test = new_data["label"]

from xgboost import XGBClassifier              #inbuild library for XG Bosst  algorithm


def XG_BOOST():
  
    model1 = XGBClassifier(n_estimators = 400, learning_rate = 0.1, max_depth = 3)
    model1.fit(x_train, y_train)
    
    model1_pred = model1.predict(x_test)
    print(model1_pred)
    
    
    print("=" * 40)
    print("==========")
    print("Classification Report : ",(classification_report(y_test, model1_pred)))
    print("Accuracy : ",accuracy_score(y_test,model1_pred)*100)
    accuracy = accuracy_score(y_test, model1_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    ACC = (accuracy_score(y_test, model1_pred) * 100)
    repo = (classification_report(y_test, model1_pred))
    
    label4 = tk.Label(root,text =str(repo),width=35,height=10,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label4.place(x=305,y=100)
    
    label5 = tk.Label(root,text ="Accuracy : "+str(ACC)+"%\nModel saved as XG_BOOST.joblib",width=35,height=3,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label5.place(x=305,y=320)
    from joblib import dump
    dump (model1,"XG_BOOST.joblib")
    print("Model saved as XG_BOOST.joblib")


def SVM():
    
    model2 = SVC(kernel='rbf', random_state = 100)  
    model2.fit(x_train[:1000], y_train[:1000])
    
    
    print("=" * 40)
    model2_pred = model2.predict(x_test)
    # print(model2_pred)
    
    print("=" * 40)
    print("==========")
    print("Classification Report : ",(classification_report(y_test, model2_pred)))
    print("Accuracy : ",accuracy_score(y_test,model2_pred)*100)
    accuracy = accuracy_score(y_test, model2_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    ACC = (accuracy_score(y_test, model2_pred) * 100)
    repo = (classification_report(y_test, model2_pred))
    
    label4 = tk.Label(root,text =str(repo),width=35,height=10,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label4.place(x=305,y=100)
    
    label5 = tk.Label(root,text ="Accuracy : "+str(ACC)+"%\nModel saved as SVM.joblib",width=35,height=3,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label5.place(x=305,y=320)
    from joblib import dump
    dump (model2,"SVM.joblib")
    print("Model saved as SVM.joblib")
    
from sklearn.tree import DecisionTreeClassifier              #inbuild library for Decision tree algorithm

def DT():
    
    model4 = DecisionTreeClassifier(criterion = "gini",random_state = 100,max_depth=3, min_samples_leaf=5)   
    
    model4.fit(x_train, y_train)
    
    
    
    print("=" * 40)
    model4.fit(x_train, y_train)
    
    model4_pred = model4.predict(x_test)
    #print(model2_pred)
    
    
    print("=" * 40)
    print("==========")
    print("Classification Report : ",(classification_report(y_test, model4_pred)))
    print("Accuracy : ",accuracy_score(y_test,model4_pred)*100)
    accuracy = accuracy_score(y_test, model4_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    ACC = (accuracy_score(y_test, model4_pred) * 100)
    repo = (classification_report(y_test, model4_pred))
    
    label4 = tk.Label(root,text =str(repo),width=35,height=10,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label4.place(x=305,y=100)
    
    label5 = tk.Label(root,text ="Accuracy : "+str(ACC)+"%\nModel saved as DT.joblib",width=35,height=3,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label5.place(x=305,y=320)
    from joblib import dump
    dump (model4,"DT.joblib")
    print("Model saved as DT.joblib")
    
from sklearn.ensemble import RandomForestClassifier                 #inbuild library for Random Forest algorithm
    
def RF():
    
    #seed = 7
    num_trees = 100
    max_features = 3
    
    model5 = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)

    
    
    model5.fit(x_train, y_train)
    
    model5_pred = model5.predict(x_test)
    print(model5_pred)
    
    
    print("=" * 40)
    print("==========")
    print("Classification Report : ",(classification_report(y_test, model5_pred)))
    print("Accuracy : ",accuracy_score(y_test,model5_pred)*100)
    accuracy = accuracy_score(y_test, model5_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    ACC = (accuracy_score(y_test, model5_pred) * 100)
    repo = (classification_report(y_test, model5_pred))
    
    label4 = tk.Label(root,text =str(repo),width=35,height=10,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label4.place(x=305,y=100)
    
    label5 = tk.Label(root,text ="Accuracy : "+str(ACC)+"%\nModel saved as RF.joblib",width=35,height=3,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label5.place(x=305,y=320)
    from joblib import dump
    dump(model5,"RF.joblib")
    print("Model saved as RF.joblib")

from sklearn.naive_bayes import GaussianNB               #inbuild library for Naivy bayes algorithm
    
def NB():
    
    model3 = GaussianNB()
    model3.fit(x_train, y_train)
    
    
    
    print("=" * 40)
    model3.fit(x_train, y_train)
    
    model3_pred = model3.predict(x_test)
  
      
    
    print("=" * 40)
    print("==========")
    print("Classification Report : ",(classification_report(y_test, model3_pred)))
    print("Accuracy : ",accuracy_score(y_test,model3_pred)*100)
    accuracy = accuracy_score(y_test, model3_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    ACC = (accuracy_score(y_test, model3_pred) * 100)
    repo = (classification_report(y_test, model3_pred))
    
    label4 = tk.Label(root,text =str(repo),width=35,height=10,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label4.place(x=305,y=100)
    
    label5 = tk.Label(root,text ="Accuracy : "+str(ACC)+"%\nModel saved as NB.joblib",width=35,height=3,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label5.place(x=305,y=320)
    from joblib import dump
    dump (model3,"NB.joblib")
    print("Model saved as NB.joblib")


def VE():

    model1 = XGBClassifier(n_estimators = 400, learning_rate = 0.1, max_depth = 3)
    model4 = DecisionTreeClassifier(criterion = "gini",random_state = 100,max_depth=3, min_samples_leaf=5)   
    # model2 = SVC(kernel='rbf', random_state = 100)  
    model5 = RandomForestClassifier(n_estimators=100, max_features=3)
    model6 = VotingClassifier(estimators=[('dt', model4),('rf',model5)], voting='hard')
    
    model6.fit(x_train, y_train)
    
    
    
    print("=" * 40)
    model6.fit(x_train, y_train)
    
    model6_pred = model6.predict(x_test)
    #print(model2_pred)
    
    
    print("=" * 40)
    print("==========")
    print("Classification Report : ",(classification_report(y_test, model6_pred)))
    print("Accuracy : ",accuracy_score(y_test,model6_pred)*100)
    accuracy = accuracy_score(y_test, model6_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    ACC = (accuracy_score(y_test, model6_pred) * 100)
    repo = (classification_report(y_test, model6_pred))
    
    label4 = tk.Label(root,text =str(repo),width=35,height=10,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label4.place(x=305,y=100)
    
    label5 = tk.Label(root,text ="Accuracy : "+str(ACC)+"%\nModel saved as VE.joblib",width=35,height=3,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label5.place(x=305,y=320)
    from joblib import dump
    dump (model6,"VE.joblib")
    print("Model saved as VE.joblib")
 

    
def EXIT():
    root.destroy()



frame = tk.LabelFrame(root,text="Control Panel",width=200,height=500,bd=1,background="cyan2",font=("Tempus Sanc ITC",15,"bold"))
frame.place(x=5,y=50)

button1 = tk.Button(frame,command=XG_BOOST,text="XG_BOOST",bg="white",fg="black",width=15,font=("Times New Roman",15,"italic"))
button1.place(x=5,y=1)

button2 = tk.Button(frame,command=SVM,text="SVM",bg="white",fg="black",width=15,font=("Times New Roman",15,"italic"))
button2.place(x=5,y=50)

button3 = tk.Button(frame,command=NB,text="Naive Bayes",bg="white",fg="black",width=15,font=("Times New Roman",15,"italic"))
button3.place(x=5,y=100)   #y=100

button4 = tk.Button(frame,command=DT,text="Decision Tree",bg="white",fg="black",width=15,font=("Times New Roman",15,"italic"))
button4.place(x=5,y=165)

button5 = tk.Button(frame,command=RF,text="Random Forest",bg="white",fg="black",width=15,font=("Times New Roman",15,"italic"))
button5.place(x=5,y=220)

button6 = tk.Button(frame,command=VE,text="Voting Ensemble",bg="white",fg="black",width=15,font=("Times New Roman",15,"italic"))
button6.place(x=5,y=260)

TMState=tk.IntVar()
TMState=""

from tkinter import ttk
State_Name ={"XG_BOOST":1,"NAIVE BAYES":2,"DECISION TREE":3,"RANDOM FOREST":4, "SUPPORT VECTOR MACHINE":5,"VOTING ENSEMBLE":6} #{"XG_BOOST":1,"SUPPORT VECTOR MACHINE":2,"NAIVE BAYES":3,"DECISION TREE":4,"RANDOM FOREST":5,"VOTING ENSEMBLE":6}
TMStateEL=ttk.Combobox(frame,values=list(State_Name.keys()),width=25,textvariable = TMState)
TMStateEL.state(['readonly'])
TMStateEL.bind("<<ComboboxSelected>>", lambda event: print(State_Name[TMStateEL.get()]))

TMStateEL.current(0)
TMStateEL.place(x=5,y=320)

# model_list = {"XG_BOOST":"D:\test\IDS_5_CLASSIFIERS2\OLD_MODELS/XG_BOOST.joblib",
#               "SUPPORT VECTOR MACHINE":"D:\test\IDS_5_CLASSIFIERS2\OLD_MODELS/SVM.joblib",
#               "NAIVE BAYES":"D:\test\IDS_5_CLASSIFIERS2\OLD_MODELS/NB.joblib",
#               "DECISION TREE":"IDS_5_CLASSIFIERS2\DT.joblib",
#               "RANDOM FOREST":"D:\test\IDS_5_CLASSIFIERS2\OLD_MODELS/RF.joblib",
#               "VOTING ENSEMBLE":"D:\test\IDS_5_CLASSIFIERS2\OLD_MODELS/VE.joblib"
#               }

model_list = {
    "XG_BOOST":"IDS_5_CLASSIFIERS2\XG_BOOST.joblib",
    "SUPPORT VECTOR MACHINE":"IDS_5_CLASSIFIERS2\SVM.joblib",
    "DECISION TREE":"IDS_5_CLASSIFIERS2\DT.joblib",
    "RANDOM FOREST":"IDS_5_CLASSIFIERS2\RF.joblib",
    "VOTING ENSEMBLE":"IDS_5_CLASSIFIERS2\VE.joblib",
    "NAIVE BAYES":"IDS_5_CLASSIFIERS2\B.joblib"
    }

def ok():
    
    print ("value is:" + TMStateEL.get())
    model_choice = TMStateEL.get()
    choosen_model = model_list[model_choice]
    print(choosen_model)
    from joblib import load
    ans = load(choosen_model)
    
    from tkinter.filedialog import askopenfilename
    fileName = askopenfilename(initialdir='D:\test\IDS_5_CLASSIFIERS2/', title='Select DataFile For BOTNET Testing',
                                       filetypes=[("all files", "*.csv*")])
    
    file =pd.read_csv(fileName)
    file['protocol_type']=le.fit_transform(file['protocol_type'])
    file['service']=le.fit_transform(file['service'])
    file['flag']=le.fit_transform(file['flag'])

    qn = file.drop(["label"],axis=1)
    
    A = ans.predict(qn)
    print(A)
    def listToString(s): 
    
        # initialize an empty string
        str1 = "" 
        
        # traverse in the string  
        for ele in s: 
            str1 += ele  
        
        # return string  
        return str1 
    print(listToString(A)) 
    B = listToString(A)
    if B == 'normal':   
        output = 'Botnet Not Detected'
    else:
        output = 'Botnet Detected'
    
    attack = tk.Label(root,text=str(output),width=30,bg='red',fg='white',font=("Times New Roman",20,'italic'))
    attack.place(x=170,y=550)
    
        
    

button7 = tk.Button(frame,command=ok,text="TEST",bg="white",fg="black",width=15,font=("Times New Roman",15,"italic"))
button7.place(x=5,y=350)

button8 = tk.Button(frame,command=EXIT,text="EXIT",bg="red",fg="black",width=15,font=("Times New Roman",15,"italic"))
button8.place(x=5,y=400)




head = tk.Label(root,text = "BOTNET DETECTION SYSTEM",width=100,height=1,bg='black',fg='white',font=("Tempus Sanc ITC",18,"italic"))
head.place(x=0,y=0)



root.mainloop()




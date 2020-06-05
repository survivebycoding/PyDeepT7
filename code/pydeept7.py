from keras import backend as K
import tensorflow as tf
import math
def pydeept7():
    import tkinter as tk

    class SampleApp(tk.Tk):
      def __init__(top):
        tk.Tk.__init__(top)
        top.geometry('550x300+500+300')
        top.title('PyDeepT7')
        top.configure(background='plum1')
        top.newline = tk.Label(top, text="", bd =5, bg='plum1').grid(row=0,column=3)
        
        
        top.caption = tk.Label(top, text="Please insert the file names for executing PyDeepT7", bd =5, bg='plum1').grid(row=2,column=1)
        top.newline = tk.Label(top, text="", bd =5, bg='plum1').grid(row=3,column=1)
        top.label1 = tk.Label(top, text="Sample peptide file", bd =5, bg='plum1').grid(row=6,column=1)
        top.label2 = tk.Label(top, text="Sample nucleotide file", bd =5, bg='plum1').grid(row=8,column=1)
        top.label3 = tk.Label(top, text="Effector feature file", bd =5, bg='plum1').grid(row=10,column=1)
        top.label4 = tk.Label(top, text="Non-effector feature file", bd =5, bg='plum1').grid(row=12,column=1)
        
        top.entry1 = tk.Entry(top, bd =3, width=40)
        top.entry1.insert(10, "mixed_pro.txt")
        top.entry2 = tk.Entry(top, bd =3, width=40)
        top.entry2.insert(10, "mixed_gen.txt")
        top.entry3 = tk.Entry(top, bd =3, width=40)
        top.entry3.insert(10, "Feature_effector7.csv")
        top.entry4 = tk.Entry(top, bd =3, width=40)
        top.entry4.insert(10, "Feature_noneffector7.csv")

        
        top.button = tk.Button(top, text="Predict!", command=top.on_button, padx=2, pady=2, width=10, bg="bisque2")
        top.entry1.grid(row=6, column=2)
        top.entry2.grid(row=8, column=2)
        top.entry3.grid(row=10, column=2)
        top.entry4.grid(row=12, column=2)
        top.newline = tk.Label(top, text="", bd =5, bg='plum1').grid(row=13,column=1)
        top.button.grid(row=16, column=2)
    
        

      def on_button(top):
        x1=top.entry1.get()
        x2=top.entry2.get()
        x3=top.entry3.get()
        x4=top.entry4.get()
        top.destroy()
        voting(x1,x2,x3,x4)
        

    app = SampleApp()
    
    app.mainloop()

def voting(peptide_predict_file,nucleotide_predict_file,effector_train,noneffector_train):

    total = 0
  
    with open(peptide_predict_file) as f:
     for line in f:
        finded = line.find('>')
        
        if finded == 0:
            total =total+ 1

    print('Total number of sequences to be classified: ',total)
    
    import math
    import random
    import pandas
    import numpy as np
    import csv
    import warnings
    import keras
    from sklearn.metrics import roc_auc_score
    from keras.wrappers.scikit_learn import KerasClassifier
    from sklearn import svm
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from random import shuffle
    from sklearn.model_selection import train_test_split, cross_val_score
    f=random.seed()
    from sklearn.metrics import classification_report
    import matplotlib.pyplot as plt
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import roc_curve, auc
    from keras.layers.normalization import BatchNormalization
    from keras.layers.advanced_activations import LeakyReLU
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.utils import np_utils
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from keras.models import Sequential
    from keras.layers import Dense
    from imblearn.over_sampling import SMOTE, ADASYN
    from collections import Counter
    import warnings
    warnings.filterwarnings("ignore")
    from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, cohen_kappa_score
    from keras.models import model_from_json
    from sklearn.preprocessing import label_binarize
    from sklearn.multiclass import OneVsRestClassifier
    from scipy import interp
    import os
    import pickle
    from joblib import dump, load
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import f1_score
    from keras.layers import Input, Embedding, LSTM, Dense, SimpleRNN
    from sklearn import metrics
    from keras import metrics
    from keras import backend as K
    import time
    start_time = time.clock()
    f=random.seed()
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    #getting feature vector of sequence to be predicted
    featurevector=featureextraction(peptide_predict_file, nucleotide_predict_file, total)
    end_time = time.clock()
    
    print('Execution time',(end_time-start_time))

 
    #getting training data
    dataframe = pandas.read_csv(effector_train, header=None, sep=',')
    dataset = dataframe.values
    eff = dataset[:,0:10000].astype('float32')

    dataframe = pandas.read_csv(noneffector_train, header=None, sep=',')
    dataset = dataframe.values
    noneff = dataset[:,0:10000].astype('float32')

    
    a1=eff.shape
    a2=noneff.shape
    X = np.ones((a1[0]+a2[0],a1[1]))
    Y = np.ones((a1[0]+a2[0],1))
    Y1 = np.ones((a1[0]+a2[0],1))    
    
    for i in range(a1[0]):
        for j in range(a1[1]):
            X[i][j]=eff[i][j]
        Y[i,0]=0
        Y1[i,0]=1
        #print(i)    
    for i in range(a2[0]):
        for j in range(a2[1]):
            X[i+a1[0]][j]=noneff[i][j]
        Y[i+a1[0]][0]=1
        Y1[i+a1[0]][0]=2
        
    print('before oversampling',X.shape)
    warnings.filterwarnings("ignore")
    
 
    
    #train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=f)
    print(X_train.shape)
    
 
    
    json_file = open('deepmodel.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("deepmodel.h5")
    print("Loaded model from disk")
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
    score = loaded_model.evaluate(X_test, y_test, verbose=0)
    #print(score)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
    probs = loaded_model.predict(featurevector)
    #print(probs)
    
    test_pred = np.ones((len(y_test)))
    y_pred_test= loaded_model.predict(X_test)
         
    for i in range(len(y_pred_test)):
        if y_pred_test[i][0]<0.5:
            test_pred[i]=0
    con=confusion_matrix(y_test, test_pred)
    total0=len(y_test)-np.count_nonzero(y_test)
    total1=np.count_nonzero(y_test)
    TP=con[0][0]
    TN=con[1][1]
    FP=con[0][1]
    FN=con[1][0]
    print(con, total0, total1, TP, TN, FP, FN)
    P=(TP/(TP+FP))
    R=(TP/(TP+FN))
    po=(TP+TN)/(TP+TN+FP+FN)
    pyes=((TP+FP)/(TP+TN+FP+FN))*((TP+FN)/(TP+TN+FP+FN))
    pno=((TN+FP)/(TP+TN+FP+FN))*((TN+FN)/(TP+TN+FP+FN))
    pe=pyes+pno
    kappa=(po-pe)/(1-pe)
    mcc=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**0.5)
    print('Precision: ', P)
    print('Recall/sensitivity: ', R)
    print('Specificity', (TN/(TN+FP)))
    print('Fscore: ', ((P*R)/(P+R)))
    print('Cohen\'s Kappa score: ', kappa)
    roc = roc_auc_score(y_test, test_pred)
    print('ROC AUC: %.3f' % roc) 
    print('MCC', mcc) 
    print('\n ----------Prediction result---------- \n')
    for i in range(len(probs)):
        if probs[i][0]<0.5:
            print('Sequence %s is a Type VII effector!' %(i+1))
        else:
            print('Sequence %s is a Non-effector' %(i+1))
#-----------------------------------------------------------------------------

def featureextraction(peptide_file_name,nucleotide_file_name, total):
    import csv
    import requests
    import webbrowser
    from selenium import webdriver
    import re
    import time
    import numpy as np
    
    feature = [[0 for x in range(1727)] for y in range(total)]

    #amino acid feature set
    aa=['A','R','N','D','C','E','Q','G','H','I','L','K','M','F','P','S','T','W','Y','V']

    #conjoint triad initialization
    S1=['A','G','V']
    S2=['I','L','F','P']
    S3=['Y','M','T','S']
    S4=['H','N','Q','W']
    S5=['R','K']
    S6=['D','E']
    S7=['C']

    #hydrophobibity index
    hydro=[]
    N=np.zeros((1,343))
    feature1=np.zeros((1,343))
    f=0
    for i in range(7):
        for j in range(7):
            for k in range(7):
                
                N[0,f]=(((i+1)*100)+((j+1)*10)+(k+1))
                f=f+1

    #dipeptide initialization            
    dipeptide=list()
    for i in range(len(aa)):
        for j in range(len(aa)):
            t=''
            t=aa[i]+aa[j]
            dipeptide.append(t)
            j=j+1
    #print(dipeptide)        
    id=open(peptide_file_name,"r")
    line=id.readline()
    line=id.readline()
    str=''
    count=0
    line_number=0
    print('Extracting features from amino-acid sequences...')
    while line:
        if '>' not in line:
            line = line.rstrip("\n")
            str=str+line
        if '>' in line:
      
            #single amino acid count
            for i in range(len(aa)):
                
                feature[line_number][i]=round(str.count(aa[i])/len(str),4)

            #dipeptide count    
            for i in range(len(dipeptide)):
                
                feature[line_number][i+20]=round(str.count(dipeptide[i])/len(str),4)

            #physicochemical prroperties
            # Charged (DEKHR) 
            feature[line_number][420]=round((str.count('D')+ str.count('E')+str.count('K')+str.count('H')+str.count('R'))/len(str),4)

            #Aliphatic (ILV)
            feature[line_number][421]=round((str.count('I')+ str.count('L')+str.count('V'))/len(str),4)

            # Aromatic (FHWY)
            feature[line_number][422]=round((str.count('F')+str.count('H')+ str.count('W')+str.count('Y'))/len(str),4)

            # Polar (DERKQN)
            feature[line_number][423]=round((str.count('D')+str.count('E')+str.count('R')+ str.count('K')+str.count('Q')+str.count('N'))/len(str),4)
            
            # Neutral (AGHPSTY)
            feature[line_number][424]=round((str.count('A')+str.count('G')+str.count('H')+str.count('P')+ str.count('S')+str.count('T')+str.count('Y'))/len(str),4)
           
            # Hydrophobic (CFILMVW)
            feature[line_number][425]=round((str.count('C')+str.count('F')+str.count('I')+str.count('L')+ str.count('M')+str.count('V')+str.count('W'))/len(str),4)
       
            # + charged (KRH)
            feature[line_number][426]=round((str.count('K')+str.count('R')+str.count('H'))/len(str),4)

            # - charged (DE)
            feature[line_number][427]=round((str.count('D')+str.count('E'))/len(str),4)

            # Tiny (ACDGST)
            feature[line_number][428]=round((str.count('A')+str.count('C')+str.count('D')+str.count('G')+str.count('S')+str.count('T'))/len(str),4)

            # Small (EHILKMNPQV)
            feature[line_number][429]=round((str.count('E')+str.count('H')+str.count('I')+str.count('L')+str.count('K')+str.count('M')+str.count('N')+str.count('P')+str.count('Q')+str.count('V'))/len(str),4)

            # Large (FRWY)
            feature[line_number][430]=round((str.count('F')+str.count('R')+str.count('W')+str.count('Y'))/len(str),4)

            #Transmembrane amino acid
            feature[line_number][431]=round((str.count('I')+str.count('L')+str.count('V')+str.count('A'))/len(str),4)

            #dipole<1.0 (A, G, V, I, L, F, P)
            feature[line_number][432]=round((str.count('A')+str.count('G')+str.count('V')+str.count('I')+str.count('L')+str.count('F')+str.count('P'))/len(str),4)

            #1.0< dipole < 2.0 (Y, M, T, S)
            feature[line_number][433]=round((str.count('Y')+str.count('M')+str.count('T')+str.count('S'))/len(str),4)

            #2.0 < dipole < 3.0 (H, N, Q, W)
            feature[line_number][434]=round((str.count('H')+str.count('N')+str.count('Q')+str.count('W'))/len(str),4)

            #dipole > 3.0 (R, K)
            feature[line_number][435]=round((str.count('R')+str.count('K'))/len(str),4)

            #dipole > 3.0 with opposite orientation (D, E)
            feature[line_number][436]=round((str.count('D')+str.count('E'))/len(str),4)

            #secondary feature
            feature[line_number][437]=round((str.count('E')+str.count('A')+str.count('L')+str.count('M')+str.count('Q')+str.count('K')+str.count('R')+str.count('H'))/len(str),4)
            feature[line_number][438]=round((str.count('V')+str.count('I')+str.count('Y')+str.count('C')+str.count('W')+str.count('F')+str.count('T'))/len(str),4)
            if line_number==3:
                print(feature[line_number][439])  
            feature[line_number][439]=round((str.count('G')+str.count('N')+str.count('P')+str.count('S')+str.count('D'))/len(str),4)
            if line_number==3:
                print(feature[line_number][439])

            feature[line_number][440]=round((str.count('A')+str.count('L')+str.count('F')+str.count('C')+str.count('G')+str.count('I')+str.count('V')+str.count('W'))/len(str),4)
            feature[line_number][441]=round((str.count('R')+str.count('K')+str.count('Q')+str.count('E')+str.count('N')+str.count('D'))/len(str),4)
            feature[line_number][442]=round((str.count('M')+str.count('S')+str.count('P')+str.count('T')+str.count('H')+str.count('Y'))/len(str),4)
            feature[line_number][443]=round((str.count('Q')+str.count('E')+str.count('D'))/len(str),4)
    
            #444-GOLD730101
            feature[line_number][444]=(str.count('A')*0.75)+(str.count('L')*2.4)+(str.count('R')*0.75)+(str.count('K')*1.5)+(str.count('N')*0.69)
            feature[line_number][444]=feature[line_number][444]+(str.count('M')*1.3)+(str.count('D')*0)+(str.count('C')*1)+(str.count('F')*2.65)+(str.count('P')*2.6)
            feature[line_number][444]=feature[line_number][444]+(str.count('Q')*0.59)+(str.count('S')*0)+(str.count('E')*0)+(str.count('T')*0.45)+(str.count('G')*0)
            feature[line_number][444]=feature[line_number][444]+(str.count('W')*3)+(str.count('H')*0)+(str.count('Y')*2.85)+(str.count('I')*2.95)+(str.count('V')*1.7)
            feature[line_number][444]=round((feature[line_number][444]/len(str)),4)
            
            #445-BIGC670101 
            feature[line_number][445]=(str.count('A')*52.6)+(str.count('L')*102)+(str.count('R')*109.1)+(str.count('K')*105.1)+(str.count('N')*75.7)
            feature[line_number][445]=feature[line_number][445]+(str.count('M')*97.7)+(str.count('D')*68.4)+(str.count('C')*68.3)+(str.count('F')*113.9)+(str.count('P')*73.6)
            feature[line_number][445]=feature[line_number][445]+(str.count('Q')*89.7)+(str.count('S')*54.9)+(str.count('E')*84.7)+(str.count('T')*71.2)+(str.count('G')*36.3)
            feature[line_number][445]=feature[line_number][445]+(str.count('W')*135.4)+(str.count('H')*91.9)+(str.count('Y')*116.2)+(str.count('I')*102)+(str.count('V')*85.1)
            feature[line_number][445]=round((feature[line_number][445]/len(str)),4)

            #446-BULH740101 
            feature[line_number][446]=(str.count('A')*-0.2)+(str.count('L')*-2.46)+(str.count('R')*-0.12)+(str.count('K')*-0.35)+(str.count('N')*0.08)
            feature[line_number][446]=feature[line_number][446]+(str.count('M')*-1.47)+(str.count('D')*-0.2)+(str.count('C')*-0.45)+(str.count('F')*-2.33)+(str.count('P')*-0.98)
            feature[line_number][446]=feature[line_number][446]+(str.count('Q')*0.16)+(str.count('S')*-0.39)+(str.count('E')*-0.3)+(str.count('T')*-0.52)+(str.count('G')*0)
            feature[line_number][446]=feature[line_number][446]+(str.count('W')*-2.01)+(str.count('H')*-0.12)+(str.count('Y')*-2.24)+(str.count('I')*-2.26)+(str.count('V')*-1.56)
            feature[line_number][446]=round((feature[line_number][446]/len(str)),4)

            #447-BULH740102 
            feature[line_number][447]=(str.count('A')*0.691)+(str.count('L')*0.842)+(str.count('R')*0.728)+(str.count('K')*0.767)+(str.count('N')*0.596)
            feature[line_number][447]=feature[line_number][447]+(str.count('M')*0.709)+(str.count('D')*0.558)+(str.count('C')*0.624)+(str.count('F')*0.756)+(str.count('P')*0.73)
            feature[line_number][447]=feature[line_number][447]+(str.count('Q')*0.649)+(str.count('S')*0.594)+(str.count('E')*0.632)+(str.count('T')*0.632)+(str.count('G')*0.592)
            feature[line_number][447]=feature[line_number][447]+(str.count('W')*0.743)+(str.count('H')*0.646)+(str.count('Y')*0.743)+(str.count('I')*0.809)+(str.count('V')*0.777)
            feature[line_number][447]=round((feature[line_number][447]/len(str)),4)

            #448-CHAM810101
            feature[line_number][448]=(str.count('A')*0.52)+(str.count('L')*0.98)+(str.count('R')*0.68)+(str.count('K')*0.68)+(str.count('N')*0.76)
            feature[line_number][448]=feature[line_number][448]+(str.count('M')*0.78)+(str.count('D')*0.76)+(str.count('C')*0.62)+(str.count('F')*0.7)+(str.count('P')*0.36)
            feature[line_number][448]=feature[line_number][448]+(str.count('Q')*0.68)+(str.count('S')*0.53)+(str.count('E')*0.68)+(str.count('T')*0.5)+(str.count('G')*0)
            feature[line_number][448]=feature[line_number][448]+(str.count('W')*0.7)+(str.count('H')*0.7)+(str.count('Y')*0.7)+(str.count('I')*1.02)+(str.count('V')*0.76)
            feature[line_number][448]=round((feature[line_number][448]/len(str)),4)

            #449-CHOC750101 
            feature[line_number][449]=(str.count('A')*91.5)+(str.count('L')*167.9)+(str.count('R')*202)+(str.count('K')*171.3)+(str.count('N')*135.2)
            feature[line_number][449]=feature[line_number][449]+(str.count('M')*170.8)+(str.count('D')*124.5)+(str.count('C')*117.7)+(str.count('F')*203.4)+(str.count('P')*129.3)
            feature[line_number][449]=feature[line_number][449]+(str.count('Q')*161.1)+(str.count('S')*99.1)+(str.count('E')*155.1)+(str.count('T')*122.1)+(str.count('G')*66.4)
            feature[line_number][449]=feature[line_number][449]+(str.count('W')*237.6)+(str.count('H')*167.3)+(str.count('Y')*203.6)+(str.count('I')*203.6)+(str.count('V')*141.7)
            feature[line_number][449]=round((feature[line_number][449]/len(str)),4)

            #450-CHOC760101
            feature[line_number][450]=(str.count('A')*115)+(str.count('L')*170)+(str.count('R')*225)+(str.count('K')*200)+(str.count('N')*160)
            feature[line_number][450]=feature[line_number][450]+(str.count('M')*185)+(str.count('D')*150)+(str.count('C')*135)+(str.count('F')*210)+(str.count('P')*145)
            feature[line_number][450]=feature[line_number][450]+(str.count('Q')*180)+(str.count('S')*115)+(str.count('E')*190)+(str.count('T')*140)+(str.count('G')*75)
            feature[line_number][450]=feature[line_number][450]+(str.count('W')*255)+(str.count('H')*195)+(str.count('Y')*230)+(str.count('I')*175)+(str.count('V')*155)
            feature[line_number][450]=round((feature[line_number][450]/len(str)),4)

            #451-EISD860101 
            feature[line_number][451]=(str.count('A')*0.67)+(str.count('L')*1.9)+(str.count('R')*-2.1)+(str.count('K')*-0.57)+(str.count('N')*-0.6)
            feature[line_number][451]=feature[line_number][451]+(str.count('M')*2.4)+(str.count('D')*-1.2)+(str.count('C')*0.38)+(str.count('F')*2.3)+(str.count('P')*1.2)
            feature[line_number][451]=feature[line_number][451]+(str.count('Q')*-0.22)+(str.count('S')*0.01)+(str.count('E')*-0.76)+(str.count('T')*0.52)+(str.count('G')*0)
            feature[line_number][451]=feature[line_number][451]+(str.count('W')*2.6)+(str.count('H')*0.64)+(str.count('Y')*1.6)+(str.count('I')*1.9)+(str.count('V')*1.5)
            feature[line_number][451]=round((feature[line_number][451]/len(str)),4)

            #452-FASG760101 
            feature[line_number][452]=(str.count('A')*89.09)+(str.count('L')*131.17)+(str.count('R')*174.2)+(str.count('K')*146.19)+(str.count('N')*132.12)
            feature[line_number][452]=feature[line_number][452]+(str.count('M')*149.21)+(str.count('D')*133.1)+(str.count('C')*121.15)+(str.count('F')*165.19)+(str.count('P')*115.13)
            feature[line_number][452]=feature[line_number][452]+(str.count('Q')*146.15)+(str.count('S')*105.09)+(str.count('E')*147.13)+(str.count('T')*119.12)+(str.count('G')*75.07)
            feature[line_number][452]=feature[line_number][452]+(str.count('W')*204.24)+(str.count('H')*155.16)+(str.count('Y')*181.19)+(str.count('I')*131.17)+(str.count('V')*117.15)
            feature[line_number][452]=round((feature[line_number][452]/len(str)),4)

            #453-FASG760102 
            feature[line_number][453]=(str.count('A')*297)+(str.count('L')*337)+(str.count('R')*238)+(str.count('K')*224)+(str.count('N')*236)
            feature[line_number][453]=feature[line_number][453]+(str.count('M')*283)+(str.count('D')*270)+(str.count('C')*178)+(str.count('F')*284)+(str.count('P')*222)
            feature[line_number][453]=feature[line_number][453]+(str.count('Q')*185)+(str.count('S')*228)+(str.count('E')*249)+(str.count('T')*253)+(str.count('G')*290)
            feature[line_number][453]=feature[line_number][453]+(str.count('W')*282)+(str.count('H')*277)+(str.count('Y')*344)+(str.count('I')*284)+(str.count('V')*293)
            feature[line_number][453]=round((feature[line_number][453]/len(str)),4)

            #454-JANJ780102 
            feature[line_number][454]=(str.count('A')*51)+(str.count('L')*60)+(str.count('R')*5)+(str.count('K')*3)+(str.count('N')*22)
            feature[line_number][454]=feature[line_number][454]+(str.count('M')*52)+(str.count('D')*19)+(str.count('C')*74)+(str.count('F')*58)+(str.count('P')*25)
            feature[line_number][454]=feature[line_number][454]+(str.count('Q')*16)+(str.count('S')*35)+(str.count('E')*16)+(str.count('T')*30)+(str.count('G')*52)
            feature[line_number][454]=feature[line_number][454]+(str.count('W')*49)+(str.count('H')*34)+(str.count('Y')*24)+(str.count('I')*66)+(str.count('V')*64)
            feature[line_number][454]=round((feature[line_number][454]/len(str)),4)

            #455-JANJ780103 
            feature[line_number][455]=(str.count('A')*15)+(str.count('L')*16)+(str.count('R')*67)+(str.count('K')*85)+(str.count('N')*49)
            feature[line_number][455]=feature[line_number][455]+(str.count('M')*20)+(str.count('D')*50)+(str.count('C')*5)+(str.count('F')*10)+(str.count('P')*45)
            feature[line_number][455]=feature[line_number][455]+(str.count('Q')*56)+(str.count('S')*32)+(str.count('E')*55)+(str.count('T')*32)+(str.count('G')*10)
            feature[line_number][455]=feature[line_number][455]+(str.count('W')*17)+(str.count('H')*34)+(str.count('Y')*41)+(str.count('I')*13)+(str.count('V')*14)
            feature[line_number][455]=round((feature[line_number][455]/len(str)),4)

            #456-d1
            feature[line_number][456]=(str.count('A')*2)+(str.count('L')*5)+(str.count('R')*8)+(str.count('K')*6)+(str.count('N')*5)
            feature[line_number][456]=feature[line_number][456]+(str.count('M')*5)+(str.count('D')*5)+(str.count('C')*3)+(str.count('F')*8)+(str.count('P')*4)
            feature[line_number][456]=feature[line_number][456]+(str.count('Q')*6)+(str.count('S')*3)+(str.count('E')*6)+(str.count('T')*4)+(str.count('G')*1)
            feature[line_number][456]=feature[line_number][456]+(str.count('W')*11)+(str.count('H')*7)+(str.count('Y')*9)+(str.count('I')*5)+(str.count('V')*4)
            feature[line_number][456]=round((feature[line_number][456]/len(str)),4)

            #457-d2
            feature[line_number][457]=(str.count('A')*1)+(str.count('L')*4)+(str.count('R')*7)+(str.count('K')*5)+(str.count('N')*4)
            feature[line_number][457]=feature[line_number][457]+(str.count('M')*4)+(str.count('D')*4)+(str.count('C')*2)+(str.count('F')*8)+(str.count('P')*4)
            feature[line_number][457]=feature[line_number][457]+(str.count('Q')*5)+(str.count('S')*2)+(str.count('E')*5)+(str.count('T')*3)+(str.count('G')*0)
            feature[line_number][457]=feature[line_number][457]+(str.count('W')*12)+(str.count('H')*6)+(str.count('Y')*9)+(str.count('I')*4)+(str.count('V')*3)
            feature[line_number][457]=round((feature[line_number][457]/len(str)),4)

            #458-d3
            feature[line_number][458]=(str.count('A')*2)+(str.count('L')*8)+(str.count('R')*12)+(str.count('K')*10)+(str.count('N')*8)
            feature[line_number][458]=feature[line_number][458]+(str.count('M')*8)+(str.count('D')*8)+(str.count('C')*4)+(str.count('F')*14)+(str.count('P')*8)
            feature[line_number][458]=feature[line_number][458]+(str.count('Q')*10)+(str.count('S')*4)+(str.count('E')*10)+(str.count('T')*6)+(str.count('G')*0)
            feature[line_number][458]=feature[line_number][458]+(str.count('W')*24)+(str.count('H')*14)+(str.count('Y')*18)+(str.count('I')*8)+(str.count('V')*6)
            feature[line_number][458]=round((feature[line_number][458]/len(str)),4)

            #459-d4
            feature[line_number][459]=(str.count('A')*1)+(str.count('L')*4)+(str.count('R')*6)+(str.count('K')*4)+(str.count('N')*4)
            feature[line_number][459]=feature[line_number][459]+(str.count('M')*4)+(str.count('D')*4)+(str.count('C')*2)+(str.count('F')*6)+(str.count('P')*4)
            feature[line_number][459]=feature[line_number][459]+(str.count('Q')*4)+(str.count('S')*2)+(str.count('E')*5)+(str.count('T')*3)+(str.count('G')*1)
            feature[line_number][459]=feature[line_number][459]+(str.count('W')*8)+(str.count('H')*6)+(str.count('Y')*7)+(str.count('I')*4)+(str.count('V')*3)
            feature[line_number][459]=round((feature[line_number][459]/len(str)),4)

            #460-d5
            feature[line_number][460]=(str.count('A')*1)+(str.count('L')*5)+(str.count('R')*8.12)+(str.count('K')*7)+(str.count('N')*5)
            feature[line_number][460]=feature[line_number][460]+(str.count('M')*5.4)+(str.count('D')*5.17)+(str.count('C')*2.33)+(str.count('F')*7)+(str.count('P')*4)
            feature[line_number][460]=feature[line_number][460]+(str.count('Q')*5.86)+(str.count('S')*1.67)+(str.count('E')*6)+(str.count('T')*3.25)+(str.count('G')*0)
            feature[line_number][460]=feature[line_number][460]+(str.count('W')*11.1)+(str.count('H')*6.71)+(str.count('Y')*8.88)+(str.count('I')*3.25)+(str.count('V')*3.25)
            feature[line_number][460]=round((feature[line_number][460]/len(str)),4)

            #461-d6
            feature[line_number][461]=(str.count('A')*1)+(str.count('L')*3)+(str.count('R')*6)+(str.count('K')*5)+(str.count('N')*3)
            feature[line_number][461]=feature[line_number][461]+(str.count('M')*3)+(str.count('D')*3)+(str.count('C')*1)+(str.count('F')*6)+(str.count('P')*4)
            feature[line_number][461]=feature[line_number][461]+(str.count('Q')*4)+(str.count('S')*2)+(str.count('E')*4)+(str.count('T')*1)+(str.count('G')*0)
            feature[line_number][461]=feature[line_number][461]+(str.count('W')*9)+(str.count('H')*6)+(str.count('Y')*6)+(str.count('I')*3)+(str.count('V')*1)
            feature[line_number][461]=round((feature[line_number][461]/len(str)),4)

            #462-d7
            feature[line_number][462]=(str.count('A')*1)+(str.count('L')*6)+(str.count('R')*12)+(str.count('K')*9)+(str.count('N')*6)
            feature[line_number][462]=feature[line_number][462]+(str.count('M')*7)+(str.count('D')*6)+(str.count('C')*3)+(str.count('F')*11)+(str.count('P')*4)
            feature[line_number][462]=feature[line_number][462]+(str.count('Q')*8)+(str.count('S')*3)+(str.count('E')*8)+(str.count('T')*4)+(str.count('G')*0)
            feature[line_number][462]=feature[line_number][462]+(str.count('W')*14)+(str.count('H')*9)+(str.count('Y')*13)+(str.count('I')*6)+(str.count('V')*4)
            feature[line_number][462]=round((feature[line_number][462]/len(str)),4)

            #463-d8
            feature[line_number][463]=(str.count('A')*1)+(str.count('L')*1.6)+(str.count('R')*1.5)+(str.count('K')*1.667)+(str.count('N')*1.6)
            feature[line_number][463]=feature[line_number][463]+(str.count('M')*1.6)+(str.count('D')*1.6)+(str.count('C')*1.333)+(str.count('F')*1.75)+(str.count('P')*2)
            feature[line_number][463]=feature[line_number][463]+(str.count('Q')*1.667)+(str.count('S')*1.333)+(str.count('E')*1.667)+(str.count('T')*1.5)+(str.count('G')*0)
            feature[line_number][463]=feature[line_number][463]+(str.count('W')*2.182)+(str.count('H')*2)+(str.count('Y')*2)+(str.count('I')*1.6)+(str.count('V')*1.5)
            feature[line_number][463]=round((feature[line_number][463]/len(str)),4)

            #464-d9
            feature[line_number][464]=(str.count('A')*2)+(str.count('L')*11.029)+(str.count('R')*12.499)+(str.count('K')*10.363)+(str.count('N')*11.539)
            feature[line_number][464]=feature[line_number][464]+(str.count('M')*9.49)+(str.count('D')*11.539)+(str.count('C')*6.243)+(str.count('F')*14.851)+(str.count('P')*12)
            feature[line_number][464]=feature[line_number][464]+(str.count('Q')*12.207)+(str.count('S')*5)+(str.count('E')*11.53)+(str.count('T')*9.928)+(str.count('G')*0)
            feature[line_number][464]=feature[line_number][464]+(str.count('W')*13.511)+(str.count('H')*12.448)+(str.count('Y')*12.868)+(str.count('I')*10.851)+(str.count('V')*9.928)
            feature[line_number][464]=round((feature[line_number][464]/len(str)),4)

            #465-d10
            feature[line_number][465]=(str.count('A')*0)+(str.count('L')*4.729)+(str.count('R')*-4.307)+(str.count('K')*-3.151)+(str.count('N')*-4.178)
            feature[line_number][465]=feature[line_number][465]+(str.count('M')*-2.812)+(str.count('D')*-4.178)+(str.count('C')*-2.243)+(str.count('F')*-4.801)+(str.count('P')*-4)
            feature[line_number][465]=feature[line_number][465]+(str.count('Q')*-4.255)+(str.count('S')*1)+(str.count('E')*-3.425)+(str.count('T')*-3.928)+(str.count('G')*0)
            feature[line_number][465]=feature[line_number][465]+(str.count('W')*-6.324)+(str.count('H')*-3.721)+(str.count('Y')*-4.793)+(str.count('I')*-6.085)+(str.count('V')*-3.928)
            feature[line_number][465]=round((feature[line_number][465]/len(str)),4)

            #466-d11
            feature[line_number][466]=(str.count('A')*1)+(str.count('L')*3.2)+(str.count('R')*3.5)+(str.count('K')*3)+(str.count('N')*3.2)
            feature[line_number][466]=feature[line_number][466]+(str.count('M')*2.8)+(str.count('D')*3.2)+(str.count('C')*2)+(str.count('F')*4.25)+(str.count('P')*4)
            feature[line_number][466]=feature[line_number][466]+(str.count('Q')*3.333)+(str.count('S')*2)+(str.count('E')*3.333)+(str.count('T')*3)+(str.count('G')*0)
            feature[line_number][466]=feature[line_number][466]+(str.count('W')*4)+(str.count('H')*4.286)+(str.count('Y')*4.333)+(str.count('I')*1.8)+(str.count('V')*3)
            feature[line_number][466]=round((feature[line_number][466]/len(str)),4)

            #467-d12
            feature[line_number][467]=(str.count('A')*2)+(str.count('L')*1.052)+(str.count('R')*-2.59)+(str.count('K')*-0.536)+(str.count('N')*0.528)
            feature[line_number][467]=feature[line_number][467]+(str.count('M')*0.678)+(str.count('D')*0.528)+(str.count('C')*2)+(str.count('F')*-1.672)+(str.count('P')*4)
            feature[line_number][467]=feature[line_number][467]+(str.count('Q')*-1.043)+(str.count('S')*2)+(str.count('E')*-0.538)+(str.count('T')*3)+(str.count('G')*0)
            feature[line_number][467]=feature[line_number][467]+(str.count('W')*-2.576)+(str.count('H')*-1.185)+(str.count('Y')*-2.054)+(str.count('I')*-1.517)+(str.count('V')*3)
            feature[line_number][467]=round((feature[line_number][467]/len(str)),4)

            #468-d13
            feature[line_number][468]=(str.count('A')*6)+(str.count('L')*12)+(str.count('R')*19)+(str.count('K')*12)+(str.count('N')*12)
            feature[line_number][468]=feature[line_number][468]+(str.count('M')*18)+(str.count('D')*12)+(str.count('C')*6)+(str.count('F')*18)+(str.count('P')*12)
            feature[line_number][468]=feature[line_number][468]+(str.count('Q')*12)+(str.count('S')*6)+(str.count('E')*12)+(str.count('T')*6)+(str.count('G')*1)
            feature[line_number][468]=feature[line_number][468]+(str.count('W')*24)+(str.count('H')*15)+(str.count('Y')*18)+(str.count('I')*12)+(str.count('V')*6)
            feature[line_number][468]=round((feature[line_number][468]/len(str)),4)

            #469-d14
            feature[line_number][469]=(str.count('A')*6)+(str.count('L')*15.6)+(str.count('R')*31.444)+(str.count('K')*24.5)+(str.count('N')*16.5)
            feature[line_number][469]=feature[line_number][469]+(str.count('M')*27.2)+(str.count('D')*16.4)+(str.count('C')*16.67)+(str.count('F')*23.25)+(str.count('P')*12)
            feature[line_number][469]=feature[line_number][469]+(str.count('Q')*21.167)+(str.count('S')*13.33)+(str.count('E')*21)+(str.count('T')*12.4)+(str.count('G')*3.5)
            feature[line_number][469]=feature[line_number][469]+(str.count('W')*27.5)+(str.count('H')*23.1)+(str.count('Y')*27.78)+(str.count('I')*15.6)+(str.count('V')*10.5)
            feature[line_number][469]=round((feature[line_number][469]/len(str)),4)

            #470-d15
            feature[line_number][470]=(str.count('A')*6)+(str.count('L')*12)+(str.count('R')*20)+(str.count('K')*18)+(str.count('N')*14)
            feature[line_number][470]=feature[line_number][470]+(str.count('M')*18)+(str.count('D')*12)+(str.count('C')*12)+(str.count('F')*18)+(str.count('P')*12)
            feature[line_number][470]=feature[line_number][470]+(str.count('Q')*15)+(str.count('S')*8)+(str.count('E')*14)+(str.count('T')*8)+(str.count('G')*1)
            feature[line_number][470]=feature[line_number][470]+(str.count('W')*18)+(str.count('H')*18)+(str.count('Y')*20)+(str.count('I')*12)+(str.count('V')*6)
            feature[line_number][470]=round((feature[line_number][470]/len(str)),4)

            #471-d16
            feature[line_number][471]=(str.count('A')*6)+(str.count('L')*18)+(str.count('R')*38)+(str.count('K')*31)+(str.count('N')*20)
            feature[line_number][471]=feature[line_number][471]+(str.count('M')*34)+(str.count('D')*20)+(str.count('C')*22)+(str.count('F')*24)+(str.count('P')*12)
            feature[line_number][471]=feature[line_number][471]+(str.count('Q')*24)+(str.count('S')*20)+(str.count('E')*26)+(str.count('T')*14)+(str.count('G')*6)
            feature[line_number][471]=feature[line_number][471]+(str.count('W')*36)+(str.count('H')*31)+(str.count('Y')*38)+(str.count('I')*18)+(str.count('V')*12)
            feature[line_number][471]=round((feature[line_number][471]/len(str)),4)

            #472-d17
            feature[line_number][472]=(str.count('A')*12)+(str.count('L')*30)+(str.count('R')*45)+(str.count('K')*37)+(str.count('N')*33.007)
            feature[line_number][472]=feature[line_number][472]+(str.count('M')*40)+(str.count('D')*34)+(str.count('C')*28)+(str.count('F')*48)+(str.count('P')*24)
            feature[line_number][472]=feature[line_number][472]+(str.count('Q')*39)+(str.count('S')*22)+(str.count('E')*40)+(str.count('T')*27)+(str.count('G')*7)
            feature[line_number][472]=feature[line_number][472]+(str.count('W')*68)+(str.count('H')*47)+(str.count('Y')*56)+(str.count('I')*30)+(str.count('V')*24.007)
            feature[line_number][472]=round((feature[line_number][472]/len(str)),4)

            #473-d18
            feature[line_number][473]=(str.count('A')*6)+(str.count('L')*6)+(str.count('R')*5)+(str.count('K')*6.17)+(str.count('N')*6.6)
            feature[line_number][473]=feature[line_number][473]+(str.count('M')*8)+(str.count('D')*6.8)+(str.count('C')*9.33)+(str.count('F')*6)+(str.count('P')*6)
            feature[line_number][473]=feature[line_number][473]+(str.count('Q')*6.5)+(str.count('S')*7.33)+(str.count('E')*6.67)+(str.count('T')*5.4)+(str.count('G')*3.5)
            feature[line_number][473]=feature[line_number][473]+(str.count('W')*5.667)+(str.count('H')*4.7)+(str.count('Y')*6.22)+(str.count('I')*6)+(str.count('V')*6)
            feature[line_number][473]=round((feature[line_number][473]/len(str)),4)

            #474-d19
            feature[line_number][474]=(str.count('A')*12)+(str.count('L')*25.021)+(str.count('R')*23.343)+(str.count('K')*22.739)+(str.count('N')*27.708)
            feature[line_number][474]=feature[line_number][474]+(str.count('M')*31.344)+(str.count('D')*28.634)+(str.count('C')*28)+(str.count('F')*26.993)+(str.count('P')*24)
            feature[line_number][474]=feature[line_number][474]+(str.count('Q')*27.831)+(str.count('S')*20)+(str.count('E')*28.731)+(str.count('T')*23.819)+(str.count('G')*7)
            feature[line_number][474]=feature[line_number][474]+(str.count('W')*29.778)+(str.count('H')*24.243)+(str.count('Y')*28.252)+(str.count('I')*24.841)+(str.count('V')*24)
            feature[line_number][474]=round((feature[line_number][474]/len(str)),4)

            #475-d20
            feature[line_number][475]=(str.count('A')*0)+(str.count('L')*0)+(str.count('R')*0)+(str.count('K')*-0.179)+(str.count('N')*0)
            feature[line_number][475]=feature[line_number][475]+(str.count('M')*0)+(str.count('D')*0)+(str.count('C')*0)+(str.count('F')*0)+(str.count('P')*0)
            feature[line_number][475]=feature[line_number][475]+(str.count('Q')*0)+(str.count('S')*0)+(str.count('E')*0)+(str.count('T')*-4.227)+(str.count('G')*0)
            feature[line_number][475]=feature[line_number][475]+(str.count('W')*0.211)+(str.count('H')*-1.734)+(str.count('Y')*-0.96)+(str.count('I')*-1.641)+(str.count('V')*0)
            feature[line_number][475]=round((feature[line_number][475]/len(str)),4)

            #476-d21
            feature[line_number][476]=(str.count('A')*6)+(str.count('L')*9.6)+(str.count('R')*10.667)+(str.count('K')*10.167)+(str.count('N')*10)
            feature[line_number][476]=feature[line_number][476]+(str.count('M')*13.6)+(str.count('D')*10.4)+(str.count('C')*11.333)+(str.count('F')*12)+(str.count('P')*12)
            feature[line_number][476]=feature[line_number][476]+(str.count('Q')*10.5)+(str.count('S')*8.667)+(str.count('E')*10.667)+(str.count('T')*9)+(str.count('G')*3.5)
            feature[line_number][476]=feature[line_number][476]+(str.count('W')*12.75)+(str.count('H')*10.4)+(str.count('Y')*12.222)+(str.count('I')*9.6)+(str.count('V')*9)
            feature[line_number][476]=round((feature[line_number][476]/len(str)),4)

            #477-d22
            feature[line_number][477]=(str.count('A')*0)+(str.count('L')*3.113)+(str.count('R')*4.2)+(str.count('K')*1.372)+(str.count('N')*3)
            feature[line_number][477]=feature[line_number][477]+(str.count('M')*2.656)+(str.count('D')*2.969)+(str.count('C')*6)+(str.count('F')*2.026)+(str.count('P')*12)
            feature[line_number][477]=feature[line_number][477]+(str.count('Q')*1.849)+(str.count('S')*6)+(str.count('E')*1.822)+(str.count('T')*6)+(str.count('G')*0)
            feature[line_number][477]=feature[line_number][477]+(str.count('W')*2.044)+(str.count('H')*1.605)+(str.count('Y')*1.599)+(str.count('I')*3.373)+(str.count('V')*6)
            feature[line_number][477]=round((feature[line_number][477]/len(str)),4)
            
            #478  ARGP820102 
            feature[line_number][478]=(str.count('A')*1.18)+(str.count('L')*3.23)+(str.count('R')*0.2)+(str.count('K')*0.06)+(str.count('N')*0.23)
            feature[line_number][478]=feature[line_number][478]+(str.count('M')*2.67)+(str.count('D')*0.05)+(str.count('C')*1.89)+(str.count('F')*1.96)+(str.count('P')*0.76)
            feature[line_number][478]=feature[line_number][478]+(str.count('Q')*0.72)+(str.count('S')*0.97)+(str.count('E')*0.11)+(str.count('T')*0.84)+(str.count('G')*0.49)
            feature[line_number][478]=feature[line_number][478]+(str.count('W')*0.77)+(str.count('H')*0.31)+(str.count('Y')*0.39)+(str.count('I')*1.45)+(str.count('V')*1.08)
            feature[line_number][478]=round((feature[line_number][478]/len(str)),4)
            
            #479 ARGP820103
            feature[line_number][479]=(str.count('A')*1.56)+(str.count('L')*2.93)+(str.count('R')*0.45)+(str.count('K')*0.15)+(str.count('N')*0.27)
            feature[line_number][479]=feature[line_number][479]+(str.count('M')*2.96)+(str.count('D')*0.14)+(str.count('C')*1.23)+(str.count('F')*2.03)+(str.count('P')*0.76)
            feature[line_number][479]=feature[line_number][479]+(str.count('Q')*0.51)+(str.count('S')*0.81)+(str.count('E')*0.23)+(str.count('T')*0.91)+(str.count('G')*0.62)
            feature[line_number][479]=feature[line_number][479]+(str.count('W')*1.08)+(str.count('H')*0.29)+(str.count('Y')*0.68)+(str.count('I')*1.67)+(str.count('V')*1.14)
            feature[line_number][479]=round((feature[line_number][479]/len(str)),4)

            #480  BHAR452101 
            feature[line_number][480]=(str.count('A')*0.357)+(str.count('L')*0.365)+(str.count('R')*0.529)+(str.count('K')*0.466)+(str.count('N')*0.463)
            feature[line_number][480]=feature[line_number][480]+(str.count('M')*0.295)+(str.count('D')*0.511)+(str.count('C')*0.346)+(str.count('F')*0.314)+(str.count('P')*0.509)
            feature[line_number][480]=feature[line_number][480]+(str.count('Q')*0.493)+(str.count('S')*0.507)+(str.count('E')*0.497)+(str.count('T')*0.444)+(str.count('G')*0.544)
            feature[line_number][480]=feature[line_number][480]+(str.count('W')*0.305)+(str.count('H')*0.323)+(str.count('Y')*0.42)+(str.count('I')*0.462)+(str.count('V')*0.386)
            feature[line_number][480]=round((feature[line_number][480]/len(str)),4)

            #481 CHAM820101 
            feature[line_number][481]=(str.count('A')*0.046)+(str.count('L')*0.186)+(str.count('R')*0.291)+(str.count('K')*0.219)+(str.count('N')*0.134)
            feature[line_number][481]=feature[line_number][481]+(str.count('M')*0.221)+(str.count('D')*0.105)+(str.count('C')*0.128)+(str.count('F')*0.29)+(str.count('P')*0.131)
            feature[line_number][481]=feature[line_number][481]+(str.count('Q')*0.18)+(str.count('S')*0.062)+(str.count('E')*0.151)+(str.count('T')*0.108)+(str.count('G')*0)
            feature[line_number][481]=feature[line_number][481]+(str.count('W')*0.409)+(str.count('H')*0.23)+(str.count('Y')*0.298)+(str.count('I')*0.186)+(str.count('V')*0.14)
            feature[line_number][481]=round((feature[line_number][481]/len(str)),4)

            #482 CHAM820102 
            feature[line_number][482]=(str.count('A')*-0.368)+(str.count('L')*1.07)+(str.count('R')*-1.03)+(str.count('K')*0)+(str.count('N')*0)
            feature[line_number][482]=feature[line_number][482]+(str.count('M')*0.656)+(str.count('D')*2.06)+(str.count('C')*4.53)+(str.count('F')*1.06)+(str.count('P')*-2.24)
            feature[line_number][482]=feature[line_number][482]+(str.count('Q')*0.731)+(str.count('S')*-0.524)+(str.count('E')*1.77)+(str.count('T')*0)+(str.count('G')*-0.525)
            feature[line_number][482]=feature[line_number][482]+(str.count('W')*1.6)+(str.count('H')*0)+(str.count('Y')*4.91)+(str.count('I')*0.791)+(str.count('V')*0.401)
            feature[line_number][482]=round((feature[line_number][482]/len(str)),4)

            #483 DAYM780201 
            feature[line_number][483]=(str.count('A')*100)+(str.count('L')*40)+(str.count('R')*65)+(str.count('K')*56)+(str.count('N')*134)
            feature[line_number][483]=feature[line_number][483]+(str.count('M')*94)+(str.count('D')*106)+(str.count('C')*20)+(str.count('F')*41)+(str.count('P')*56)
            feature[line_number][483]=feature[line_number][483]+(str.count('Q')*93)+(str.count('S')*120)+(str.count('E')*102)+(str.count('T')*97)+(str.count('G')*49)
            feature[line_number][483]=feature[line_number][483]+(str.count('W')*18)+(str.count('H')*66)+(str.count('Y')*41)+(str.count('I')*96)+(str.count('V')*74)
            feature[line_number][483]=round((feature[line_number][483]/len(str)),4)

            #484 EISD860102
            feature[line_number][484]=(str.count('A')*0)+(str.count('L')*1)+(str.count('R')*10)+(str.count('K')*5.7)+(str.count('N')*1.3)
            feature[line_number][484]=feature[line_number][484]+(str.count('M')*1.9)+(str.count('D')*1.9)+(str.count('C')*0.17)+(str.count('F')*1.1)+(str.count('P')*0.18)
            feature[line_number][484]=feature[line_number][484]+(str.count('Q')*1.9)+(str.count('S')*0.73)+(str.count('E')*3)+(str.count('T')*1.5)+(str.count('G')*0)
            feature[line_number][484]=feature[line_number][484]+(str.count('W')*1.6)+(str.count('H')*0.99)+(str.count('Y')*1.8)+(str.count('I')*1.2)+(str.count('V')*0.48)
            feature[line_number][484]=round((feature[line_number][484]/len(str)),4)

            #485 FAUJ452103
            feature[line_number][485]=(str.count('A')*1)+(str.count('L')*4)+(str.count('R')*6.13)+(str.count('K')*4.77)+(str.count('N')*2.95)
            feature[line_number][485]=feature[line_number][485]+(str.count('M')*4.43)+(str.count('D')*2.78)+(str.count('C')*2.43)+(str.count('F')*5.89)+(str.count('P')*2.72)
            feature[line_number][485]=feature[line_number][485]+(str.count('Q')*3.95)+(str.count('S')*1.6)+(str.count('E')*3.78)+(str.count('T')*2.6)+(str.count('G')*0)
            feature[line_number][485]=feature[line_number][485]+(str.count('W')*8.08)+(str.count('H')*4.66)+(str.count('Y')*6.47)+(str.count('I')*4)+(str.count('V')*3)
            feature[line_number][485]=round((feature[line_number][485]/len(str)),4)

            #486 FAUJ452108
            feature[line_number][486]=(str.count('A')*-0.01)+(str.count('L')*-0.01)+(str.count('R')*0.04)+(str.count('K')*0)+(str.count('N')*0.06)
            feature[line_number][486]=feature[line_number][486]+(str.count('M')*0.04)+(str.count('D')*0.15)+(str.count('C')*0.12)+(str.count('F')*0.03)+(str.count('P')*0)
            feature[line_number][486]=feature[line_number][486]+(str.count('Q')*0.05)+(str.count('S')*0.11)+(str.count('E')*0.07)+(str.count('T')*0.04)+(str.count('G')*0)
            feature[line_number][486]=feature[line_number][486]+(str.count('W')*0)+(str.count('H')*0.08)+(str.count('Y')*0.03)+(str.count('I')*-0.01)+(str.count('V')*0.01)
            feature[line_number][486]=round((feature[line_number][486]/len(str)),4)

            #487 GARJ730101 
            feature[line_number][487]=(str.count('A')*0.28)+(str.count('L')*1)+(str.count('R')*0.1)+(str.count('K')*0.09)+(str.count('N')*0.25)
            feature[line_number][487]=feature[line_number][487]+(str.count('M')*0.74)+(str.count('D')*0.21)+(str.count('C')*0.28)+(str.count('F')*2.18)+(str.count('P')*0.39)
            feature[line_number][487]=feature[line_number][487]+(str.count('Q')*0.35)+(str.count('S')*0.12)+(str.count('E')*0.33)+(str.count('T')*0.21)+(str.count('G')*0.17)
            feature[line_number][487]=feature[line_number][487]+(str.count('W')*5.7)+(str.count('H')*0.21)+(str.count('Y')*1.26)+(str.count('I')*0.82)+(str.count('V')*0.6)
            feature[line_number][487]=round((feature[line_number][487]/len(str)),4)

            #488  HOPA770101 
            feature[line_number][488]=(str.count('A')*1)+(str.count('L')*0.8)+(str.count('R')*2.3)+(str.count('K')*5.3)+(str.count('N')*2.2)
            feature[line_number][488]=feature[line_number][488]+(str.count('M')*0.7)+(str.count('D')*6.5)+(str.count('C')*0.1)+(str.count('F')*1.4)+(str.count('P')*0.9)
            feature[line_number][488]=feature[line_number][488]+(str.count('Q')*2.1)+(str.count('S')*1.7)+(str.count('E')*6.2)+(str.count('T')*1.5)+(str.count('G')*1.1)
            feature[line_number][488]=feature[line_number][488]+(str.count('W')*1.9)+(str.count('H')*2.8)+(str.count('Y')*2.1)+(str.count('I')*0.8)+(str.count('V')*0.9)
            feature[line_number][488]=round((feature[line_number][488]/len(str)),4)

            #489 HUTJ700101
            feature[line_number][489]=(str.count('A')*29.22)+(str.count('L')*48.03)+(str.count('R')*26.37)+(str.count('K')*57.1)+(str.count('N')*38.3)
            feature[line_number][489]=feature[line_number][489]+(str.count('M')*69.32)+(str.count('D')*37.09)+(str.count('C')*50.7)+(str.count('F')*48.52)+(str.count('P')*36.13)
            feature[line_number][489]=feature[line_number][489]+(str.count('Q')*44.02)+(str.count('S')*32.4)+(str.count('E')*41.84)+(str.count('T')*35.2)+(str.count('G')*23.71)
            feature[line_number][489]=feature[line_number][489]+(str.count('W')*56.92)+(str.count('H')*59.64)+(str.count('Y')*51.73)+(str.count('I')*45)+(str.count('V')*40.35)
            feature[line_number][489]=round((feature[line_number][489]/len(str)),4)

            #490 HUTJ700102
            feature[line_number][490]=(str.count('A')*30.88)+(str.count('L')*50.62)+(str.count('R')*68.43)+(str.count('K')*63.21)+(str.count('N')*41.7)
            feature[line_number][490]=feature[line_number][490]+(str.count('M')*55.32)+(str.count('D')*40.66)+(str.count('C')*53.83)+(str.count('F')*51.06)+(str.count('P')*39.21)
            feature[line_number][490]=feature[line_number][490]+(str.count('Q')*46.62)+(str.count('S')*35.65)+(str.count('E')*44.98)+(str.count('T')*36.5)+(str.count('G')*24.74)
            feature[line_number][490]=feature[line_number][490]+(str.count('W')*60)+(str.count('H')*65.99)+(str.count('Y')*51.15)+(str.count('I')*49.71)+(str.count('V')*42.75)
            feature[line_number][490]=round((feature[line_number][490]/len(str)),4)

            #491 HUTJ700103
            feature[line_number][491]=(str.count('A')*154.33)+(str.count('L')*232.3)+(str.count('R')*341.01)+(str.count('K')*300.46)+(str.count('N')*207.9)
            feature[line_number][491]=feature[line_number][491]+(str.count('M')*202.65)+(str.count('D')*194.91)+(str.count('C')*219.79)+(str.count('F')*204.74)+(str.count('P')*179.93)
            feature[line_number][491]=feature[line_number][491]+(str.count('Q')*235.51)+(str.count('S')*174.06)+(str.count('E')*223.16)+(str.count('T')*205.8)+(str.count('G')*127.9)
            feature[line_number][491]=feature[line_number][491]+(str.count('W')*237.01)+(str.count('H')*242.54)+(str.count('Y')*229.15)+(str.count('I')*233.21)+(str.count('V')*207.6)
            feature[line_number][491]=round((feature[line_number][491]/len(str)),4)

            #492  MCMT640101 
            feature[line_number][492]=(str.count('A')*4.34)+(str.count('L')*18.78)+(str.count('R')*26.66)+(str.count('K')*21.29)+(str.count('N')*13.28)
            feature[line_number][492]=feature[line_number][492]+(str.count('M')*21.64)+(str.count('D')*12)+(str.count('C')*35.77)+(str.count('F')*29.4)+(str.count('P')*10.93)
            feature[line_number][492]=feature[line_number][492]+(str.count('Q')*17.56)+(str.count('S')*6.35)+(str.count('E')*17.26)+(str.count('T')*11.01)+(str.count('G')*0)
            feature[line_number][492]=feature[line_number][492]+(str.count('W')*42.53)+(str.count('H')*21.81)+(str.count('Y')*31.53)+(str.count('I')*19.06)+(str.count('V')*13.92)
            feature[line_number][492]=round((feature[line_number][492]/len(str)),4)

            #493 MEEJ800101
            feature[line_number][493]=(str.count('A')*0.5)+(str.count('L')*8.8)+(str.count('R')*0.8)+(str.count('K')*0.1)+(str.count('N')*0.8)
            feature[line_number][493]=feature[line_number][493]+(str.count('M')*4.8)+(str.count('D')*-8.2)+(str.count('C')*-6.8)+(str.count('F')*13.2)+(str.count('P')*6.1)
            feature[line_number][493]=feature[line_number][493]+(str.count('Q')*-4.8)+(str.count('S')*1.2)+(str.count('E')*-16.9)+(str.count('T')*2.7)+(str.count('G')*0)
            feature[line_number][493]=feature[line_number][493]+(str.count('W')*14.9)+(str.count('H')*-3.5)+(str.count('Y')*6.1)+(str.count('I')*13.9)+(str.count('V')*2.7)
            feature[line_number][493]=round((feature[line_number][493]/len(str)),4)

            #494 MEEJ800102
            feature[line_number][494]=(str.count('A')*-0.1)+(str.count('L')*10)+(str.count('R')*-4.5)+(str.count('K')*-3.2)+(str.count('N')*-1.6)
            feature[line_number][494]=feature[line_number][494]+(str.count('M')*7.1)+(str.count('D')*-2.8)+(str.count('C')*-2.2)+(str.count('F')*13.9)+(str.count('P')*8)
            feature[line_number][494]=feature[line_number][494]+(str.count('Q')*-2.5)+(str.count('S')*-3.7)+(str.count('E')*-7.5)+(str.count('T')*1.5)+(str.count('G')*-0.5)
            feature[line_number][494]=feature[line_number][494]+(str.count('W')*18.1)+(str.count('H')*0.8)+(str.count('Y')*8.2)+(str.count('I')*11.8)+(str.count('V')*3.3)
            feature[line_number][494]=round((feature[line_number][494]/len(str)),4)

            #495 SNEP660101
            feature[line_number][495]=(str.count('A')*0.239)+(str.count('L')*0.281)+(str.count('R')*0.211)+(str.count('K')*0.228)+(str.count('N')*0.249)
            feature[line_number][495]=feature[line_number][495]+(str.count('M')*0.253)+(str.count('D')*0.171)+(str.count('C')*0.22)+(str.count('F')*0.234)+(str.count('P')*0.165)
            feature[line_number][495]=feature[line_number][495]+(str.count('Q')*0.26)+(str.count('S')*0.236)+(str.count('E')*0.187)+(str.count('T')*0.213)+(str.count('G')*0.16)
            feature[line_number][495]=feature[line_number][495]+(str.count('W')*0.183)+(str.count('H')*0.205)+(str.count('Y')*0.193)+(str.count('I')*0.273)+(str.count('V')*0.255)
            feature[line_number][495]=round((feature[line_number][495]/len(str)),4)

            #496 SNEP660102
            feature[line_number][496]=(str.count('A')*0.33)+(str.count('L')*0.129)+(str.count('R')*-0.176)+(str.count('K')*-0.075)+(str.count('N')*-0.233)
            feature[line_number][496]=feature[line_number][496]+(str.count('M')*-0.092)+(str.count('D')*-0.371)+(str.count('C')*0.074)+(str.count('F')*-0.011)+(str.count('P')*0.37)
            feature[line_number][496]=feature[line_number][496]+(str.count('Q')*-0.254)+(str.count('S')*0.022)+(str.count('E')*-0.409)+(str.count('T')*0.136)+(str.count('G')*0.37)
            feature[line_number][496]=feature[line_number][496]+(str.count('W')*-0.011)+(str.count('H')*-0.078)+(str.count('Y')*-0.138)+(str.count('I')*0.149)+(str.count('V')*0.245)
            feature[line_number][496]=round((feature[line_number][496]/len(str)),4)

            #497 SNEP660103
            feature[line_number][497]=(str.count('A')*-0.11)+(str.count('L')*-0.008)+(str.count('R')*0.079)+(str.count('K')*0.049)+(str.count('N')*-0.136)
            feature[line_number][497]=feature[line_number][497]+(str.count('M')*-0.041)+(str.count('D')*-0.285)+(str.count('C')*-0.184)+(str.count('F')*0.438)+(str.count('P')*-0.016)
            feature[line_number][497]=feature[line_number][497]+(str.count('Q')*-0.067)+(str.count('S')*-0.153)+(str.count('E')*-0.246)+(str.count('T')*-0.208)+(str.count('G')*-0.073)
            feature[line_number][497]=feature[line_number][497]+(str.count('W')*0.493)+(str.count('H')*0.32)+(str.count('Y')*0.381)+(str.count('I')*0.001)+(str.count('V')*-0.155)
            feature[line_number][497]=round((feature[line_number][497]/len(str)),4)

            #498 SNEP660104
            feature[line_number][498]=(str.count('A')*-0.062)+(str.count('L')*-0.264)+(str.count('R')*-0.167)+(str.count('K')*-0.371)+(str.count('N')*0.166)
            feature[line_number][498]=feature[line_number][498]+(str.count('M')*0.077)+(str.count('D')*-0.079)+(str.count('C')*0.38)+(str.count('F')*0.074)+(str.count('P')*-0.036)
            feature[line_number][498]=feature[line_number][498]+(str.count('Q')*-0.025)+(str.count('S')*0.47)+(str.count('E')*-0.184)+(str.count('T')*0.348)+(str.count('G')*-0.017)
            feature[line_number][498]=feature[line_number][498]+(str.count('W')*0.05)+(str.count('H')*0.056)+(str.count('Y')*0.22)+(str.count('I')*-0.309)+(str.count('V')*-0.212)
            feature[line_number][498]=round((feature[line_number][498]/len(str)),4)

            #PSSM
            L=len(str)
            p = [[0 for i in range(20)] for j in range(L)]
            aa=['A','R','N','D','C','E','Q','G','H','I','L','K','M','F','P','S','T','W','Y','V']

            for i in range(L):
                if str[i]=='A':
                    p[i][0]=p[i][0]+1
                if str[i]=='R':
                    p[i][1]=p[i][1]+1
                if str[i]=='N':
                    p[i][2]=p[i][2]+1
                if str[i]=='D':
                    p[i][3]=p[i][3]+1
                if str[i]=='C':
                    p[i][4]=p[i][4]+1
                if str[i]=='E':
                    p[i][5]=p[i][5]+1
                if str[i]=='Q':
                    p[i][6]=p[i][6]+1
                if str[i]=='G':
                    p[i][7]=p[i][7]+1
                if str[i]=='H':
                    p[i][8]=p[i][8]+1
                if str[i]=='I':
                    p[i][9]=p[i][9]+1
                if str[i]=='L':
                    p[i][10]=p[i][10]+1
                if str[i]=='K':
                    p[i][11]=p[i][11]+1
                if str[i]=='M':
                    p[i][12]=p[i][12]+1
                if str[i]=='F':
                    p[i][13]=p[i][13]+1
                if str[i]=='P':
                    p[i][14]=p[i][14]+1
                if str[i]=='S':
                    p[i][15]=p[i][15]+1
                if str[i]=='T':
                    p[i][16]=p[i][16]+1
                if str[i]=='W':
                    p[i][17]=p[i][17]+1
                if str[i]=='Y':
                    p[i][18]=p[i][18]+1
                if str[i]=='V':
                    p[i][19]=p[i][19]+1

            #PSSM    
            pssm=p
            for i in range(L):
                 for j in range(20):
                     if pssm[i][j]>0:
                         pssm[i][j]=math.log((pssm[i][j]/0.05),2)
            
            #FPSSM
            fpssm=pssm
            for i in range(L):
                 for j in range(20):
                     if fpssm[i][j]<0:
                         fpssm[i][j]=0
                     elif fpssm[i][j]>7:
                         fpssm[i][j]=7

##            for i in range(L):
##                print(fpssm[i][:])
                
            #S-FPSSM
            sfpssm = [[0 for i in range(20)] for j in range(20)]
            for i in range(20):
                for j in range(20):
                    total=0
                    for k in range(L-1):
                        if str[k]==aa[i]:
                            total=total+(fpssm[k][j]*1)
                        else:
                            total=total+(fpssm[k][j]*0)
                    sfpssm[i][j]=total
            
                        
            #DPC-PSSM
            dpcpssm  = [[0 for i in range(20)] for j in range(20)]
            for i in range(20):
                for j in range(20):
                    total=0
                    for k in range(L-1):
                       total=total+(p[k][i]*p[k+1][j])
                    total=total/(L-1)
                    dpcpssm[i][j]=total
            cc=1
            for i in range(20):
                for j in range(20):
                    feature[line_number][498+cc]=dpcpssm[i][j]
                    cc=cc+1
            for i in range(20):
                for j in range(20):
                    feature[line_number][498+cc]=sfpssm[i][j]
                    cc=cc+1
            #print(pssm)
            
            #CTD
            for i in range(len(str)-2):
               p=0
               X=str[i]
               Y=str[i+1]
               Z=str[i+2]
               if X in S1:
                 p=1*100
               if X in S2:
                 p=2*100
               if X in S3:
                 p=3*100
               if X in S4:
                 p=4*100
               if X in S5:
                 p=5*100
               if X in S6:
                 p=6*100
               if X in S7:
                 p=7*100   
    
               if Y in S1:
                 p=p+1*10
               if Y in S2:
                 p=p+2*10
               if Y in S3:
                 p=p+3*10
               if Y in S4:
                 p=p+4*10
               if Y in S5:
                 p=p+5*10
               if Y in S6:
                 p=p+6*10
               if Y in S7:
                 p=p+7*10

               if Z in S1:
                 p=p+1
               if Z in S2:
                 p=p+2
               if Z in S3:
                 p=p+3
               if Z in S4:
                 p=p+4
               if Z in S5:
                 p=p+5
               if Z in S6:
                 p=p+6
               if Z in S7:
                 p=p+7
            
               for j in range(343):
                 if p==N[0,j]:
                   k=j
               feature1[0,k]=feature1[0,k]+1
            feature1[0,:]=feature1[0,:]/(len(str)-2)*100
            feature[line_number][1299:1642]=feature1[0,:]
            feature1=np.zeros((1,343))
            str=''
            
            line_number=line_number+1
        line=id.readline()
    id.close()
    
    for i in range(len(aa)):
                feature[line_number][i]=str.count(aa[i])/len(str)
                
    for i in range(len(dipeptide)):
                feature[line_number][i+20]=str.count(dipeptide[i])/len(str)
                
    #physicochemical properties
    # Charged (DEKHR) 
    feature[line_number][420]=round((str.count('D')+ str.count('E')+str.count('K')+str.count('H')+str.count('R')) /len(str),4)

    #Aliphatic (ILV)
    feature[line_number][421]=round((str.count('I')+ str.count('L')+str.count('V'))/len(str),4)

    # Aromatic (FHWY)
    feature[line_number][422]=round((str.count('F')+str.count('H')+ str.count('W')+str.count('Y'))/len(str),4)

    # Polar (DERKQN)
    feature[line_number][423]=round((str.count('D')+str.count('E')+str.count('R')+ str.count('K')+str.count('Q')+str.count('N'))/len(str),4)
            
    # Neutral (AGHPSTY)
    feature[line_number][424]=round((str.count('A')+str.count('G')+str.count('H')+str.count('P')+ str.count('S')+str.count('T')+str.count('Y'))/len(str),4)
           
    # Hydrophobic (CFILMVW)
    feature[line_number][425]=round((str.count('C')+str.count('F')+str.count('I')+str.count('L')+ str.count('M')+str.count('V')+str.count('W'))/len(str),4)
       
    # + charged (KRH)
    feature[line_number][426]=round((str.count('K')+str.count('R')+str.count('H'))/len(str),4)

    # - charged (DE)
    feature[line_number][427]=round((str.count('D')+str.count('E'))/len(str),4)

    # Tiny (ACDGST)
    feature[line_number][428]=round((str.count('A')+str.count('C')+str.count('D')+str.count('G')+str.count('S')+str.count('T'))/len(str),4)

    # Small (EHILKMNPQV)
    feature[line_number][429]=round((str.count('E')+str.count('H')+str.count('I')+str.count('L')+str.count('K')+str.count('M')+str.count('N')+str.count('P')+str.count('Q')+str.count('V'))/len(str),4)

    # Large (FRWY)
    feature[line_number][430]=round((str.count('F')+str.count('R')+str.count('W')+str.count('Y'))/len(str),4)

    #Transmembrane amino acid
    feature[line_number][431]=round((str.count('I')+str.count('L')+str.count('V')+str.count('A'))/len(str),4)

    #dipole<1.0 (A, G, V, I, L, F, P)
    feature[line_number][432]=round((str.count('A')+str.count('G')+str.count('V')+str.count('I')+str.count('L')+str.count('F')+str.count('P'))/len(str),4)

    #1.0< dipole < 2.0 (Y, M, T, S)
    feature[line_number][433]=round((str.count('Y')+str.count('M')+str.count('T')+str.count('S'))/len(str),4)

    #2.0 < dipole < 3.0 (H, N, Q, W)
    feature[line_number][434]=round((str.count('H')+str.count('N')+str.count('Q')+str.count('W'))/len(str),4)

    #dipole > 3.0 (R, K)
    feature[line_number][435]=round((str.count('R')+str.count('K'))/len(str),4)

    #dipole > 3.0 with opposite orientation (D, E)
    feature[line_number][436]=round((str.count('D')+str.count('E'))/len(str),4)

    feature[line_number][437]=round((str.count('E')+str.count('A')+str.count('L')+str.count('M')+str.count('Q')+str.count('K')+str.count('R')+str.count('H'))/len(str),4)
    feature[line_number][438]=round((str.count('V')+str.count('I')+str.count('Y')+str.count('C')+str.count('W')+str.count('F')+str.count('T'))/len(str),4)
    feature[line_number][439]=round((str.count('G')+str.count('N')+str.count('P')+str.count('S')+str.count('D'))/len(str),4)


    feature[line_number][440]=round((str.count('A')+str.count('L')+str.count('F')+str.count('C')+str.count('G')+str.count('I')+str.count('V')+str.count('W'))/len(str),4)
    feature[line_number][441]=round((str.count('R')+str.count('K')+str.count('Q')+str.count('E')+str.count('N')+str.count('D'))/len(str),4)
    feature[line_number][442]=round((str.count('M')+str.count('S')+str.count('P')+str.count('T')+str.count('H')+str.count('Y'))/len(str),4)
    feature[line_number][443]=round((str.count('Q')+str.count('E')+str.count('D'))/len(str),4)
    
    #444-GOLD730101
    feature[line_number][444]=(str.count('A')*0.75)+(str.count('L')*2.4)+(str.count('R')*0.75)+(str.count('K')*1.5)+(str.count('N')*0.69)
    feature[line_number][444]=feature[line_number][444]+(str.count('M')*1.3)+(str.count('D')*0)+(str.count('C')*1)+(str.count('F')*2.65)+(str.count('P')*2.6)
    feature[line_number][444]=feature[line_number][444]+(str.count('Q')*0.59)+(str.count('S')*0)+(str.count('E')*0)+(str.count('T')*0.45)+(str.count('G')*0)
    feature[line_number][444]=feature[line_number][444]+(str.count('W')*3)+(str.count('H')*0)+(str.count('Y')*2.85)+(str.count('I')*2.95)+(str.count('V')*1.7)
    feature[line_number][444]=round((feature[line_number][444]/len(str)),4)
            
    #445-BIGC670101 
    feature[line_number][445]=(str.count('A')*52.6)+(str.count('L')*102)+(str.count('R')*109.1)+(str.count('K')*105.1)+(str.count('N')*75.7)
    feature[line_number][445]=feature[line_number][445]+(str.count('M')*97.7)+(str.count('D')*68.4)+(str.count('C')*68.3)+(str.count('F')*113.9)+(str.count('P')*73.6)
    feature[line_number][445]=feature[line_number][445]+(str.count('Q')*89.7)+(str.count('S')*54.9)+(str.count('E')*84.7)+(str.count('T')*71.2)+(str.count('G')*36.3)
    feature[line_number][445]=feature[line_number][445]+(str.count('W')*135.4)+(str.count('H')*91.9)+(str.count('Y')*116.2)+(str.count('I')*102)+(str.count('V')*85.1)
    feature[line_number][445]=round((feature[line_number][445]/len(str)),4)

    #446-BULH740101 
    feature[line_number][446]=(str.count('A')*-0.2)+(str.count('L')*-2.46)+(str.count('R')*-0.12)+(str.count('K')*-0.35)+(str.count('N')*0.08)
    feature[line_number][446]=feature[line_number][446]+(str.count('M')*-1.47)+(str.count('D')*-0.2)+(str.count('C')*-0.45)+(str.count('F')*-2.33)+(str.count('P')*-0.98)
    feature[line_number][446]=feature[line_number][446]+(str.count('Q')*0.16)+(str.count('S')*-0.39)+(str.count('E')*-0.3)+(str.count('T')*-0.52)+(str.count('G')*0)
    feature[line_number][446]=feature[line_number][446]+(str.count('W')*-2.01)+(str.count('H')*-0.12)+(str.count('Y')*-2.24)+(str.count('I')*-2.26)+(str.count('V')*-1.56)
    feature[line_number][446]=round((feature[line_number][446]/len(str)),4)

    #447-BULH740102 
    feature[line_number][447]=(str.count('A')*0.691)+(str.count('L')*0.842)+(str.count('R')*0.728)+(str.count('K')*0.767)+(str.count('N')*0.596)
    feature[line_number][447]=feature[line_number][447]+(str.count('M')*0.709)+(str.count('D')*0.558)+(str.count('C')*0.624)+(str.count('F')*0.756)+(str.count('P')*0.73)
    feature[line_number][447]=feature[line_number][447]+(str.count('Q')*0.649)+(str.count('S')*0.594)+(str.count('E')*0.632)+(str.count('T')*0.632)+(str.count('G')*0.592)
    feature[line_number][447]=feature[line_number][447]+(str.count('W')*0.743)+(str.count('H')*0.646)+(str.count('Y')*0.743)+(str.count('I')*0.809)+(str.count('V')*0.777)
    feature[line_number][447]=round((feature[line_number][447]/len(str)),4)

    #448-CHAM810101
    feature[line_number][448]=(str.count('A')*0.52)+(str.count('L')*0.98)+(str.count('R')*0.68)+(str.count('K')*0.68)+(str.count('N')*0.76)
    feature[line_number][448]=feature[line_number][448]+(str.count('M')*0.78)+(str.count('D')*0.76)+(str.count('C')*0.62)+(str.count('F')*0.7)+(str.count('P')*0.36)
    feature[line_number][448]=feature[line_number][448]+(str.count('Q')*0.68)+(str.count('S')*0.53)+(str.count('E')*0.68)+(str.count('T')*0.5)+(str.count('G')*0)
    feature[line_number][448]=feature[line_number][448]+(str.count('W')*0.7)+(str.count('H')*0.7)+(str.count('Y')*0.7)+(str.count('I')*1.02)+(str.count('V')*0.76)
    feature[line_number][448]=round((feature[line_number][448]/len(str)),4)

    #449-CHOC750101 
    feature[line_number][449]=(str.count('A')*91.5)+(str.count('L')*167.9)+(str.count('R')*202)+(str.count('K')*171.3)+(str.count('N')*135.2)
    feature[line_number][449]=feature[line_number][449]+(str.count('M')*170.8)+(str.count('D')*124.5)+(str.count('C')*117.7)+(str.count('F')*203.4)+(str.count('P')*129.3)
    feature[line_number][449]=feature[line_number][449]+(str.count('Q')*161.1)+(str.count('S')*99.1)+(str.count('E')*155.1)+(str.count('T')*122.1)+(str.count('G')*66.4)
    feature[line_number][449]=feature[line_number][449]+(str.count('W')*237.6)+(str.count('H')*167.3)+(str.count('Y')*203.6)+(str.count('I')*203.6)+(str.count('V')*141.7)
    feature[line_number][449]=round((feature[line_number][449]/len(str)),4)

    #450-CHOC760101
    feature[line_number][450]=(str.count('A')*115)+(str.count('L')*170)+(str.count('R')*225)+(str.count('K')*200)+(str.count('N')*160)
    feature[line_number][450]=feature[line_number][450]+(str.count('M')*185)+(str.count('D')*150)+(str.count('C')*135)+(str.count('F')*210)+(str.count('P')*145)
    feature[line_number][450]=feature[line_number][450]+(str.count('Q')*180)+(str.count('S')*115)+(str.count('E')*190)+(str.count('T')*140)+(str.count('G')*75)
    feature[line_number][450]=feature[line_number][450]+(str.count('W')*255)+(str.count('H')*195)+(str.count('Y')*230)+(str.count('I')*175)+(str.count('V')*155)
    feature[line_number][450]=round((feature[line_number][450]/len(str)),4)

    #451-EISD860101 
    feature[line_number][451]=(str.count('A')*0.67)+(str.count('L')*1.9)+(str.count('R')*-2.1)+(str.count('K')*-0.57)+(str.count('N')*-0.6)
    feature[line_number][451]=feature[line_number][451]+(str.count('M')*2.4)+(str.count('D')*-1.2)+(str.count('C')*0.38)+(str.count('F')*2.3)+(str.count('P')*1.2)
    feature[line_number][451]=feature[line_number][451]+(str.count('Q')*-0.22)+(str.count('S')*0.01)+(str.count('E')*-0.76)+(str.count('T')*0.52)+(str.count('G')*0)
    feature[line_number][451]=feature[line_number][451]+(str.count('W')*2.6)+(str.count('H')*0.64)+(str.count('Y')*1.6)+(str.count('I')*1.9)+(str.count('V')*1.5)
    feature[line_number][451]=round((feature[line_number][451]/len(str)),4)

    #452-FASG760101 
    feature[line_number][452]=(str.count('A')*89.09)+(str.count('L')*131.17)+(str.count('R')*174.2)+(str.count('K')*146.19)+(str.count('N')*132.12)
    feature[line_number][452]=feature[line_number][452]+(str.count('M')*149.21)+(str.count('D')*133.1)+(str.count('C')*121.15)+(str.count('F')*165.19)+(str.count('P')*115.13)
    feature[line_number][452]=feature[line_number][452]+(str.count('Q')*146.15)+(str.count('S')*105.09)+(str.count('E')*147.13)+(str.count('T')*119.12)+(str.count('G')*75.07)
    feature[line_number][452]=feature[line_number][452]+(str.count('W')*204.24)+(str.count('H')*155.16)+(str.count('Y')*181.19)+(str.count('I')*131.17)+(str.count('V')*117.15)
    feature[line_number][452]=round((feature[line_number][452]/len(str)),4)

    #453-FASG760102 
    feature[line_number][453]=(str.count('A')*297)+(str.count('L')*337)+(str.count('R')*238)+(str.count('K')*224)+(str.count('N')*236)
    feature[line_number][453]=feature[line_number][453]+(str.count('M')*283)+(str.count('D')*270)+(str.count('C')*178)+(str.count('F')*284)+(str.count('P')*222)
    feature[line_number][453]=feature[line_number][453]+(str.count('Q')*185)+(str.count('S')*228)+(str.count('E')*249)+(str.count('T')*253)+(str.count('G')*290)
    feature[line_number][453]=feature[line_number][453]+(str.count('W')*282)+(str.count('H')*277)+(str.count('Y')*344)+(str.count('I')*284)+(str.count('V')*293)
    feature[line_number][453]=round((feature[line_number][453]/len(str)),4)

    #454-JANJ780102 
    feature[line_number][454]=(str.count('A')*51)+(str.count('L')*60)+(str.count('R')*5)+(str.count('K')*3)+(str.count('N')*22)
    feature[line_number][454]=feature[line_number][454]+(str.count('M')*52)+(str.count('D')*19)+(str.count('C')*74)+(str.count('F')*58)+(str.count('P')*25)
    feature[line_number][454]=feature[line_number][454]+(str.count('Q')*16)+(str.count('S')*35)+(str.count('E')*16)+(str.count('T')*30)+(str.count('G')*52)
    feature[line_number][454]=feature[line_number][454]+(str.count('W')*49)+(str.count('H')*34)+(str.count('Y')*24)+(str.count('I')*66)+(str.count('V')*64)
    feature[line_number][454]=round((feature[line_number][454]/len(str)),4)

    #455-JANJ780103 
    feature[line_number][455]=(str.count('A')*15)+(str.count('L')*16)+(str.count('R')*67)+(str.count('K')*85)+(str.count('N')*49)
    feature[line_number][455]=feature[line_number][455]+(str.count('M')*20)+(str.count('D')*50)+(str.count('C')*5)+(str.count('F')*10)+(str.count('P')*45)
    feature[line_number][455]=feature[line_number][455]+(str.count('Q')*56)+(str.count('S')*32)+(str.count('E')*55)+(str.count('T')*32)+(str.count('G')*10)
    feature[line_number][455]=feature[line_number][455]+(str.count('W')*17)+(str.count('H')*34)+(str.count('Y')*41)+(str.count('I')*13)+(str.count('V')*14)
    feature[line_number][455]=round((feature[line_number][455]/len(str)),4)

    #456-d1
    feature[line_number][456]=(str.count('A')*2)+(str.count('L')*5)+(str.count('R')*8)+(str.count('K')*6)+(str.count('N')*5)
    feature[line_number][456]=feature[line_number][456]+(str.count('M')*5)+(str.count('D')*5)+(str.count('C')*3)+(str.count('F')*8)+(str.count('P')*4)
    feature[line_number][456]=feature[line_number][456]+(str.count('Q')*6)+(str.count('S')*3)+(str.count('E')*6)+(str.count('T')*4)+(str.count('G')*1)
    feature[line_number][456]=feature[line_number][456]+(str.count('W')*11)+(str.count('H')*7)+(str.count('Y')*9)+(str.count('I')*5)+(str.count('V')*4)
    feature[line_number][456]=round((feature[line_number][456]/len(str)),4)

    #457-d2
    feature[line_number][457]=(str.count('A')*1)+(str.count('L')*4)+(str.count('R')*7)+(str.count('K')*5)+(str.count('N')*4)
    feature[line_number][457]=feature[line_number][457]+(str.count('M')*4)+(str.count('D')*4)+(str.count('C')*2)+(str.count('F')*8)+(str.count('P')*4)
    feature[line_number][457]=feature[line_number][457]+(str.count('Q')*5)+(str.count('S')*2)+(str.count('E')*5)+(str.count('T')*3)+(str.count('G')*0)
    feature[line_number][457]=feature[line_number][457]+(str.count('W')*12)+(str.count('H')*6)+(str.count('Y')*9)+(str.count('I')*4)+(str.count('V')*3)
    feature[line_number][457]=round((feature[line_number][457]/len(str)),4)

    #458-d3
    feature[line_number][458]=(str.count('A')*2)+(str.count('L')*8)+(str.count('R')*12)+(str.count('K')*10)+(str.count('N')*8)
    feature[line_number][458]=feature[line_number][458]+(str.count('M')*8)+(str.count('D')*8)+(str.count('C')*4)+(str.count('F')*14)+(str.count('P')*8)
    feature[line_number][458]=feature[line_number][458]+(str.count('Q')*10)+(str.count('S')*4)+(str.count('E')*10)+(str.count('T')*6)+(str.count('G')*0)
    feature[line_number][458]=feature[line_number][458]+(str.count('W')*24)+(str.count('H')*14)+(str.count('Y')*18)+(str.count('I')*8)+(str.count('V')*6)
    feature[line_number][458]=round((feature[line_number][458]/len(str)),4)

    #459-d4
    feature[line_number][459]=(str.count('A')*1)+(str.count('L')*4)+(str.count('R')*6)+(str.count('K')*4)+(str.count('N')*4)
    feature[line_number][459]=feature[line_number][459]+(str.count('M')*4)+(str.count('D')*4)+(str.count('C')*2)+(str.count('F')*6)+(str.count('P')*4)
    feature[line_number][459]=feature[line_number][459]+(str.count('Q')*4)+(str.count('S')*2)+(str.count('E')*5)+(str.count('T')*3)+(str.count('G')*1)
    feature[line_number][459]=feature[line_number][459]+(str.count('W')*8)+(str.count('H')*6)+(str.count('Y')*7)+(str.count('I')*4)+(str.count('V')*3)
    feature[line_number][459]=round((feature[line_number][459]/len(str)),4)

    #460-d5
    feature[line_number][460]=(str.count('A')*1)+(str.count('L')*5)+(str.count('R')*8.12)+(str.count('K')*7)+(str.count('N')*5)
    feature[line_number][460]=feature[line_number][460]+(str.count('M')*5.4)+(str.count('D')*5.17)+(str.count('C')*2.33)+(str.count('F')*7)+(str.count('P')*4)
    feature[line_number][460]=feature[line_number][460]+(str.count('Q')*5.86)+(str.count('S')*1.67)+(str.count('E')*6)+(str.count('T')*3.25)+(str.count('G')*0)
    feature[line_number][460]=feature[line_number][460]+(str.count('W')*11.1)+(str.count('H')*6.71)+(str.count('Y')*8.88)+(str.count('I')*3.25)+(str.count('V')*3.25)
    feature[line_number][460]=round((feature[line_number][460]/len(str)),4)

    #461-d6
    feature[line_number][461]=(str.count('A')*1)+(str.count('L')*3)+(str.count('R')*6)+(str.count('K')*5)+(str.count('N')*3)
    feature[line_number][461]=feature[line_number][461]+(str.count('M')*3)+(str.count('D')*3)+(str.count('C')*1)+(str.count('F')*6)+(str.count('P')*4)
    feature[line_number][461]=feature[line_number][461]+(str.count('Q')*4)+(str.count('S')*2)+(str.count('E')*4)+(str.count('T')*1)+(str.count('G')*0)
    feature[line_number][461]=feature[line_number][461]+(str.count('W')*9)+(str.count('H')*6)+(str.count('Y')*6)+(str.count('I')*3)+(str.count('V')*1)
    feature[line_number][461]=round((feature[line_number][461]/len(str)),4)

    #462-d7
    feature[line_number][462]=(str.count('A')*1)+(str.count('L')*6)+(str.count('R')*12)+(str.count('K')*9)+(str.count('N')*6)
    feature[line_number][462]=feature[line_number][462]+(str.count('M')*7)+(str.count('D')*6)+(str.count('C')*3)+(str.count('F')*11)+(str.count('P')*4)
    feature[line_number][462]=feature[line_number][462]+(str.count('Q')*8)+(str.count('S')*3)+(str.count('E')*8)+(str.count('T')*4)+(str.count('G')*0)
    feature[line_number][462]=feature[line_number][462]+(str.count('W')*14)+(str.count('H')*9)+(str.count('Y')*13)+(str.count('I')*6)+(str.count('V')*4)
    feature[line_number][462]=round((feature[line_number][462]/len(str)),4)

    #463-d8
    feature[line_number][463]=(str.count('A')*1)+(str.count('L')*1.6)+(str.count('R')*1.5)+(str.count('K')*1.667)+(str.count('N')*1.6)
    feature[line_number][463]=feature[line_number][463]+(str.count('M')*1.6)+(str.count('D')*1.6)+(str.count('C')*1.333)+(str.count('F')*1.75)+(str.count('P')*2)
    feature[line_number][463]=feature[line_number][463]+(str.count('Q')*1.667)+(str.count('S')*1.333)+(str.count('E')*1.667)+(str.count('T')*1.5)+(str.count('G')*0)
    feature[line_number][463]=feature[line_number][463]+(str.count('W')*2.182)+(str.count('H')*2)+(str.count('Y')*2)+(str.count('I')*1.6)+(str.count('V')*1.5)
    feature[line_number][463]=round((feature[line_number][463]/len(str)),4)

    #464-d9
    feature[line_number][464]=(str.count('A')*2)+(str.count('L')*11.029)+(str.count('R')*12.499)+(str.count('K')*10.363)+(str.count('N')*11.539)
    feature[line_number][464]=feature[line_number][464]+(str.count('M')*9.49)+(str.count('D')*11.539)+(str.count('C')*6.243)+(str.count('F')*14.851)+(str.count('P')*12)
    feature[line_number][464]=feature[line_number][464]+(str.count('Q')*12.207)+(str.count('S')*5)+(str.count('E')*11.53)+(str.count('T')*9.928)+(str.count('G')*0)
    feature[line_number][464]=feature[line_number][464]+(str.count('W')*13.511)+(str.count('H')*12.448)+(str.count('Y')*12.868)+(str.count('I')*10.851)+(str.count('V')*9.928)
    feature[line_number][464]=round((feature[line_number][464]/len(str)),4)

    #465-d10
    feature[line_number][465]=(str.count('A')*0)+(str.count('L')*4.729)+(str.count('R')*-4.307)+(str.count('K')*-3.151)+(str.count('N')*-4.178)
    feature[line_number][465]=feature[line_number][465]+(str.count('M')*-2.812)+(str.count('D')*-4.178)+(str.count('C')*-2.243)+(str.count('F')*-4.801)+(str.count('P')*-4)
    feature[line_number][465]=feature[line_number][465]+(str.count('Q')*-4.255)+(str.count('S')*1)+(str.count('E')*-3.425)+(str.count('T')*-3.928)+(str.count('G')*0)
    feature[line_number][465]=feature[line_number][465]+(str.count('W')*-6.324)+(str.count('H')*-3.721)+(str.count('Y')*-4.793)+(str.count('I')*-6.085)+(str.count('V')*-3.928)
    feature[line_number][465]=round((feature[line_number][465]/len(str)),4)

    #466-d11
    feature[line_number][466]=(str.count('A')*1)+(str.count('L')*3.2)+(str.count('R')*3.5)+(str.count('K')*3)+(str.count('N')*3.2)
    feature[line_number][466]=feature[line_number][466]+(str.count('M')*2.8)+(str.count('D')*3.2)+(str.count('C')*2)+(str.count('F')*4.25)+(str.count('P')*4)
    feature[line_number][466]=feature[line_number][466]+(str.count('Q')*3.333)+(str.count('S')*2)+(str.count('E')*3.333)+(str.count('T')*3)+(str.count('G')*0)
    feature[line_number][466]=feature[line_number][466]+(str.count('W')*4)+(str.count('H')*4.286)+(str.count('Y')*4.333)+(str.count('I')*1.8)+(str.count('V')*3)
    feature[line_number][466]=round((feature[line_number][466]/len(str)),4)

    #467-d12
    feature[line_number][467]=(str.count('A')*2)+(str.count('L')*1.052)+(str.count('R')*-2.59)+(str.count('K')*-0.536)+(str.count('N')*0.528)
    feature[line_number][467]=feature[line_number][467]+(str.count('M')*0.678)+(str.count('D')*0.528)+(str.count('C')*2)+(str.count('F')*-1.672)+(str.count('P')*4)
    feature[line_number][467]=feature[line_number][467]+(str.count('Q')*-1.043)+(str.count('S')*2)+(str.count('E')*-0.538)+(str.count('T')*3)+(str.count('G')*0)
    feature[line_number][467]=feature[line_number][467]+(str.count('W')*-2.576)+(str.count('H')*-1.185)+(str.count('Y')*-2.054)+(str.count('I')*-1.517)+(str.count('V')*3)
    feature[line_number][467]=round((feature[line_number][467]/len(str)),4)

    #468-d13
    feature[line_number][468]=(str.count('A')*6)+(str.count('L')*12)+(str.count('R')*19)+(str.count('K')*12)+(str.count('N')*12)
    feature[line_number][468]=feature[line_number][468]+(str.count('M')*18)+(str.count('D')*12)+(str.count('C')*6)+(str.count('F')*18)+(str.count('P')*12)
    feature[line_number][468]=feature[line_number][468]+(str.count('Q')*12)+(str.count('S')*6)+(str.count('E')*12)+(str.count('T')*6)+(str.count('G')*1)
    feature[line_number][468]=feature[line_number][468]+(str.count('W')*24)+(str.count('H')*15)+(str.count('Y')*18)+(str.count('I')*12)+(str.count('V')*6)
    feature[line_number][468]=round((feature[line_number][468]/len(str)),4)

    #469-d14
    feature[line_number][469]=(str.count('A')*6)+(str.count('L')*15.6)+(str.count('R')*31.444)+(str.count('K')*24.5)+(str.count('N')*16.5)
    feature[line_number][469]=feature[line_number][469]+(str.count('M')*27.2)+(str.count('D')*16.4)+(str.count('C')*16.67)+(str.count('F')*23.25)+(str.count('P')*12)
    feature[line_number][469]=feature[line_number][469]+(str.count('Q')*21.167)+(str.count('S')*13.33)+(str.count('E')*21)+(str.count('T')*12.4)+(str.count('G')*3.5)
    feature[line_number][469]=feature[line_number][469]+(str.count('W')*27.5)+(str.count('H')*23.1)+(str.count('Y')*27.78)+(str.count('I')*15.6)+(str.count('V')*10.5)
    feature[line_number][469]=round((feature[line_number][469]/len(str)),4)

    #470-d15
    feature[line_number][470]=(str.count('A')*6)+(str.count('L')*12)+(str.count('R')*20)+(str.count('K')*18)+(str.count('N')*14)
    feature[line_number][470]=feature[line_number][470]+(str.count('M')*18)+(str.count('D')*12)+(str.count('C')*12)+(str.count('F')*18)+(str.count('P')*12)
    feature[line_number][470]=feature[line_number][470]+(str.count('Q')*15)+(str.count('S')*8)+(str.count('E')*14)+(str.count('T')*8)+(str.count('G')*1)
    feature[line_number][470]=feature[line_number][470]+(str.count('W')*18)+(str.count('H')*18)+(str.count('Y')*20)+(str.count('I')*12)+(str.count('V')*6)
    feature[line_number][470]=round((feature[line_number][470]/len(str)),4)

    #471-d16
    feature[line_number][471]=(str.count('A')*6)+(str.count('L')*18)+(str.count('R')*38)+(str.count('K')*31)+(str.count('N')*20)
    feature[line_number][471]=feature[line_number][471]+(str.count('M')*34)+(str.count('D')*20)+(str.count('C')*22)+(str.count('F')*24)+(str.count('P')*12)
    feature[line_number][471]=feature[line_number][471]+(str.count('Q')*24)+(str.count('S')*20)+(str.count('E')*26)+(str.count('T')*14)+(str.count('G')*6)
    feature[line_number][471]=feature[line_number][471]+(str.count('W')*36)+(str.count('H')*31)+(str.count('Y')*38)+(str.count('I')*18)+(str.count('V')*12)
    feature[line_number][471]=round((feature[line_number][471]/len(str)),4)

    #472-d17
    feature[line_number][472]=(str.count('A')*12)+(str.count('L')*30)+(str.count('R')*45)+(str.count('K')*37)+(str.count('N')*33.007)
    feature[line_number][472]=feature[line_number][472]+(str.count('M')*40)+(str.count('D')*34)+(str.count('C')*28)+(str.count('F')*48)+(str.count('P')*24)
    feature[line_number][472]=feature[line_number][472]+(str.count('Q')*39)+(str.count('S')*22)+(str.count('E')*40)+(str.count('T')*27)+(str.count('G')*7)
    feature[line_number][472]=feature[line_number][472]+(str.count('W')*68)+(str.count('H')*47)+(str.count('Y')*56)+(str.count('I')*30)+(str.count('V')*24.007)
    feature[line_number][472]=round((feature[line_number][472]/len(str)),4)

    #473-d18
    feature[line_number][473]=(str.count('A')*6)+(str.count('L')*6)+(str.count('R')*5)+(str.count('K')*6.17)+(str.count('N')*6.6)
    feature[line_number][473]=feature[line_number][473]+(str.count('M')*8)+(str.count('D')*6.8)+(str.count('C')*9.33)+(str.count('F')*6)+(str.count('P')*6)
    feature[line_number][473]=feature[line_number][473]+(str.count('Q')*6.5)+(str.count('S')*7.33)+(str.count('E')*6.67)+(str.count('T')*5.4)+(str.count('G')*3.5)
    feature[line_number][473]=feature[line_number][473]+(str.count('W')*5.667)+(str.count('H')*4.7)+(str.count('Y')*6.22)+(str.count('I')*6)+(str.count('V')*6)
    feature[line_number][473]=round((feature[line_number][473]/len(str)),4)

    #474-d19
    feature[line_number][474]=(str.count('A')*12)+(str.count('L')*25.021)+(str.count('R')*23.343)+(str.count('K')*22.739)+(str.count('N')*27.708)
    feature[line_number][474]=feature[line_number][474]+(str.count('M')*31.344)+(str.count('D')*28.634)+(str.count('C')*28)+(str.count('F')*26.993)+(str.count('P')*24)
    feature[line_number][474]=feature[line_number][474]+(str.count('Q')*27.831)+(str.count('S')*20)+(str.count('E')*28.731)+(str.count('T')*23.819)+(str.count('G')*7)
    feature[line_number][474]=feature[line_number][474]+(str.count('W')*29.778)+(str.count('H')*24.243)+(str.count('Y')*28.252)+(str.count('I')*24.841)+(str.count('V')*24)
    feature[line_number][474]=round((feature[line_number][474]/len(str)),4)

    #475-d20
    feature[line_number][475]=(str.count('A')*0)+(str.count('L')*0)+(str.count('R')*0)+(str.count('K')*-0.179)+(str.count('N')*0)
    feature[line_number][475]=feature[line_number][475]+(str.count('M')*0)+(str.count('D')*0)+(str.count('C')*0)+(str.count('F')*0)+(str.count('P')*0)
    feature[line_number][475]=feature[line_number][475]+(str.count('Q')*0)+(str.count('S')*0)+(str.count('E')*0)+(str.count('T')*-4.227)+(str.count('G')*0)
    feature[line_number][475]=feature[line_number][475]+(str.count('W')*0.211)+(str.count('H')*-1.734)+(str.count('Y')*-0.96)+(str.count('I')*-1.641)+(str.count('V')*0)
    feature[line_number][475]=round((feature[line_number][475]/len(str)),4)

    #476-d21
    feature[line_number][476]=(str.count('A')*6)+(str.count('L')*9.6)+(str.count('R')*10.667)+(str.count('K')*10.167)+(str.count('N')*10)
    feature[line_number][476]=feature[line_number][476]+(str.count('M')*13.6)+(str.count('D')*10.4)+(str.count('C')*11.333)+(str.count('F')*12)+(str.count('P')*12)
    feature[line_number][476]=feature[line_number][476]+(str.count('Q')*10.5)+(str.count('S')*8.667)+(str.count('E')*10.667)+(str.count('T')*9)+(str.count('G')*3.5)
    feature[line_number][476]=feature[line_number][476]+(str.count('W')*12.75)+(str.count('H')*10.4)+(str.count('Y')*12.222)+(str.count('I')*9.6)+(str.count('V')*9)
    feature[line_number][476]=round((feature[line_number][476]/len(str)),4)

    #477-d22
    feature[line_number][477]=(str.count('A')*0)+(str.count('L')*3.113)+(str.count('R')*4.2)+(str.count('K')*1.372)+(str.count('N')*3)
    feature[line_number][477]=feature[line_number][477]+(str.count('M')*2.656)+(str.count('D')*2.969)+(str.count('C')*6)+(str.count('F')*2.026)+(str.count('P')*12)
    feature[line_number][477]=feature[line_number][477]+(str.count('Q')*1.849)+(str.count('S')*6)+(str.count('E')*1.822)+(str.count('T')*6)+(str.count('G')*0)
    feature[line_number][477]=feature[line_number][477]+(str.count('W')*2.044)+(str.count('H')*1.605)+(str.count('Y')*1.599)+(str.count('I')*3.373)+(str.count('V')*6)
    feature[line_number][477]=round((feature[line_number][477]/len(str)),4)

    #478  ARGP820102 
    feature[line_number][478]=(str.count('A')*1.18)+(str.count('L')*3.23)+(str.count('R')*0.2)+(str.count('K')*0.06)+(str.count('N')*0.23)
    feature[line_number][478]=feature[line_number][478]+(str.count('M')*2.67)+(str.count('D')*0.05)+(str.count('C')*1.89)+(str.count('F')*1.96)+(str.count('P')*0.76)
    feature[line_number][478]=feature[line_number][478]+(str.count('Q')*0.72)+(str.count('S')*0.97)+(str.count('E')*0.11)+(str.count('T')*0.84)+(str.count('G')*0.49)
    feature[line_number][478]=feature[line_number][478]+(str.count('W')*0.77)+(str.count('H')*0.31)+(str.count('Y')*0.39)+(str.count('I')*1.45)+(str.count('V')*1.08)
    feature[line_number][478]=round((feature[line_number][478]/len(str)),4)
            
    #479 ARGP820103
    feature[line_number][479]=(str.count('A')*1.56)+(str.count('L')*2.93)+(str.count('R')*0.45)+(str.count('K')*0.15)+(str.count('N')*0.27)
    feature[line_number][479]=feature[line_number][479]+(str.count('M')*2.96)+(str.count('D')*0.14)+(str.count('C')*1.23)+(str.count('F')*2.03)+(str.count('P')*0.76)
    feature[line_number][479]=feature[line_number][479]+(str.count('Q')*0.51)+(str.count('S')*0.81)+(str.count('E')*0.23)+(str.count('T')*0.91)+(str.count('G')*0.62)
    feature[line_number][479]=feature[line_number][479]+(str.count('W')*1.08)+(str.count('H')*0.29)+(str.count('Y')*0.68)+(str.count('I')*1.67)+(str.count('V')*1.14)
    feature[line_number][479]=round((feature[line_number][479]/len(str)),4)

    #480  BHAR452101 
    feature[line_number][480]=(str.count('A')*0.357)+(str.count('L')*0.365)+(str.count('R')*0.529)+(str.count('K')*0.466)+(str.count('N')*0.463)
    feature[line_number][480]=feature[line_number][480]+(str.count('M')*0.295)+(str.count('D')*0.511)+(str.count('C')*0.346)+(str.count('F')*0.314)+(str.count('P')*0.509)
    feature[line_number][480]=feature[line_number][480]+(str.count('Q')*0.493)+(str.count('S')*0.507)+(str.count('E')*0.497)+(str.count('T')*0.444)+(str.count('G')*0.544)
    feature[line_number][480]=feature[line_number][480]+(str.count('W')*0.305)+(str.count('H')*0.323)+(str.count('Y')*0.42)+(str.count('I')*0.462)+(str.count('V')*0.386)
    feature[line_number][480]=round((feature[line_number][480]/len(str)),4)

    #481 CHAM820101 
    feature[line_number][481]=(str.count('A')*0.046)+(str.count('L')*0.186)+(str.count('R')*0.291)+(str.count('K')*0.219)+(str.count('N')*0.134)
    feature[line_number][481]=feature[line_number][481]+(str.count('M')*0.221)+(str.count('D')*0.105)+(str.count('C')*0.128)+(str.count('F')*0.29)+(str.count('P')*0.131)
    feature[line_number][481]=feature[line_number][481]+(str.count('Q')*0.18)+(str.count('S')*0.062)+(str.count('E')*0.151)+(str.count('T')*0.108)+(str.count('G')*0)
    feature[line_number][481]=feature[line_number][481]+(str.count('W')*0.409)+(str.count('H')*0.23)+(str.count('Y')*0.298)+(str.count('I')*0.186)+(str.count('V')*0.14)
    feature[line_number][481]=round((feature[line_number][481]/len(str)),4)

    #482 CHAM820102 
    feature[line_number][482]=(str.count('A')*-0.368)+(str.count('L')*1.07)+(str.count('R')*-1.03)+(str.count('K')*0)+(str.count('N')*0)
    feature[line_number][482]=feature[line_number][482]+(str.count('M')*0.656)+(str.count('D')*2.06)+(str.count('C')*4.53)+(str.count('F')*1.06)+(str.count('P')*-2.24)
    feature[line_number][482]=feature[line_number][482]+(str.count('Q')*0.731)+(str.count('S')*-0.524)+(str.count('E')*1.77)+(str.count('T')*0)+(str.count('G')*-0.525)
    feature[line_number][482]=feature[line_number][482]+(str.count('W')*1.6)+(str.count('H')*0)+(str.count('Y')*4.91)+(str.count('I')*0.791)+(str.count('V')*0.401)
    feature[line_number][482]=round((feature[line_number][482]/len(str)),4)

    #483 DAYM780201 
    feature[line_number][483]=(str.count('A')*100)+(str.count('L')*40)+(str.count('R')*65)+(str.count('K')*56)+(str.count('N')*134)
    feature[line_number][483]=feature[line_number][483]+(str.count('M')*94)+(str.count('D')*106)+(str.count('C')*20)+(str.count('F')*41)+(str.count('P')*56)
    feature[line_number][483]=feature[line_number][483]+(str.count('Q')*93)+(str.count('S')*120)+(str.count('E')*102)+(str.count('T')*97)+(str.count('G')*49)
    feature[line_number][483]=feature[line_number][483]+(str.count('W')*18)+(str.count('H')*66)+(str.count('Y')*41)+(str.count('I')*96)+(str.count('V')*74)
    feature[line_number][483]=round((feature[line_number][483]/len(str)),4)

    #484 EISD860102
    feature[line_number][484]=(str.count('A')*0)+(str.count('L')*1)+(str.count('R')*10)+(str.count('K')*5.7)+(str.count('N')*1.3)
    feature[line_number][484]=feature[line_number][484]+(str.count('M')*1.9)+(str.count('D')*1.9)+(str.count('C')*0.17)+(str.count('F')*1.1)+(str.count('P')*0.18)
    feature[line_number][484]=feature[line_number][484]+(str.count('Q')*1.9)+(str.count('S')*0.73)+(str.count('E')*3)+(str.count('T')*1.5)+(str.count('G')*0)
    feature[line_number][484]=feature[line_number][484]+(str.count('W')*1.6)+(str.count('H')*0.99)+(str.count('Y')*1.8)+(str.count('I')*1.2)+(str.count('V')*0.48)
    feature[line_number][484]=round((feature[line_number][484]/len(str)),4)

    #485 FAUJ452103
    feature[line_number][485]=(str.count('A')*1)+(str.count('L')*4)+(str.count('R')*6.13)+(str.count('K')*4.77)+(str.count('N')*2.95)
    feature[line_number][485]=feature[line_number][485]+(str.count('M')*4.43)+(str.count('D')*2.78)+(str.count('C')*2.43)+(str.count('F')*5.89)+(str.count('P')*2.72)
    feature[line_number][485]=feature[line_number][485]+(str.count('Q')*3.95)+(str.count('S')*1.6)+(str.count('E')*3.78)+(str.count('T')*2.6)+(str.count('G')*0)
    feature[line_number][485]=feature[line_number][485]+(str.count('W')*8.08)+(str.count('H')*4.66)+(str.count('Y')*6.47)+(str.count('I')*4)+(str.count('V')*3)
    feature[line_number][485]=round((feature[line_number][485]/len(str)),4)

    #486 FAUJ452108
    feature[line_number][486]=(str.count('A')*-0.01)+(str.count('L')*-0.01)+(str.count('R')*0.04)+(str.count('K')*0)+(str.count('N')*0.06)
    feature[line_number][486]=feature[line_number][486]+(str.count('M')*0.04)+(str.count('D')*0.15)+(str.count('C')*0.12)+(str.count('F')*0.03)+(str.count('P')*0)
    feature[line_number][486]=feature[line_number][486]+(str.count('Q')*0.05)+(str.count('S')*0.11)+(str.count('E')*0.07)+(str.count('T')*0.04)+(str.count('G')*0)
    feature[line_number][486]=feature[line_number][486]+(str.count('W')*0)+(str.count('H')*0.08)+(str.count('Y')*0.03)+(str.count('I')*-0.01)+(str.count('V')*0.01)
    feature[line_number][486]=round((feature[line_number][486]/len(str)),4)

    #487 GARJ730101 
    feature[line_number][487]=(str.count('A')*0.28)+(str.count('L')*1)+(str.count('R')*0.1)+(str.count('K')*0.09)+(str.count('N')*0.25)
    feature[line_number][487]=feature[line_number][487]+(str.count('M')*0.74)+(str.count('D')*0.21)+(str.count('C')*0.28)+(str.count('F')*2.18)+(str.count('P')*0.39)
    feature[line_number][487]=feature[line_number][487]+(str.count('Q')*0.35)+(str.count('S')*0.12)+(str.count('E')*0.33)+(str.count('T')*0.21)+(str.count('G')*0.17)
    feature[line_number][487]=feature[line_number][487]+(str.count('W')*5.7)+(str.count('H')*0.21)+(str.count('Y')*1.26)+(str.count('I')*0.82)+(str.count('V')*0.6)
    feature[line_number][487]=round((feature[line_number][487]/len(str)),4)

    #488  HOPA770101 
    feature[line_number][488]=(str.count('A')*1)+(str.count('L')*0.8)+(str.count('R')*2.3)+(str.count('K')*5.3)+(str.count('N')*2.2)
    feature[line_number][488]=feature[line_number][488]+(str.count('M')*0.7)+(str.count('D')*6.5)+(str.count('C')*0.1)+(str.count('F')*1.4)+(str.count('P')*0.9)
    feature[line_number][488]=feature[line_number][488]+(str.count('Q')*2.1)+(str.count('S')*1.7)+(str.count('E')*6.2)+(str.count('T')*1.5)+(str.count('G')*1.1)
    feature[line_number][488]=feature[line_number][488]+(str.count('W')*1.9)+(str.count('H')*2.8)+(str.count('Y')*2.1)+(str.count('I')*0.8)+(str.count('V')*0.9)
    feature[line_number][488]=round((feature[line_number][488]/len(str)),4)

    #489 HUTJ700101
    feature[line_number][489]=(str.count('A')*29.22)+(str.count('L')*48.03)+(str.count('R')*26.37)+(str.count('K')*57.1)+(str.count('N')*38.3)
    feature[line_number][489]=feature[line_number][489]+(str.count('M')*69.32)+(str.count('D')*37.09)+(str.count('C')*50.7)+(str.count('F')*48.52)+(str.count('P')*36.13)
    feature[line_number][489]=feature[line_number][489]+(str.count('Q')*44.02)+(str.count('S')*32.4)+(str.count('E')*41.84)+(str.count('T')*35.2)+(str.count('G')*23.71)
    feature[line_number][489]=feature[line_number][489]+(str.count('W')*56.92)+(str.count('H')*59.64)+(str.count('Y')*51.73)+(str.count('I')*45)+(str.count('V')*40.35)
    feature[line_number][489]=round((feature[line_number][489]/len(str)),4)

    #490 HUTJ700102
    feature[line_number][490]=(str.count('A')*30.88)+(str.count('L')*50.62)+(str.count('R')*68.43)+(str.count('K')*63.21)+(str.count('N')*41.7)
    feature[line_number][490]=feature[line_number][490]+(str.count('M')*55.32)+(str.count('D')*40.66)+(str.count('C')*53.83)+(str.count('F')*51.06)+(str.count('P')*39.21)
    feature[line_number][490]=feature[line_number][490]+(str.count('Q')*46.62)+(str.count('S')*35.65)+(str.count('E')*44.98)+(str.count('T')*36.5)+(str.count('G')*24.74)
    feature[line_number][490]=feature[line_number][490]+(str.count('W')*60)+(str.count('H')*65.99)+(str.count('Y')*51.15)+(str.count('I')*49.71)+(str.count('V')*42.75)
    feature[line_number][490]=round((feature[line_number][490]/len(str)),4)

    #491 HUTJ700103
    feature[line_number][491]=(str.count('A')*154.33)+(str.count('L')*232.3)+(str.count('R')*341.01)+(str.count('K')*300.46)+(str.count('N')*207.9)
    feature[line_number][491]=feature[line_number][491]+(str.count('M')*202.65)+(str.count('D')*194.91)+(str.count('C')*219.79)+(str.count('F')*204.74)+(str.count('P')*179.93)
    feature[line_number][491]=feature[line_number][491]+(str.count('Q')*235.51)+(str.count('S')*174.06)+(str.count('E')*223.16)+(str.count('T')*205.8)+(str.count('G')*127.9)
    feature[line_number][491]=feature[line_number][491]+(str.count('W')*237.01)+(str.count('H')*242.54)+(str.count('Y')*229.15)+(str.count('I')*233.21)+(str.count('V')*207.6)
    feature[line_number][491]=round((feature[line_number][491]/len(str)),4)

    #492  MCMT640101 
    feature[line_number][492]=(str.count('A')*4.34)+(str.count('L')*18.78)+(str.count('R')*26.66)+(str.count('K')*21.29)+(str.count('N')*13.28)
    feature[line_number][492]=feature[line_number][492]+(str.count('M')*21.64)+(str.count('D')*12)+(str.count('C')*35.77)+(str.count('F')*29.4)+(str.count('P')*10.93)
    feature[line_number][492]=feature[line_number][492]+(str.count('Q')*17.56)+(str.count('S')*6.35)+(str.count('E')*17.26)+(str.count('T')*11.01)+(str.count('G')*0)
    feature[line_number][492]=feature[line_number][492]+(str.count('W')*42.53)+(str.count('H')*21.81)+(str.count('Y')*31.53)+(str.count('I')*19.06)+(str.count('V')*13.92)
    feature[line_number][492]=round((feature[line_number][492]/len(str)),4)

    #493 MEEJ800101
    feature[line_number][493]=(str.count('A')*0.5)+(str.count('L')*8.8)+(str.count('R')*0.8)+(str.count('K')*0.1)+(str.count('N')*0.8)
    feature[line_number][493]=feature[line_number][493]+(str.count('M')*4.8)+(str.count('D')*-8.2)+(str.count('C')*-6.8)+(str.count('F')*13.2)+(str.count('P')*6.1)
    feature[line_number][493]=feature[line_number][493]+(str.count('Q')*-4.8)+(str.count('S')*1.2)+(str.count('E')*-16.9)+(str.count('T')*2.7)+(str.count('G')*0)
    feature[line_number][493]=feature[line_number][493]+(str.count('W')*14.9)+(str.count('H')*-3.5)+(str.count('Y')*6.1)+(str.count('I')*13.9)+(str.count('V')*2.7)
    feature[line_number][493]=round((feature[line_number][493]/len(str)),4)

    #494 MEEJ800102
    feature[line_number][494]=(str.count('A')*-0.1)+(str.count('L')*10)+(str.count('R')*-4.5)+(str.count('K')*-3.2)+(str.count('N')*-1.6)
    feature[line_number][494]=feature[line_number][494]+(str.count('M')*7.1)+(str.count('D')*-2.8)+(str.count('C')*-2.2)+(str.count('F')*13.9)+(str.count('P')*8)
    feature[line_number][494]=feature[line_number][494]+(str.count('Q')*-2.5)+(str.count('S')*-3.7)+(str.count('E')*-7.5)+(str.count('T')*1.5)+(str.count('G')*-0.5)
    feature[line_number][494]=feature[line_number][494]+(str.count('W')*18.1)+(str.count('H')*0.8)+(str.count('Y')*8.2)+(str.count('I')*11.8)+(str.count('V')*3.3)
    feature[line_number][494]=round((feature[line_number][494]/len(str)),4)

    #495 SNEP660101
    feature[line_number][495]=(str.count('A')*0.239)+(str.count('L')*0.281)+(str.count('R')*0.211)+(str.count('K')*0.228)+(str.count('N')*0.249)
    feature[line_number][495]=feature[line_number][495]+(str.count('M')*0.253)+(str.count('D')*0.171)+(str.count('C')*0.22)+(str.count('F')*0.234)+(str.count('P')*0.165)
    feature[line_number][495]=feature[line_number][495]+(str.count('Q')*0.26)+(str.count('S')*0.236)+(str.count('E')*0.187)+(str.count('T')*0.213)+(str.count('G')*0.16)
    feature[line_number][495]=feature[line_number][495]+(str.count('W')*0.183)+(str.count('H')*0.205)+(str.count('Y')*0.193)+(str.count('I')*0.273)+(str.count('V')*0.255)
    feature[line_number][495]=round((feature[line_number][495]/len(str)),4)

    #496 SNEP660102
    feature[line_number][496]=(str.count('A')*0.33)+(str.count('L')*0.129)+(str.count('R')*-0.176)+(str.count('K')*-0.075)+(str.count('N')*-0.233)
    feature[line_number][496]=feature[line_number][496]+(str.count('M')*-0.092)+(str.count('D')*-0.371)+(str.count('C')*0.074)+(str.count('F')*-0.011)+(str.count('P')*0.37)
    feature[line_number][496]=feature[line_number][496]+(str.count('Q')*-0.254)+(str.count('S')*0.022)+(str.count('E')*-0.409)+(str.count('T')*0.136)+(str.count('G')*0.37)
    feature[line_number][496]=feature[line_number][496]+(str.count('W')*-0.011)+(str.count('H')*-0.078)+(str.count('Y')*-0.138)+(str.count('I')*0.149)+(str.count('V')*0.245)
    feature[line_number][496]=round((feature[line_number][496]/len(str)),4)

    #497 SNEP660103
    feature[line_number][497]=(str.count('A')*-0.11)+(str.count('L')*-0.008)+(str.count('R')*0.079)+(str.count('K')*0.049)+(str.count('N')*-0.136)
    feature[line_number][497]=feature[line_number][497]+(str.count('M')*-0.041)+(str.count('D')*-0.285)+(str.count('C')*-0.184)+(str.count('F')*0.438)+(str.count('P')*-0.016)
    feature[line_number][497]=feature[line_number][497]+(str.count('Q')*-0.067)+(str.count('S')*-0.153)+(str.count('E')*-0.246)+(str.count('T')*-0.208)+(str.count('G')*-0.073)
    feature[line_number][497]=feature[line_number][497]+(str.count('W')*0.493)+(str.count('H')*0.32)+(str.count('Y')*0.381)+(str.count('I')*0.001)+(str.count('V')*-0.155)
    feature[line_number][497]=round((feature[line_number][497]/len(str)),4)

    #498 SNEP660104
    feature[line_number][498]=(str.count('A')*-0.062)+(str.count('L')*-0.264)+(str.count('R')*-0.167)+(str.count('K')*-0.371)+(str.count('N')*0.166)
    feature[line_number][498]=feature[line_number][498]+(str.count('M')*0.077)+(str.count('D')*-0.079)+(str.count('C')*0.38)+(str.count('F')*0.074)+(str.count('P')*-0.036)
    feature[line_number][498]=feature[line_number][498]+(str.count('Q')*-0.025)+(str.count('S')*0.47)+(str.count('E')*-0.184)+(str.count('T')*0.348)+(str.count('G')*-0.017)
    feature[line_number][498]=feature[line_number][498]+(str.count('W')*0.05)+(str.count('H')*0.056)+(str.count('Y')*0.22)+(str.count('I')*-0.309)+(str.count('V')*-0.212)
    feature[line_number][498]=round((feature[line_number][498]/len(str)),4)
            
    #PSSM
    L=len(str)
    p = [[0 for i in range(20)] for j in range(L)]
    aa=['A','R','N','D','C','E','Q','G','H','I','L','K','M','F','P','S','T','W','Y','V']

    for i in range(L):
                if str[i]=='A':
                    p[i][0]=p[i][0]+1
                if str[i]=='R':
                    p[i][1]=p[i][1]+1
                if str[i]=='N':
                    p[i][2]=p[i][2]+1
                if str[i]=='D':
                    p[i][3]=p[i][3]+1
                if str[i]=='C':
                    p[i][4]=p[i][4]+1
                if str[i]=='E':
                    p[i][5]=p[i][5]+1
                if str[i]=='Q':
                    p[i][6]=p[i][6]+1
                if str[i]=='G':
                    p[i][7]=p[i][7]+1
                if str[i]=='H':
                    p[i][8]=p[i][8]+1
                if str[i]=='I':
                    p[i][9]=p[i][9]+1
                if str[i]=='L':
                    p[i][10]=p[i][10]+1
                if str[i]=='K':
                    p[i][11]=p[i][11]+1
                if str[i]=='M':
                    p[i][12]=p[i][12]+1
                if str[i]=='F':
                    p[i][13]=p[i][13]+1
                if str[i]=='P':
                    p[i][14]=p[i][14]+1
                if str[i]=='S':
                    p[i][15]=p[i][15]+1
                if str[i]=='T':
                    p[i][16]=p[i][16]+1
                if str[i]=='W':
                    p[i][17]=p[i][17]+1
                if str[i]=='Y':
                    p[i][18]=p[i][18]+1
                if str[i]=='V':
                    p[i][19]=p[i][19]+1
                
    pssm=p
    for i in range(L):
                 for j in range(20):
                     if pssm[i][j]>0:
                         pssm[i][j]=math.log((pssm[i][j]/0.05),2)

    fpssm=pssm
    for i in range(L):
                 for j in range(20):
                     if fpssm[i][j]<0:
                         fpssm[i][j]=0
                     elif fpssm[i][j]>7:
                         fpssm[i][j]=7

    sfpssm = [[0 for i in range(20)] for j in range(20)]
    for i in range(20):
                for j in range(20):
                    total=0
                    for k in range(L-1):
                        if str[k]==aa[i]:
                            total=total+(fpssm[k][j]*1)
                        else:
                            total=total+(fpssm[k][j]*0)
                    sfpssm[i][j]=total
    #print(sfpssm)                
    dpcpssm  = [[0 for i in range(20)] for j in range(20)]
    for i in range(20):
                for j in range(20):
                    total=0
                    for k in range(L-1):
                       total=total+(p[k][i]*p[k+1][j])
                    total=total/(L-1)
                    dpcpssm[i][j]=total
    cc=1
    for i in range(20):
           for j in range(20):
                feature[line_number][498+cc]=dpcpssm[i][j]
                cc=cc+1
    for i in range(20):
           for j in range(20):
                feature[line_number][498+cc]=sfpssm[i][j]
                cc=cc+1

    #CTD
    for i in range(len(str)-2):
               p=0
               X=str[i]
               Y=str[i+1]
               Z=str[i+2]
               if X in S1:
                 p=1*100
               if X in S2:
                 p=2*100
               if X in S3:
                 p=3*100
               if X in S4:
                 p=4*100
               if X in S5:
                 p=5*100
               if X in S6:
                 p=6*100
               if X in S7:
                 p=7*100   
    
               if Y in S1:
                 p=p+1*10
               if Y in S2:
                 p=p+2*10
               if Y in S3:
                 p=p+3*10
               if Y in S4:
                 p=p+4*10
               if Y in S5:
                 p=p+5*10
               if Y in S6:
                 p=p+6*10
               if Y in S7:
                 p=p+7*10

               if Z in S1:
                 p=p+1
               if Z in S2:
                 p=p+2
               if Z in S3:
                 p=p+3
               if Z in S4:
                 p=p+4
               if Z in S5:
                 p=p+5
               if Z in S6:
                 p=p+6
               if Z in S7:
                 p=p+7
            
               for j in range(343):
                 if p==N[0,j]:
                   k=j
               feature1[0,k]=feature1[0,k]+1
    feature1[0,:]=feature1[0,:]/(len(str)-2)*100
    feature[line_number][1299:1642]=feature1[0,:]

    print('Amino acid feature extraction done!')
    #-----------------------------------------------------------------------------------
    print('Extracting features from nucleotide sequences...')
    #nucleotide feature set
    nucleotide=['A','T','G','C']
    dinucleotide=list()
    trinucleotide=list()
    for i in range(len(nucleotide)):
        for j in range(len(nucleotide)):
            t=''
            t=nucleotide[i]+nucleotide[j]
            dinucleotide.append(t)
            
    
    for i in range(len(nucleotide)):
        for j in range(len(nucleotide)):
            for k in range(len(nucleotide)):
             t=''
             t=nucleotide[i]+nucleotide[j]+nucleotide[k]
             trinucleotide.append(t)
             
            
    id=open(nucleotide_file_name,"r")
    line=id.readline()
    line=id.readline()
    str=''
    count=0
    line_number=0
    while line:
        if '>' not in line:
            str=str+line
        if '>' in line:
            #single nucleotide count
            for i in range(len(nucleotide)):
                
                feature[line_number][i+1642]=round(str.count(nucleotide[i])/len(str),4)

            #dinucleotide count    
            for i in range(len(dinucleotide)):
                
                feature[line_number][i+4+1642]=round(str.count(dinucleotide[i])/len(str),4)
            

            #trinucleotide count    
            for i in range(len(trinucleotide)):
                
                feature[line_number][i+4+16+1642]=round(str.count(trinucleotide[i])/len(str),4)
            feature[line_number][1726]=round((str.count('G')+ str.count('C'))/len(str),4)
            line_number=line_number+1
            
        line=id.readline()
    
    #single nucleotide count
    for i in range(len(nucleotide)):
             feature[line_number][i+1642]=round(str.count(nucleotide[i])/len(str),4)


    #dinucleotide count    
    for i in range(len(dinucleotide)):
             feature[line_number][i+4+1642]=round(str.count(dinucleotide[i])/len(str),4)
            

    #trinucleotide count    
    for i in range(len(trinucleotide)):
             feature[line_number][i+4+16+1642]=round(str.count(trinucleotide[i])/len(str),4)
    feature[line_number][1726]=round((str.count('G')+ str.count('C'))/len(str),4)         
    id.close()
    
    print('Nucleotide feature extraction done!')
    with open("C:/Users/Rishika/Desktop/feature_eff.csv", 'w') as myfile:
      wr = csv.writer(myfile)
      wr.writerows(feature)
    return feature

    

def fxn():
    warnings.warn("deprecated", DeprecationWarning)
  
def cqnr(X, Y):
    #mainly used for plotting the test cases of CQNR-OS
    from sklearn import svm
    import math
    import csv
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    from random import shuffle
    from sklearn.preprocessing import LabelEncoder
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation, Flatten
    from keras.layers import Convolution2D, MaxPooling2D
    from keras.utils import np_utils
    from sklearn.naive_bayes import BernoulliNB
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier
    from matplotlib import pyplot as plt
    from scipy.cluster.hierarchy import dendrogram, linkage
    import numpy as np
    from scipy.cluster.hierarchy import cophenet
    from sklearn.cluster import MeanShift
    from scipy.spatial.distance import pdist
    from sklearn.cluster import KMeans
    import pandas
    import decimal
    import random
    import time
    from sklearn.metrics import accuracy_score
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    f1=random.seed()
    from sklearn.model_selection import cross_val_score
    from imblearn.over_sampling import SMOTE
    start_time = time.time()

      
    X3=[]
    X4=[]
    file=open('C:/Users/Rishika/Desktop/before.csv','w')
    
    with file:
       writer=csv.writer(file)
       writer.writerows(X)
    
    file=open('C:/Users/Rishika/Desktop/class.csv','w')
    
    with file:
       writer=csv.writer(file)
       writer.writerows(Y)
    #more than 2 class
    print('input shape',X.shape,Y.shape)
    un=np.unique(Y)
    #print(un)
    max_class=0
    max_count=0
    for i in range(len(un)):
       count=0
       for j in range(len(Y)):
          if Y[j]==un[i]:
             count=count+1
       if count>max_count:
          max_count=count
          max_class=un[i]
    print('max class', max_class)
    X4=[]
    X3=[]
    for i in range(len(Y)):
       if Y[i]==max_class:
           X4.append(X[i])
    n=X.shape
    print(n)
    X_syn=[]
    Y_syn=[]
    
    for i in range(len(un)):
       
       for j in range(len(Y)):
          if Y[j]!= max_class and Y[j]==un[i]:
             X3.append(X[j])
          
       if len(X3)>len(X4):
         smaller=X4
         larger=X3
       else:
         smaller=X3
         larger=X4
       
       if not X3:
          continue
         
       new_sample=syntheticsample(smaller,larger,n[1])
       
       
       print('\n -------------------------')
       size=n[1]
       
       S1 = [[0 for x in range(size)] for y in range(len(smaller))]
       for i1 in range(len(smaller)):
          f=smaller[i1].tolist()
          S1[i1][:]=f[:]
          
       S2 = [[0 for x in range(size)] for y in range(len(new_sample))]
       for i1 in range(len(new_sample)):
          f=new_sample[i1].tolist()
          S2[i1][:]=f[:]
          
       #combining the datas
       for i1 in range(len(S1)):
            X_syn.append(S1[i1][:])
            Y_syn.append(i+1)
       
       for i1 in range(len(S2)):
            X_syn.append(S2[i1][:])
            Y_syn.append(i+1)
            i1=i1+1
       print('smaller:',len(smaller))
       print('larger:',len(larger))
       print('new sample:',len(new_sample))
       X3=[]
       smaller=[]
       #larger=[]
       new_sample=[]
    #print(len(X_syn))  
    size=n[1]
    S3 = [[0 for x in range(size)] for y in range(len(larger))]
    for i1 in range(len(larger)):
          f=larger[i1].tolist()
          S3[i1][:]=f[:]
    for i1 in range(len(S3)):
            X_syn.append(S3[i1][:])
            Y_syn.append(0)      
    
    S4 = [[0 for x in range(size+1)] for y in range(len(X_syn))]
    for i1 in range(len(X_syn)):
          S4[i1][0:size]=X_syn[i1]
          S4[i1][size]=Y_syn[i1]
    
    #print(S4[0], S3[0])
    #file=open('davies.csv','w')
    
    
    file=open('C:/Users/Rishika/Desktop/oversampled.csv','w')
    
    with file:
       writer=csv.writer(file)
       writer.writerows(S4)
#    print('writing complete')
    elapsed_time = time.time() - start_time
    print(elapsed_time)
    return S4


def syntheticsample(smaller, larger, dim):
    from sklearn.cluster import KMeans
    import numpy as np
    import pandas
    import math
    import random
    from scipy.spatial import distance
    f1=random.seed()
    
    min_dbi=1000
    max_c=0
    num=2
    while num < 20:
      Z=KMeans(n_clusters=num)
      X=Z.fit(smaller)
      P=X.labels_
      #print(P)
      #print(len(smaller))
      
      center = [[0 for x in range(dim)] for y in range(num)]
      for i in range(num):
       max_diam=0

      #validity index
      sum_1 = [[0 for x in range(dim)] for y in range(1)]

      #cluster center
      for i in range(num):
        count=0  
        for j in range(len(P)):
           if P[j]==i:
              v1=[]
              v1=smaller[j][:]
              for h in range(dim):
                  sum_1[0][h]=sum_1[0][h]+v1[h]
              count=count+1
        for h in range(dim):
           center[i][h]=sum_1[0][h]/count
      #print(center)
      center=X.cluster_centers_
      #print(X.cluster_centers_)
      #average distance of each point from cluster center
      sum_2 = [[0 for x in range(dim)] for y in range(1)]
      #sum_2=0     
      S = [[0 for x in range(1)] for y in range(num)]
      for i in range(num):
        count=0
        X1=0
        for j in range(len(P)):
           if P[j]==i:
              v1=[]
              v1=smaller[j][:]
              for h in range(dim):
                  sum_2[0][h]=sum_2[0][h]+(center[i][h]-v1[h])**2
                  #sum_2=sum_2+distance.euclidean(center[i],v1)
              count=count+1
              
        for h in range(dim):
               #print(type(sum_2[0][h]), sum_2[0][h])
         X1=X1+sum_2[0][h]
        X1=X1/count
        S[i]=X1**(0.5)

      #Distance between each centroid
      M = [[0 for x in range(num)] for y in range(num)]
      for i in range(num):
          
        for j in range(num):
          t=0  
          if i != j:
           for g in range(dim):
             t=t+(center[i][g]-center[j][g])**2
             #t=t+distance.euclidean(center[i],center[j])
           M[i][j]=t**0.5
          
      #finding R(i,j)
      R = [[0 for x in range(num)] for y in range(num)]     
      l1=0
      for i in range(num):
        for j in range(num):
          if i !=j:
           R[i][j]=(S[i]+S[j])/M[i][j]
                
      
      #Finding Di
      D = [[0 for x in range(1)] for y in range(num)]     
      max=0
      for i in range(num):
        for j in range(num):
          if i!=j:  
           if R[i][j]>max:
            max=R[i][j]
        D[i]=max
        max=0

      #finding db
      dbi=0  
      for i in range(num):
          dbi=dbi+D[i]
      dbi=dbi/num    
  
      print(num,dbi)
      if min_dbi>dbi:
         min_dbi=dbi
         max_c=num
         
      num=num+1
    
    print(max_c)
    #final clustering
    Z=KMeans(n_clusters=max_c)
    X=Z.fit(smaller)
    P=X.labels_
    #print(P)
    d=len(larger)-len(smaller)
    #print(d)
    i=0
    
    freq=[[0 for x in range(1)] for y in range(len(np.unique(P)))]
    num_old=[[0 for x in range(1)] for y in range(len(np.unique(P)))]
    
    while i<len(np.unique(P)):
       count=0
       for j in range(len(P)):
          if P[j]==i:
             count=count+1
       
       num_old[i]=count
       freq[i]=count/len(smaller)*100
       
       i=i+1
    new=[[0 for x in range(1)] for y in range(len(np.unique(P)))]
    new_sample=[[0 for x in range(2)] for y in range(d)]
    new_sample=[]
    i=0
    while i<len(freq):
       new[i]=math.floor((freq[i]*d)/100+0.5)
       i=i+1
    #print(new)

    #new sets
    i=0
    #print(new)
    while i<(len(np.unique(P))):
       g=0
       while g<new[i]:
         i1=random.randint(0,num_old[i]-1)
         i2=random.randint(0,num_old[i]-1)
         #print(i1,i2)
         k=-1
         for j in range(len(P)):
            if P[j]==i:
               k=k+1
               if k==i1:
                  val1=smaller[j][:]
                  v1=j
               if k==i2:
                  val2=smaller[j][:]
                  v2=j
               #print(j)
         w1=random.uniform(0,1)
         w2=1-w1
         
         f=(w1*val1)+(w2*val2)
         new_sample.append(f)
         g=g+1
       i=i+1
       

    n=np.shape(smaller)

    return new_sample
     

def sensitivity(y_true, y_pred):
    print(y_pred)
    true_positives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_positives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((y_true) * (y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

def focal_loss(y_true, y_pred):
    gamma = 2.0
    alpha = 0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))

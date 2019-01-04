# Packages for analysis
import pandas as pd
import numpy as np
import operator
import  time

import random

gdata = pd.read_csv('glass_dataset.csv')


x = gdata.drop(['Type','Id'], axis=1)

y = gdata['Type']

np.random.seed(5)
s=np.arange(214)
np.random.shuffle(s)
x=x.as_matrix()
x=np.array(x[s])
y=y.as_matrix()
y=np.array(y[s])

from sklearn.model_selection import train_test_split

x_train, x_test, y_trainlabel, y_testlabel = train_test_split(x, y, test_size = 0.20,shuffle=False)

from sklearn.svm import SVC  


from sklearn.cross_validation import cross_val_score
Cs = [2**-5,2**-4,2**-3,2**-2,2**-1,2**0,2**1,2**2,2**3,2**4,2**5,2**6,2**7,2**8,2**9,2**10]

gamma_range = [2**-15,2**-13,2**-11,2**-9,2**-7,2**-5,2**-3,2**-1,2**1,2**3,2**5,2**4,2**6,2**7,2*8]
acc_score = []
opt=[]


from sklearn.multiclass import OneVsRestClassifier


acc_score = []
opt=[]

for C in Cs:
    for g in gamma_range:
        svc =OneVsRestClassifier(SVC(kernel='rbf', gamma=g,C=C))

        scores = cross_val_score(svc, x_train, y_trainlabel, cv=5, scoring='accuracy')

        print("c=",C,";gamma=",g,";score=",np.amax(scores))
        opt.append((C,g,np.amax(scores)))
        opt.sort(key=operator.itemgetter(2),reverse=True)



print("optimal hyperparameter are:","c=",opt[0][0],"gamma=",opt[0][1],"accuracy=",opt[0][2])
optg=opt[0][1]
optc=opt[0][0]

svc =OneVsRestClassifier(SVC(kernel='rbf', gamma=optg,C=optc))
start=time.time()
svc.fit(x_train,y_trainlabel)
end =time.time()
print("training time is",end-start)
y_pred = svc.predict(x_test)
from sklearn.metrics import accuracy_score

print("overall accuracy for rbf",accuracy_score(y_testlabel,y_pred))
acc_score=[]
opt=[]

for C in Cs:

    svc =OneVsRestClassifier(SVC(kernel='linear',C=C))
    svc.fit(x,y)
    scores = cross_val_score(svc, x, y, cv=5, scoring='accuracy')

    print("c=",C,"score=",np.amax(scores))
    opt.append((C,np.amax(scores)))
    opt.sort(key=operator.itemgetter(1),reverse=True)
print("optimal hyperparameter are:","c=",opt[0][0],"accuracy=",opt[0][1])
optc=opt[0][0]

svc =OneVsRestClassifier(SVC(kernel='linear',C=optc))
start=time.time()
svc.fit(x_train,y_trainlabel)
end=time.time()
print("training time",end-start)
y_pred = svc.predict(x_test)

print("overall accuracy for linear",accuracy_score(y_testlabel,y_pred))




acc_score=[]
opt=[]
pol=[1,2,3,4,5,6,7,8]
for g in gamma_range:
    for p in pol:
        for c in Cs:
            svc =OneVsRestClassifier(SVC(kernel='poly',degree=p,C=c,gamma=g))

            scores = cross_val_score(svc, x_train, y_trainlabel, cv=5, scoring='accuracy')

            print("degree=",p,"c=",c,";score=",np.amax(scores))
            opt.append((p,c,np.amax(scores),g))
            opt.sort(key=operator.itemgetter(2),reverse=True)
print("optimal hyperparameter are:","degree=",opt[0][0],"c=",opt[0][1],"gamma=",opt[0][3],"accuracy=",opt[0][2])
optc=opt[0][1]
optp=opt[0][0]
optg=opt[0][3]
svc = OneVsRestClassifier(SVC(kernel='poly',C=optc,degree=optp,gamma=optg))
start=time.time()
svc.fit(x_train,y_trainlabel)
end=time.time()
print("training time",end-start)
y_pred = svc.predict(x_test)


print("overall accuracy for poly",accuracy_score(y_testlabel,y_pred))

acc_score=[]
opt=[]
coeff=[1,0.9,0.8,0.7,-1]
for co in coeff:
    for g in gamma_range:
        for c in Cs:
            svc =OneVsRestClassifier(SVC(kernel='sigmoid', gamma=g,coef0=co,C=c))
            svc.fit(x,y)
            scores = cross_val_score(svc, x, y, cv=5, scoring='accuracy')
            acc_score.append(np.amax(scores))
            print("c=",c,";gamma=",g,"coeff",co,";score=",np.amax(scores))
            opt.append((co,g,c,np.amax(scores)))
            opt.sort(key=operator.itemgetter(3),reverse=True)
print("optimal hyperparameter are:","gamma=",opt[0][1],"c0eff=",opt[0][0],"c=",opt[0][2],"accuracy=",opt[0][3])
optg=opt[0][1]
optco=opt[0][0]
optc=opt[0][2]


svc =OneVsRestClassifier(SVC(kernel='sigmoid',C=optc,gamma=optg,coef0=optco))
start=time.time()
svc.fit(x_train,y_trainlabel)
end=time.time()
print("training time",end-start)
y_pred = svc.predict(x_test)


print("overall accuracy for sigmoid",accuracy_score(y_testlabel,y_pred))
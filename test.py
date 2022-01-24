# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 03:01:05 2022

@author: LEGION
"""



import pickle
model1=pickle.load(open('/content/GB11_model.pkl','rb'))
model2=pickle.load(open('/content/RF22_model.pkl','rb'))
model3=pickle.load(open('/content/XGB33_model.pkl','rb'))
st_model=pickle.load(open('/content/Stack_model.pkl','rb'))

a=float(input('Age='))
w=float(input('WBC='))
l=float(input('Lympocyte='))
m=float(input('Monocyte='))

xt1=pd.DataFrame([a,w,l,m])
xt1=np.array(xt1)
xt1=xt1.reshape(1,-1)

y_prob1=model1.predict_proba(np.array(xt1)) 
y_prob2=model2.predict_proba(np.array(xt1))
y_prob3=model3.predict_proba(np.array(xt1))
x_pr=np.concatenate((y_prob1,y_prob2,y_prob3),1)
y_pr=st_model.predict(np.array(x_pr))
print(y_pr)
if y_pr==0:
  print(f'This subject is COVID-19 negative')
elif y_pr==1:
  print(f'This subject is COVID-19 positive')

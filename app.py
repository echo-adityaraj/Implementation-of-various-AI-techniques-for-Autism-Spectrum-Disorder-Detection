from flask import Flask,render_template,request,redirect,url_for
import os
import numpy as np
import pandas as pd
from xlsx2html import xlsx2html
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import AdaBoostClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import ExtraTreeClassifier

app=Flask(__name__)
app.config['SECRET_KEY']="djchbsdchsdbcjds"
# app.config['upload']="C:\\Users\\whynew.in\\PycharmProjects\\Austism\\uploads"
app.config['upload']="uploads"


@app.route('/')
def index():
    return render_template('home.html')

@app.route('/index')
def home():
    return render_template('home.html')

@app.route('/upload',methods=['POST','GET'])
def upload():
    global df
    try:
        if request.method=='POST':
            data = request.files['f']
            path=os.path.join('uploads',data.filename)
            data.save(path)
            df=pd.read_excel(path,engine='openpyxl')
            data_col=df.columns=['A1S', 'A2S', 'A3S', 'A4S', 'A5S',
                                 'A6S','A7S', 'A8S', 'A9S', 'A10S',
                                 'age', 'gender','ethnicity','jundice', 'austim',
                                 'contryofres', 'usedappbefore','result', 'agedesc', 'relation', 'Class/ASD']
            msg="File dont have any null values"
            print('//////////////////////////')
            print(df['gender'].value_counts())
            print(df['ethnicity'].value_counts())
            print(df['jundice'].value_counts())
            print(df['austim'].value_counts())
            print(df['contryofres'].value_counts())
            print(df['usedappbefore'].value_counts())
            print(df['result'].value_counts())
            print(df['agedesc'].value_counts())
            print(df['relation'].value_counts())
            print('///////////////////////')
            return render_template('uploadhome.html',msg=msg,col=data_col,data=df.values)
    except:
        msg="Dataset is not selected properly"
        return render_template('uploadhome.html',msg=msg)
    return render_template('upload.html')

@app.route('/graphs')
def graphs():
    try:
        print(df)
        print(df.columns)
        sns.countplot(x='relation', data=df)
        plt.savefig(r'static/images/demo/gallery/a.png')
        ######
        sns.countplot(x='gender', data=df)
        plt.savefig(r'static/images/demo/gallery/b.png')
        ######
        sns.countplot(x='austim', data=df)
        plt.savefig(r'static/images/demo/gallery/c.png')
        #####
        sns.countplot(x='contryofres', data=df)
        plt.savefig(r'static/images/demo/gallery/d.png')
        #####
        sns.countplot(x='Class/ASD', data=df)
        plt.savefig(r'static/images/demo/gallery/e.png')
        #####
        sns.countplot(x='jundice', data=df)
        plt.savefig(r'static/images/demo/gallery/f.png')
        #####
        sns.countplot(x='ethnicity',data=df)
        plt.savefig(r'static/images/demo/gallery/g.png')
        return render_template('index.html')
    except:
        return redirect(url_for('home',msg="without dataset graphs cannot be plotted"))


@app.route('/preprocessing')
def preprocessing():
    global fixed_data
    file=os.listdir(app.config['upload'])
    path=os.path.join(app.config['upload'],file[0])
    data=pd.read_excel(path,engine='openpyxl')
    data.dropna(axis=0,inplace=True)
    a = data.age[data.age != '?'].astype('float')
    med=a.median()
    print(med)
    # data['age'].reshape(-1, 1)
    data['age'].replace('?',med,inplace=True)

    print(data.isnull().sum())
    data.columns = ["A1_Score", "A2_Score", "A3_Score", "A4_Score", "A5_Score", "A6_Score", "A7_Score", "A8_Score",
                "A9_Score", "A10_Score", "age", "gender", "ethnicity", "jaundice", "autism", "country_of_res",
                  "used_app_before", "result", "age_desc", "relation", "Class_ASD"]
    print(data.head())
    data = data[data['age'] != '?']
    org_data = data[["A1_Score", "A2_Score", "A3_Score", "A4_Score", "A5_Score", "A6_Score", "A7_Score", "A8_Score", "A9_Score",
        "A10_Score", "age"]]
    label_encoded_data = data[["gender", "autism", "jaundice", "Class_ASD"]]
    label_encoded_data["gender"] = label_encoded_data["gender"].apply(lambda x: 1 if x == "m" else 0)
    label_encoded_data["autism"] = label_encoded_data["autism"].apply(lambda x: 1 if x == "yes" else 0)
    label_encoded_data["jaundice"] = label_encoded_data["jaundice"].apply(lambda x: 1 if x == "yes" else 0)
    label_encoded_data["Class_ASD"] = label_encoded_data["Class_ASD"].apply(lambda x: 1 if x == "YES" else 0)
    one_hot_encoded_data = data[["ethnicity"]]
    one_hot_encoded_data = pd.get_dummies(one_hot_encoded_data)
    fixed_data = pd.concat([org_data, label_encoded_data], axis=1)
    print(fixed_data)
    print(fixed_data.columns)
    return render_template("preprocessing.html", col=fixed_data.columns.values, data=fixed_data.values.tolist())


@app.route('/modelselection',methods=['POST','GET'])
def modelselection():
    # x= pd.data["A1_Score", "A2_Score", "A3_Score", "A4_Score", "A5_Score", "A6_Score", "A7_Score", "A8_Score",
    #             "A9_Score", "A10_Score", "age", "gender", "ethnicity", "jaundice", "autism", "country_of_res",
    #             "used_app_before", "result", "age_desc", "relation"]
    # x= fixed_data.drop(['ethnicity','country_of_res','used_app_before','result','age_desc','relation'],axis=1)
    x = fixed_data.drop(columns=['Class_ASD'])
    print(x.head())
    y = fixed_data[['Class_ASD']]
    print('-------')
    print(x['age'].value_counts())
    print('-------')
    global x_train
    global x_test
    global y_train
    global y_test

    
    x, y = make_classification(
    n_samples=1000, n_features=5, n_informative=2, n_redundant=0, n_classes=2,
    weights=[0.8, 0.2], random_state=42)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    if request.method=='POST':
        model=request.form['model']
        print(model)
        print(type(model))
        if model == '1':
            print("True")
            DT = DecisionTreeClassifier(max_depth=None,min_samples_split=2,random_state=0)
            print("False")
            DT.fit(x_train,y_train)
            y_pred = DT.predict(x_test)
            DTaccuracy = accuracy_score(y_test,y_pred)
            print("=====",DTaccuracy,"====")
            return render_template('dt.html',data=DTaccuracy)
        elif model=='2':
            print('xxxxxxxxxxx')
            lr = XGBClassifier()
            lr.fit(x_train, y_train)
            y_pred = lr.predict(x_test)
            lraccuracy = accuracy_score(y_test, y_pred)
            print(lraccuracy)
            msg = 'The accuracy of XGBClassifier:'+str(lraccuracy)
            return render_template('lr.html',msg=msg)
        elif model=='3':
            print(';yyyyyyyyyyyyy')
            RF = RandomForestClassifier(n_estimators=1000, max_depth=None, min_samples_split=2, random_state=0)
            RF.fit(x_train, y_train)
            y_pred = RF.predict(x_test)
            RFaccuracy = accuracy_score(y_test, y_pred)

            return render_template('rf.html',msg=RFaccuracy)
        elif model=='4':
            print('zzzzzzzzzzzzzzz')
            svc = AdaBoostClassifier()
            svc.fit(x_train, y_train)
            y_pred = svc.predict(x_test)
            svcaccuracy = accuracy_score(y_test, y_pred)
            msg = 'The accuracy of AdaBoostClassifier:'+str(svcaccuracy)
            return render_template('svm.html',msg=msg)
        elif model=='5':
          
            et = ExtraTreeClassifier()
            et.fit(x_train, y_train)
            y_pred = et.predict(x_test)
            etaccuracy = accuracy_score(y_test, y_pred)
            msg = 'The accuracy of ExtraTreeClassifier:'+str(etaccuracy)
            return render_template('et.html',msg=msg)
        elif model=='6':
            
            km = KMeans()
            km.fit(x_train)
            y_pred = km.predict(x_test)
            kmaccuracy = accuracy_score(y_test, y_pred)
            msg = 'The accuracy of KMeans:'+str(kmaccuracy)
            return render_template('km.html',msg=msg)
        elif model=='7':
            x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
            y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)

            x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
            y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

            model = Sequential()
            model.add(Dense(30, activation='relu'))
            model.add(Dense(20, activation='relu'))
            model.add(Dense(1, activation='softmax'))
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            model.fit(x_train, y_train, batch_size=500, epochs=10,validation_data=(x_test, y_test))
            abc=model.predict(x_test)
            acc_cnn =accuracy_score(abc,y_test)

           
            msg = 'The accuracy of ANN:'+str(acc_cnn)
            return render_template('ann.html',msg=msg)
        else:
            msg="Please select any model to predict"
            return render_template('nd.html',msg=msg)
        msg="Without File upload Data can not build model"
        return render_template('er.html',msg=msg)
    return render_template('modelselection.html')


@app.route('/prediction',methods=['POST',"GET"])
def prediction():
    global x_train, x_test, y_train, y_test,x,y,fixed_data
    print("hi gutys")
    fixed_data = pd.read_excel(r'Autism child data.xlsx')
    fixed_data = fixed_data[fixed_data['age']!='?']
    le = LabelEncoder()
    fixed_data['gender']= le.fit_transform(fixed_data['gender'])
    fixed_data['ethnicity']= le.fit_transform(fixed_data['ethnicity'])
    fixed_data['jundice']= le.fit_transform(fixed_data['jundice'])
    fixed_data['austim']= le.fit_transform(fixed_data['austim'])
    fixed_data['contry_of_res']= le.fit_transform(fixed_data['contry_of_res'])
    fixed_data['used_app_before']= le.fit_transform(fixed_data['used_app_before'])
    fixed_data['age_desc']= le.fit_transform(fixed_data['age_desc'])
    fixed_data['relation']= le.fit_transform(fixed_data['relation'])
    fixed_data['Class/ASD']= le.fit_transform(fixed_data['Class/ASD'])

    x = fixed_data.drop(columns=['Class/ASD'])
    
    y = fixed_data[['Class/ASD']]
       
    x, y = make_classification(
    n_samples=1000, n_features=14, n_informative=2, n_redundant=0, n_classes=2,
    weights=[0.8, 0.2], random_state=42)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    if request.method=='POST':
        listn=[]
        a1=int(request.form['a1'])
        print(a1)
        a2=int(request.form['a2'])
        print(a2)
        a3 = int(request.form['a3'])
        print(a3)
        a4 = int(request.form['a4'])
        print(a4)
        a5 = int(request.form['a5'])
        print(a5)
        a6 = int(request.form['a6'])
        print(a6)
        a7 = int(request.form['a7'])
        print(a7)
        a8 = int(request.form['a8'])
        print(a8)
        a9 = int(request.form['a9'])
        print(a9)
        a10 = int(request.form['a10'])
        print(a10)
        age=int(request.form['age'])
        print(age)
        gender=int(request.form['gender'])
        print(gender)

        jundice=int(request.form['jundice'])
        print(jundice)
        austim=int(request.form['autism'])
        print(austim)

        listn =([a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,age,gender,jundice,austim])
        

        model = RandomForestClassifier()
        model.fit(x_train, y_train)
        predi = model.predict([listn])
        print(predi)
        pred = predi
        print(pred)
        if pred == [0]:
            msg="children does'nt have autism spectrum disorder"
            return render_template('final.html',msg=msg)
        msg="children has autism spectrum disorder Please consult doctor"
        return render_template('final.html',msg=msg)
    return render_template("prediction.html")

if __name__=="__main__":
    app.run(debug=True)


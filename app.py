from flask import *
import joblib
from flask import request
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


app = Flask(__name__)
model = joblib.load("model/model_male.sav")

dia_female = pd.read_csv("data/dia-female.csv")
dia_male = pd.read_csv("data/dia-male.csv")

X = dia_female.iloc[:, 0:8]
y = dia_female.iloc[:,8]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

from sklearn import preprocessing 
  
# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder()
dia_male['Family_Diabetes']= label_encoder.fit_transform(dia_male['Family_Diabetes'])
dia_male['Smoking']= label_encoder.fit_transform(dia_male['Smoking'])
dia_male['Alcohol']= label_encoder.fit_transform(dia_male['Alcohol'])
dia_male['RegularMedicine']= label_encoder.fit_transform(dia_male['RegularMedicine'])
dia_male['JunkFood']= label_encoder.fit_transform(dia_male['JunkFood'])
dia_male['BMI'] = dia_male['BMI'].fillna((dia_male['BMI'].mean()))
X1 = dia_male.iloc[:, 0:10]
y1 = dia_male.iloc[:,10]

X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, random_state=1)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/female")
def female():
    return render_template("female.html")

@app.route("/male")
def male():
    return render_template("male.html")



@app.route("/predict", methods = ['POST'])
def predict():
    
    int_features= [float(x) for x in request.form.values()]
    print(int_features,len(int_features))
    final=[np.array(int_features)]
    print(final)
    RF = RandomForestClassifier()
    RF.fit(X_train, y_train)
    prediction = RF.predict(final)
    output=round(prediction[0],2)
    print(output)

    
    if (int(output)==1):
        prediction = "Sorry you chances of getting the Type-II Diabetes. Please consult the doctor immediately!"
    elif (int(output)==2):
        prediction = "Sorry you chances of getting the Gestational diabetes. Please consult the doctor immediately!"
    else:
        prediction = "No need to fear. You have no dangerous symptoms of diabetes and take regular diet to avoid diabetes!"
    return (render_template('female.html', prediction_text = prediction))

@app.route("/predict1", methods = ['POST'])
def predict1():
    
    int_features= [int(x) for x in request.form.values()]
    print(int_features,len(int_features))
    final1=[np.array(int_features)]
    print(final1)
    #RF = RandomForestClassifier()
    #RF.fit(X_train1, y_train1)
    prediction = model.predict(final1)
    output1=round(prediction[0],2)
    print(output1)

    
    if (int(output1)==1):
        prediction = "Sorry you chances of getting the Type-II Diabetes. Please consult the doctor immediately!"
    
    else:
        prediction = "No need to fear. You have no dangerous symptoms of diabetes and take regular diet to avoid diabetes!"
    return (render_template('male.html', prediction_text = prediction))


if __name__ == '__main__':
    app.run(debug = True)


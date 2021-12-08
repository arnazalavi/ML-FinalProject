import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd



app = Flask(__name__)

def ValuePredictor_LR(final_input):
    col_list = ["pclass",	"sex",	"age"	,"sibsp" ,"	parch"	,"fare"]
    #model_LogisticRegression = pickle.load('titanicData.pkl')
    #titanicData_Scaler.pkl
    print(final_input)
    print("Inside LR")
    #Loading the saved decision tree model pickle
    Logistic_model_pkl  = open('titanicData_LR.pkl', 'rb')
    LogisticReg_model = pickle.load(Logistic_model_pkl )
    pred = LogisticReg_model.predict(final_input)
    print ("Logistic Regression model :: ", pred)
    return pred

def ValuePredictor_SVM(final_input):
    #col_list = ["pclass",	"sex",	"age"	,"sibsp" ,"	parch"	,"fare"]
    #model_LogisticRegression = pickle.load('titanicData.pkl')
    #titanicData_Scaler.pkl
    print("Inside LR")
    print(final_input)
    #Loading the saved decision tree model pickle
    SVM_model_pkl  = open('titanicData_SVM.pkl', 'rb')
    SVM_model = pickle.load(SVM_model_pkl )
    pred = SVM_model.predict(final_input)
    print ("SVM model :: ", pred)
    return pred

@app.route('/')
def index():
    return render_template('index.html')



@app.route('/predict',methods=['GET','POST'])
def predict():

    if request.method == 'POST':
     #parse form request in json format
     #parse_request = request.json
     #print(json.dumps(parse_request, indent=4, sort_keys=True))
     #ValuePredictor(parse_request)
     #  
     pickModel =  int(request.form["Pick_model"])
     age = float(request.form["age"])
     pclass = int(request.form["pclass"])
     gender = int(request.form["gender"])
     parch = int(request.form["parch"])
     sibsp = int(request.form["sibsp"])
     fare = float(request.form["fare"])

    
     print(pickModel)

    print(age)
    print(pclass)
    print(gender)

    print(parch)
    print(sibsp)

    print(fare)
    new_titanic_data =[pclass,age,sibsp,parch,fare,gender]
    #ValuePredictor_LR(new_titanic_data)
    #return jsonify(age)
    #return jsonify(new_titanic_data)
     #array to pass 
    final_input = [np.array( new_titanic_data)]

    if pickModel == 0:
        pred = ValuePredictor_LR(final_input)
    else:
        pred = ValuePredictor_SVM(final_input)
    #return pred
    #print(pred)

    Final_output = round(pred[0], 2)
    print(Final_output)
    if Final_output == 0:
         prediction_LR ='Not Survived'
    else:
        prediction_LR= 'Survived'

   
    titanic_data= {}
    titanic_data['age'] = age
    titanic_data['Passenger_Class'] =pclass
    if gender == 0:
        titanic_data['Sex'] = 'Female'
    else: titanic_data['Sex'] = 'Male'
    titanic_data['parch']= parch
    titanic_data['sibsp'] = sibsp
    titanic_data['Fare'] =fare
    titanic_data['Prediction'] =prediction_LR
    if pickModel == 0:
         titanic_data['model'] ='Logical Regression'
    else: 
        titanic_data['model'] = 'SVM'
   

    titanic_outputdata = pd.DataFrame({'AGE':[age], 'PASSENGER_CLASS':[pclass],'SEX':[gender],'PARCH':[parch],'SIBBLING':[sibsp],'FARE':[fare],'PREDICTION':[prediction_LR]})
    print(titanic_outputdata)

    #return pred
    #predict = ValuePredictor
    #return render_template('results.html',pred=pred)
    #return render_template('results.html',prediction_LR='Survival prediction {}'.format(Final_output))
    #return render_template('results.html',prediction_LR=titanic_outputdata)
    return render_template('results.html',prediction_model=titanic_data)


if __name__ == "__main__":
    app.run(debug=True)
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


app = Flask(__name__)

def ValuePredictor_LR(final_input):
    col_list = ["pclass",	"sex",	"age"	,"sibsp" ,"	parch"	,"fare"]
    #model_LogisticRegression = pickle.load('titanicData.pkl')
    #titanicData_Scaler.pkl
    print(final_input)
    #Loading the saved decision tree model pickle
    Logistic_model_pkl  = open('titanicData.pkl', 'rb')
    LogisticReg_model = pickle.load(Logistic_model_pkl )
    pred = LogisticReg_model.predict(final_input)
    print ("Loaded Decision tree model :: ", pred)
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
     
     age = float(request.form["age"])
     pclass = int(request.form["pclass"])
     gender = int(request.form["gender"])
     parch = int(request.form["parch"])
     sibsp = int(request.form["sibsp"])
     fare = float(request.form["fare"])

    print(age)
    print(pclass)
    print(gender)

    print(parch)
    print(sibsp)

    print(fare)
    new_titanic_data =[age,pclass,gender,parch,sibsp,fare]
    #ValuePredictor_LR(new_titanic_data)
    #return jsonify(age)
    #return jsonify(new_titanic_data)
     #array to pass 
    final_input = [np.array( new_titanic_data)]

    pred = ValuePredictor_LR(final_input)
    #return pred
    print(pred)
    #return pred
    #predict = ValuePredictor
    return render_template('results.html',pred=pred)
   

if __name__ == "__main__":
    app.run(debug=True)
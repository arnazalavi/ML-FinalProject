import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import joblib

app = Flask(__name__)

def ValuePredictor(to_predict_list):
    col_list = ["pclass",	"sex",	"age"	,"sibsp" ,"	parch"	,"fare"]
    #model_LogisticRegression = joblib.load('logistic_reg.sav')
    

# Loading the saved decision tree model pickle
#decision_tree_model_pkl = open(decision_tree_pkl_filename, 'rb')
#decision_tree_model = pickle.load(decision_tree_model_pkl)
#print "Loaded Decision tree model :: ", decision_tree_model

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
     
     age = request.form["age"]
     pclass = request.form["pclass"]
     gender = request.form["gender"]
     parch = request.form["parch"]
     sibsp = request.form["sibsp"]
     fare = request.form["fare"]

    print(age)
    print(pclass)
    print(gender)
    print(parch)
    print(sibsp)
    print(fare)
    return jsonify(age)

     #array to pass 
    #final_input = [np.array(parse_request)]

   # predict = ValuePredictor
    #return render_template('index.html')
   

if __name__ == "__main__":
    app.run(debug=True)
import pickle  
from flask import Flask , request, app , jsonify , url_for , render_template  
import numpy as np 
import pandas as pd  



app= Flask(__name__)  

# Load the model    
regmodel =pickle.load(open('regmodel.pkl' , 'rb') ) 
scaler = pickle.load(open('scaling.pkl' , 'rb') )  

@app.route('/') 
def home(): 
    return render_template('home.html')  


@app.route('/predict_api', methods=['POST'])
 
def predict_api():
    data=request.json['data']                                   # the ip we are giving which is in the json format which is capture in form data key  as we hit predict_api 
    print(data)                                                  # next we get data in form of  key values  pair we require values of it  
    print(np.array(list(data.values())).reshape(1,-1)) 
    
    new_data = scaler.transform(np.array(list(data.values())).reshape(1,-1))   # tranforming data.values for prediction 
    output= regmodel.predict(new_data) 
    print(output[0])                                                       # it is in form of 2d aray 
    return jsonify(output[0])   



# now we create small web application where we provide the input fin term of form  as we provide data we predict it val 

@app.route('/predict' , methods=['POST'])
def predict() : 
    data=[float(x) for x in request.form.values()]           # we taking values  
    final_input = scaler.transform(np.array(data).reshape(1,-1) ) 
    print(final_input)  
    output = regmodel.predict(final_input)[0]
    return render_template('home.html' , prediction_text = "The House price Prediction is :{}".format(output) )
    
    

if __name__ == '__main__': 
    app.run(debug=True)
    

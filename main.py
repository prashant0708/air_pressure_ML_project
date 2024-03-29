from flask import Flask , render_template, request
import pickle
import numpy as np
import pandas as pd

app=Flask(__name__)

model=pickle.load(open('model.pkl','rb'))



## render the template 

@app.route('/')
def air_pressure():
    return render_template('air_pressure.html')




@app.route("/ml_prediction",methods=["post"])
def air_pressure_prediction():

    # get the input values from the form
    data=[float(x) for x in request.form.values()]
    feature=[np.array(data)]
    

    ## make the prediction using the loaded model
    predictions=model.predict(feature)[0]
    return render_template('air_pressure.html',result=round(predictions,2))




if __name__=="__main__":
    app.run(debug=True)



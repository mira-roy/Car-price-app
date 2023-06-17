from flask import Flask, render_template,request
import pandas as pd
import pickle
import numpy as np
app = Flask(__name__)
model = pickle.load(open("LinearRegressionModel.pkl","rb"))

car=pd.read_csv('Cleaned_Car_data.csv')

@app.route('/')
def index():
    Brand_Name = sorted(car['Brand Name'].unique())
    Model = sorted(car['Model Variant'].unique())
    Year =sorted(car['Year'].unique(),reverse=True)
    Fuel_Type = sorted(car['Fuel Type'].unique())
    Brand_Name.insert(0,"Select Company")
    return render_template('index.html',Brand_Name=Brand_Name,Model=Model,Year=Year,Fuel_Type=Fuel_Type)
@app.route("/predict",methods=["POST"])
def predict():
    brand =request.form.get("company")
    car_model= request.form.get("model")
    year= int(request.form.get("year"))
    fuel= request.form.get("fuel_type")
    kms= int(request.form.get("kilo_driven"))
    print(brand,car_model,year,fuel,kms)

    prediction= model.predict(pd.DataFrame([[brand,year,fuel,kms,car_model]],columns=["Brand Name","Year","Fuel Type","Kms Driven","Model Variant"]))
    print(prediction)
    return str(np.round(prediction[0],2))
if(__name__=='__main__'):
    app.run(debug=True)
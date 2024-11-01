from flask import Flask,render_template,request,session
import pickle
import numpy as np
import requests
from bs4 import BeautifulSoup
randomforest_model=pickle.load(open('air1.pkl','rb'))
app = Flask(__name__)
@app.route("/")
def air():
    return render_template("index.html")

@app.route("/location")
def location():
    return render_template('location.html')

@app.route("/predict",methods=["post"])
def predict():
    state_0=request.form['locationinput']
    state_city={'Delhi':'delhi','Haryana':'haryana','Uttar_Pradesh':'uttar-pradesh','Maharashtra':'maharashtra',
                'Karnataka':'karnataka','Madhya Pradesh':'madhya-pradesh','West_Bengal':'west-bengal',
                'Rajasthan':'rajasthan','Punjab':'punjab','Telangana':'telangana','Bihar':'bihar',
                'TamilNadu':'tamilnadu','Gujarat':'gujarat','Andhra_Pradesh':'andhra-pradesh','Kerala':'kerala',
                'Odisha':'odisha','Jharkhand':'jharkhand','Assam':'assam','Chandigarh':'chandigarh',
                'Meghalaya':'meghalaya'}
    state_1=state_city[state_0]
    url = 'https://www.aqi.in/dashboard/india/'+state_1
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    target_class = 'Pollutants_sensor_text'
    span_elements = soup.find_all('span', class_=target_class)
    pollutants=[]
    for span in span_elements:
        if "," in span.text:
            cleaned_string=span.text.replace(",","")
            span_text=int(cleaned_string)
        else:
            span_text = int(float(span.text))
        pollutants.append(span_text)
    poll_dict={'pm2.5':pollutants[0],'pm10':pollutants[1],'so2':pollutants[2],'co':pollutants[3],
               'ozone':pollutants[4],'no2':pollutants[5]}

    return render_template("predict.html",poll_dict=poll_dict,state=state_0)

@app.route("/result",methods=['post','get'])
def result():
    city_data=request.form['locationinput1']
    dict_city={'Delhi':0,'Haryana':1,'Uttar_Pradesh':2,'Maharashtra':3,'Karnataka':4,'Madhya Pradesh':5,'West_Bengal':6,'Rajasthan':7,'Punjab':8,'Telangana':9,'Bihar':10,'TamilNadu':11,'Gujarat':12,'Andhra_Pradesh':13,'Kerala':14,'Odisha':15,'Jharkhand':16,'Assam':17,'Chandigarh':18,'Meghalaya':19}
    
    city_data_binary=dict_city[city_data]
    data1=float(request.form['pm2.5'])
    data2=float(request.form['pm10'])
    data3=float(request.form['so2'])
    data4=float(request.form['co'])
    data5=float(request.form['ozone'])
    data6=float(request.form['no2'])
    arr=np.array([[city_data_binary,data1,data2,data6,data3,data4,data5]])
    pred=randomforest_model.predict(arr)
    
    pred_value=int(pred[0])
    pred=pred[0]
    
    def value_function(pred_value):
        if pred_value in range(0,51):
            return "Good"
        elif pred_value in range(51,101):
            return "moderate"
        elif pred_value in range(101,201):
           return "Unhealthy for sensitive groups"
        elif pred_value in range(201,301):
           return "Unhealthy"
        elif pred_value in range(301,401):
           return "Very Unhealthy"
        else:
            return "Hazardous"
    value=value_function(pred_value)
    return render_template('result.html',data=pred,data2=value)

if __name__ == "__main__":
    app.run(debug=True)

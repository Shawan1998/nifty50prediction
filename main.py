from flask import Flask,render_template,request
import numpy as np
import pickle


from sklearn.preprocessing import PolynomialFeatures
polynom = PolynomialFeatures(degree =2)

with open('model.pkl','rb') as model_file:
    model = pickle.load(model_file)

app = Flask(__name__)

@app.route('/')

def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])

def stockmarket():
    year=request.form['year']
    prediction=str((model.predict(polynom.fit_transform([[year]]))))

    head,sep,tail=prediction.partition('.')



    return render_template('index.html', pred=head[2:])




app.run()

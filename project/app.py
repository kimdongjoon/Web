from flask import Flask, render_template, request
import os
import joblib
import numpy as np
from tensorflow import keras
from keras.models import load_model

app = Flask(__name__)
app.debug = True

model_iris_lr = None
model_iris_svm = None
model_iris_dt = None
model_iris_deep = None

def load_iris():
    global model_iris_lr, model_iris_svm , model_iris_dt, model_iris_deep
    model_iris_lr = joblib.load(os.path.join(app.root_path,'model/iris_lr.pkl'))
    model_iris_svm = joblib.load(os.path.join(app.root_path,'model/iris_svm.pkl'))
    model_iris_dt = joblib.load(os.path.join(app.root_path,'model/iris_dt.pkl'))
    # deep러닝은 다른 방식으로 load 해야 함. 
    # model_iris_deep = load_model(os.path.join(app.root_path,'model/iris_deep.hdf5'))
@app.route('/')
def index():
    menu = {'home':True, 'rgrs':False, 'stmt':False, 'clsf':False, 'clst':False, 'user':False}
    return render_template('home.html', menu=menu)

@app.route('/regression', methods=['GET', 'POST'])
def regression():
    menu = {'home':False, 'rgrs':True, 'stmt':False, 'clsf':False, 'clst':False, 'user':False}
    if request.method == 'GET':
        return render_template('regression.html', menu=menu)
    else:
        sp_names = ['Setosa', 'Versicolor', 'Virginica']
        slen = float(request.form['slen'])      # Sepal Length
        swid = float(request.form['swid'])      # Petal Width
        plen = float(request.form['plen'])      # Petal Length
        pwid = float(request.form['pwid'])      # Petal Width
        sp = int(request.form['species'])       # Species
        species = sp_names[sp]
        swid = 0.63711424 * slen - 0.53485016 * plen + 0.55807355 * pwid - 0.12647156 * sp + 0.78264901
        swid = round(swid, 4)
        iris = {'slen':slen, 'swid':swid, 'plen':plen, 'pwid':pwid, 'species':species}
        return render_template('reg_result.html', menu=menu, iris=iris)

@app.route('/sentiment')
def sentiment():
    pass

@app.route('/classification')
def classification():
    pass

@app.route('/classification_iris', methods=['GET', 'POST'])
def classification_iris():
    menu = {'home':False, 'rgrs':False, 'stmt':False, 'clsf':True, 'clst':False, 'user':False}
    if request.method == 'GET':
        return render_template('classification_iris.html', menu=menu)
    else:
        sp_names = ['Setosa', 'Versicolor', 'Virginica']
        slen = float(request.form['slen'])      # Sepal Length
        swid = float(request.form['swid'])      # Petal Width
        plen = float(request.form['plen'])      # Petal Length
        pwid = float(request.form['pwid'])      # Petal Width

        test_data = np.array(slen, swid, plen, pwid).reshape(1,4)

        species_lr   = sp_names[model_iris_lr.predict(test_data)][0]
        species_svm  = sp_names[model_iris_svm.predict(test_data)][0]
        species_dt   = sp_names[model_iris_dt.predict(test_data)][0]
        
        #species_deep = sp_names[model_iris_deep.predict_classes(test_data)[0]

        iris = {'slen':slen, 'swid':swid, 'plen':plen, 'pwid':pwid, 
                'species_lr':species_lr,'species_svm':species_svm,
                'species_dt':species_dt#,'species_deep':species_deep
                }

        return render_template('cla_iris_result.html', menu=menu, iris=iris)
        


@app.route('/clustering')
def clustering():
    pass

if __name__ == '__main__':
    load_iris()
    app.run(debug=True)
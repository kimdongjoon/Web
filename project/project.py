from flask import Flask, render_template, request
app = Flask(__name__) 

@app.route('/') 
def index(): 
    menu = {'home' : True, 'rgrs' : False , 'stmt' : False , 'clsf' : False , 'user':False}
    return render_template('base.html', menu = menu) 


@app.route('/regression', methods=['GET','POST'])
def regression():
    menu = {'home' : False, 'rgrs' : True , 'stmt' : False , 'clsf' : False , 'clst' : False ,'user':False}

    if request.method == "GET":
        return render_template('regression.html', menu = menu) 
    else:
        return render_template('regression_result.html', nemu = menu)#, iris=iris) 

@app.route('/sentimentation')
def sentimentation():
    menu = {'home' : False, 'rgrs' : False , 'stmt' : True , 'clsf' : False , 'clst' : False, 'user':False}
    return render_template('sentimentation.html', menu = menu) 

@app.route('/classification')
def classification():
    menu = {'home' : False, 'rgrs' : False , 'stmt' : False , 'clsf' : True , 'clst' : False, 'user':False}
    return render_template('classification.html', menu = menu) 

@app.route('/clustering')
def clustering():
    menu = {'home' : False, 'rgrs' : False , 'stmt' : False , 'clsf' : False , 'clst' : True, 'user':False}
    return render_template('clustering.html', nemu = menu) 


if __name__ == '__main__':
    app.run(debug=True)

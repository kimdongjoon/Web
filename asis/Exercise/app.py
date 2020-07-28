from flask import Flask, render_template, request
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('01_home.html')

@app.route('/typo')
def typo():
    return render_template('03_typography.html')

@app.route('/iris', methods=['GET', 'POST'])
def iris():
    if request.method == 'GET':
        return render_template('12_form-iris.html')
    else:
        slen1 = float(request.form['slen']) * 2
        plen1 = float(request.form['plen']) * 2
        pwid1 = float(request.form['pwid']) * 2
        species1 = int(request.form['species'])
        comment1 = request.form['comment']
        return render_template('12_iris-result.html', 
                slen=slen1, plen=plen1, pwid=pwid1, species=species1, comment=comment1)

@app.route('/project')
def project():
    return render_template('17_templates.html')

@app.route('/hello')
@app.route('/hello/<name>')
def hello(name=None):
    return render_template('hello.html', name=name)

if __name__ == '__main__':
    app.run(debug=True)
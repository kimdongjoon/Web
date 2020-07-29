from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import joblib
from konlpy.tag import Okt
import re
import os
from tensorflow import keras
from keras.models import load_model
from keras.applications.vgg16 import VGG16, decode_predictions
import numpy as np
from PIL import Image
from clu_util import clustering_iris

app = Flask(__name__)
app.debug=True

stopwords=['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
okt = Okt()
tfidf_vector = None
model_lr = None
dtmvector = None
model_nb = None
vgg = None
model_iris_deep = None
model_iris_lr = None
model_iris_svm = None
model_iris_dt = None

def load_lr():
    global tfidf_vector, model_lr
    tfidf_vector = joblib.load(os.path.join(app.root_path, 'model/movie_lr_dtm.pkl'))
    model_lr = joblib.load(os.path.join(app.root_path, 'model/movie_lr.pkl'))

def tw_tokenizer(text):
    # 입력 인자로 들어온 text 를 형태소 단어로 토큰화 하여 list 객체 반환
    tokens_ko = okt.morphs(text)
    return tokens_ko

def lr_transform(review):
    review = re.sub(r"\d+", " ", review)
    test_dtm = tfidf_vector.transform([review])
    return test_dtm

def load_nb():
    global dtmvector, model_nb
    dtmvector = joblib.load(os.path.join(app.root_path, 'model/movie_nb_dtm.pkl'))
    model_nb = joblib.load(os.path.join(app.root_path, 'model/movie_nb.pkl'))

def nb_transform(review):
    review = review.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
    morphs = okt.morphs(review, stem=True) # 토큰화
    test = ' '.join(morph for morph in morphs if not morph in stopwords)
    test_dtm = dtmvector.transform([test])
    return test_dtm

def load_vgg():
    global vgg
    vgg = VGG16()

def load_iris():
    global model_iris_deep, model_iris_lr, model_iris_svm, model_iris_dt
    model_iris_deep = load_model(os.path.join(app.root_path, 'model/iris.hdf5'))
    model_iris_lr = joblib.load(os.path.join(app.root_path, 'model/iris_lr.pkl'))
    model_iris_svm = joblib.load(os.path.join(app.root_path, 'model/iris_svm.pkl'))
    model_iris_dt = joblib.load(os.path.join(app.root_path, 'model/iris_dt.pkl'))

@app.route('/')
def index():
    menu = {'home':True, 'regression':False, 'senti':False, 'classification':False, 'clustering':False, 'member':False, 'creative' : False}
    return render_template('home.html', menu=menu)

@app.route('/regression', methods=['GET', 'POST'])
def regression():
    menu = {'home':False, 'regression':True, 'senti':False, 'classification':False, 'clustering':False, 'member':False, 'creative' : False}
    if request.method == 'GET':
        return render_template('regression.html', menu=menu)
    else:
        sp_names = ['Setosa', 'Versicolor', 'Virginica']
        slen = float(request.form['slen'])      # Sepal Length
        plen = float(request.form['plen'])      # Petal Length
        pwid = float(request.form['pwid'])      # Petal Width
        sp = int(request.form['species'])       # Species
        species = sp_names[sp]
        swid = 0.63711424 * slen - 0.53485016 * plen + 0.55807355 * pwid - 0.12647156 * sp + 0.78264901
        swid = round(swid, 4)
        iris = {'slen':slen, 'swid':swid, 'plen':plen, 'pwid':pwid, 'species':species}
        return render_template('reg_result.html', menu=menu, iris=iris)

@app.route('/senti', methods=['GET', 'POST'])
def senti():
    menu = {'home':False, 'regression':False, 'senti':True, 'classification':False, 'clustering':False, 'member':False, 'creative' : False}
    if request.method == 'GET':
        return render_template('senti.html', menu=menu)
    else:
        review = request.form['review']
        review_dtm_lr = lr_transform(review)
        lr_result = model_lr.predict(review_dtm_lr)[0]
        review_dtm_nb = nb_transform(review)
        nb_result = model_nb.predict(review_dtm_nb)[0]
        lr = '긍정' if lr_result else '부정'
        nb = '긍정' if nb_result else '부정'
        movie = {'review':review, 'lr':lr, 'nb':nb}
        return render_template('senti_result.html', menu=menu, movie=movie)

@app.route('/iris_classification', methods=['GET', 'POST'])
def iris_classification():
    menu = {'home':False, 'regression':False, 'senti':False, 'classification':True, 'clustering':False, 'member':False, 'creative' : False}
    if request.method == 'GET':
        return render_template('iris_classification.html', menu=menu)
    else:
        sp_names = ['Setosa', 'Versicolor', 'Virginica']
        sp_count = [0,0,0]

        slen = float(request.form['slen'])      # Sepal Length
        swid = float(request.form['swid'])      # Sepal Width
        plen = float(request.form['plen'])      # Petal Length
        pwid = float(request.form['pwid'])      # Petal Width
        iris_test = np.array([slen, swid, plen, pwid]).reshape(1, 4)
        sp = np.argmax(model_iris_deep.predict(iris_test))

        # Voting 추가.
        # 앙상블 기법을 적용하는게 아닌 각각의 결과에 대한 가중치를 부여한다. 
        # deep = 1.9      lr = 1      svm = 1.5     dt = 1
        # 도합 5를 기준으로 가장 높은 값을 선택하여 최종결과로 리턴한다. 
        
        sp_count[sp] += 1.9
        sp_count[model_iris_lr.predict(iris_test)[0]] += 1
        sp_count[model_iris_svm.predict(iris_test)[0]] += 1.5
        sp_count[model_iris_dt.predict(iris_test)[0]] += 1

        species_deep = sp_names[sp]
        species_lr = sp_names[model_iris_lr.predict(iris_test)[0]]
        species_svm = sp_names[model_iris_svm.predict(iris_test)[0]]
        species_dt = sp_names[model_iris_dt.predict(iris_test)[0]]

        vote_index = np.argmax(sp_count)
        species_vote = sp_names[vote_index]
        species_bool = {'Setosa':False, 'Versicolor':False, 'Virginica':False}
        species_bool[species_vote] = True


        iris = {'slen':slen, 'swid':swid, 'plen':plen, 'pwid':pwid, 
                'species_deep':species_deep, 'species_lr':species_lr,
                'species_svm':species_svm, 'species_dt':species_dt, 'species_vote' : species_vote , 'vote_index' : vote_index, 'species_bool':species_bool}
        return render_template('iris_cla_result.html', menu=menu, iris=iris)

@app.route('/classification', methods=['GET', 'POST'])
def classification():
    menu = {'home':False, 'regression':False, 'senti':False, 'classification':True, 'clustering':False, 'member':False, 'creative' : False}
    if request.method == 'GET':
        return render_template('classification.html', menu=menu)
    else:
        f = request.files['image']
        filename = os.path.join(app.root_path, 'static/images/uploads/') + secure_filename(f.filename)
        f.save(filename)
        img = np.array(Image.open(filename).resize((224, 224)))
        yhat = vgg.predict(img.reshape(-1, 224, 224, 3))
        label_key = np.argmax(yhat)
        label = decode_predictions(yhat)
        label = label[0][0]
        return render_template('cla_result.html', menu=menu, 
                                filename=secure_filename(f.filename), 
                                label=label[1], pct='%.2f' % (label[2]*100))

@app.route('/clustering', methods=['GET', 'POST'])
def clustering():
    menu = {'home':False, 'regression':False, 'senti':False, 'classification':False, 'clustering':True, 'member':False, 'creative' : False}
    if request.method == 'GET':
        return render_template('clustering.html', menu=menu)
    else:
        f = request.files['csv']
        filename = 'static/images/uploads/' + secure_filename(f.filename)
        f.save(os.path.join(app.root_path, filename))
        ncls = int(request.form['K'])
        clustering_iris(app, ncls, filename=filename)

        file_path = os.path.join(app.root_path, 'static/images/kmc.png')
        mtime = int(os.stat(file_path).st_mtime)
        return render_template('clu_result.html', menu=menu,  
                                K=ncls, mtime=mtime)


@app.route('/member/KDJ', methods = ['GET','POST'])
def KDJ():
    menu = {'home':False, 'regression':False, 'senti':False, 'classification':False, 'clustering':False, 'member':True, 'creative' : False}
    if request.method == 'GET':
        return render_template('/member/KDJ.html', menu=menu)
    else:
        pass


@app.route('/member/KTH', methods = ['GET','POST'])
def KTH():
    menu = {'home':False, 'regression':False, 'senti':False, 'classification':False, 'clustering':False, 'member':True, 'creative' : False}
    if request.method == 'GET':
        return render_template('/member/KTH.html', menu=menu)
    else:
        pass

@app.route('/creative', methods = ['GET','POST'])
def creative():
    menu = {'home':False, 'regression':False, 'senti':False, 'classification':False, 'clustering':False, 'member':False, 'creative' : True}
    if request.method == 'GET':
        return render_template('/creative.html', menu=menu)
    else:
        pass

@app.route('/shutdown', methods = ['GET','POST'])




@app.errorhandler(404)
def page_not_found(error):
    return render_template('page_not_found.html'), 404

if __name__ == '__main__':
    load_lr()
    load_nb()
    load_vgg()
    load_iris()
    app.run()
    #app.run(host='0.0.0.0')     #외부에서 접속시 

### 테스트
# @app.route('/')
# def index():
#     menu = {'home':True, 'regression':False, 'senti':False, 'classification':False, 'clustering':False}
#     return render_template('home.html', menu=menu)


# @app.errorhandler(404)
# def page_not_found(error):
#     return render_template('page_not_found.html'), 404

# if __name__ == '__main__':
#     app.run()
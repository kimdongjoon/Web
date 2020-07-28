import joblib
from konlpy.tag import Okt

stopwords=['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
review = '와 진짜 너무 재미있다 짜릿하고 소름 돋고 화나고 공감되고 ㅠㅠ 언니들 최고다 또 봐야지'
rev = review.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
print(rev)

okt = Okt()
morphs = okt.morphs(rev, stem=True) # 토큰화
test = ' '.join(morph for morph in morphs if not morph in stopwords)
print(test)

dtmvector = joblib.load('model/movie_nb_dtm.pkl')
test_dtm = dtmvector.transform([test])
print(test_dtm.shape)

model_nb = joblib.load('model/movie_nb.pkl')
predicted = model_nb.predict(test_dtm)

print(predicted)
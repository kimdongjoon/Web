import joblib
import re
from konlpy.tag import Okt

review = '너무 좋아요. 미투 운동의 OO점이네요. 당신들의 용기로 많은 여성들의 숨통이 열렸습니다'
review = re.sub(r"\d+", " ", review)
print(review)

okt = Okt()
def tw_tokenizer(text):
    # 입력 인자로 들어온 text 를 형태소 단어로 토큰화 하여 list 객체 반환
    tokens_ko = okt.morphs(text)
    return tokens_ko

tfidf_vector = joblib.load('model/movie_lr_dtm.pkl')
test_dtm = tfidf_vector.transform([review])
print(test_dtm.shape)

model_lr = joblib.load('model/movie_lr.pkl')
predicted = model_lr.predict(test_dtm)

print(predicted)
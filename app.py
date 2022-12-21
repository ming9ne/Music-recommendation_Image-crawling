# streamlit run app.py

import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import streamlit as st
from PIL import Image
import pandas as pd

model = load_model('model2.h5')         # 학습시킨 모델 호출
st.title('사진을 통한 음악 추천 프로그램')

st.markdown('''
사진을 넣어주세요
''')

uploaded_file = st.file_uploader("", type="jpg")  # 분석할 사진 업로드
if uploaded_file is not None:       # 사진이 올라갔을 때 조건문 실행
    
    image = Image.open(uploaded_file)
    image = image.resize((256,256), Image.NEAREST)
    st.image(image, caption='Uploaded Image.', use_column_width=False)
    st.write("")

    # 이미지 사이즈 조정
    image = np.asarray(image)/255.0
    image = image.reshape(1,256,256,3)

    # 모델 예측
    yhat = model.predict(image)
    st.write(yhat)

    # label에 모델 예측값 중 최댓값 저장(가장 유사 값)
    label = np.argmax(yhat, axis=1)[0]
    dic = {0:'슬픔', 1:'행복'}
    result = dic[label]
    st.write('이 사진은 {:.1%}의 확률로 {}한 분위기로 예상됩니다.'.format(np.max(yhat), result))
    st.write(label) #label 값 확인 
  
  

    if label == 1:  # 1일 경우 행복(dic 값)
        df = pd.read_csv('행복한 노래.csv',index_col=0) # 인덱스 제거
        df = df.sample(frac=1).reset_index(drop=True)   # 셔플
        for i in range(3):
            st.write('추천 노래 링크 : https://www.melon.com/song/detail.htm?songId={}'.format(df['song_id'][i]))

    else:           # 0일 경우 슬픔(dic 값)
        df = pd.read_csv('슬픈 노래.csv',index_col=0)
        df = df.sample(frac=1).reset_index(drop=True)
        for i in range(3):
            st.write('추천 노래 링크 : https://www.melon.com/song/detail.htm?songId={}'.format(df['song_id'][i]))
    
from flask import Flask,request,render_template
import pandas as pd
import pickle
import numpy as np

model=pickle.load(open('final_model.pkl','rb'))

cv=pickle.load(open('count_vectorizer_pickle.pkl','rb'))

app=Flask(__name__)


@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
	if request.method== 'POST':
		msg=request.form['text_area']
		arry=[msg]
		cv_trans=cv.transform(arry).toarray()
		predn=model.predict(cv_trans)
	return render_template('result.html',prediction=predn)

if __name__ == '__main__':
	app.run(debug=True)
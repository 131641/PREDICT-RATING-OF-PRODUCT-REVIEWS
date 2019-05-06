import numpy as np
import pickle
from flask import Flask,request, render_template
import os

app = Flask(__name__)

# Vector convertion
## General convertion
def bow(review):
	loaded_model = pickle.load(open('model1/BOW.sav', 'rb'))
	result = loaded_model.transform(np.array([review]))
	print("Converted using BOW")
	return result
	
def tfidf(review):
	loaded_model = pickle.load(open('model1/TFIDF.sav', 'rb'))
	result = loaded_model.transform(np.array([review]))
	print("Converted using TFIDF")
	return result
	
# For Naive Bayes
def bown(review):
	loaded_model = pickle.load(open('model/BOW.sav', 'rb'))
	result = loaded_model.transform(np.array([review]))
	print("Converted using BOW of Naive bayes")
	return result
	
def tfidfn(review):
	loaded_model = pickle.load(open('model/TFIDF.sav', 'rb'))
	result = loaded_model.transform(np.array([review]))
	print("Converted using TFIDF of Naive bayes")
	return result
	
# Algorithms
## KNN
def knn_bow(review):
	review = bow(review)
	loaded_model = pickle.load(open('model1/KNN_BOW.sav', 'rb'))
	result = loaded_model.predict(review)
	print(result)
	return result
	
def knn_tfidf(review):
	review = tfidf(review)
	loaded_model = pickle.load(open('model1/KNN_TFIDF.sav', 'rb'))
	result = loaded_model.predict(review)
	print(result)
	return result

## Logistic Regression
def lr_bow(review):
	review = bow(review)
	loaded_model = pickle.load(open('model1/LR_BOW.sav', 'rb'))
	result = loaded_model.predict(review)
	print(result)
	return result
	
def lr_tfidf(review):
	review = tfidf(review)
	loaded_model = pickle.load(open('model1/LR_TFIDF.sav', 'rb'))
	result = loaded_model.predict(review)
	print(result)
	return result
	
## Decision Tree
def dt_bow(review):
	review = bow(review)
	loaded_model = pickle.load(open('model1/DT_BOW.sav', 'rb'))
	result = loaded_model.predict(review)
	print(result)
	return result
	
def dt_tfidf(review):
	review = tfidf(review)
	loaded_model = pickle.load(open('model1/DT_TFIDF.sav', 'rb'))
	result = loaded_model.predict(review)
	print(result)
	return result
	
## SVM
def svm_bow(review):
	review = bow(review)
	loaded_model = pickle.load(open('model1/SVM_BOW.sav', 'rb'))
	result = loaded_model.predict(review)
	print(result)
	return result
	
def svm_tfidf(review):
	review = tfidf(review)
	loaded_model = pickle.load(open('model1/SVM_TFIDF.sav', 'rb'))
	result = loaded_model.predict(review)
	print(result)
	return result
	
## Naive Bayes
def naive_bow(review):
	review = bown(review)
	loaded_model = pickle.load(open('model/Naive_Bow.sav', 'rb'))
	pred = loaded_model.predict(review)
	print("pred",pred)
	return pred

def naive_tfidf(review):
	review = tfidfn(review)
	loaded_model = pickle.load(open('model/Naive_Tfidf.sav', 'rb'))
	pred = loaded_model.predict(review)
	print("pred",pred)
	return pred

@app.route('/')
def welcome():
	return render_template('amazon.html')
	
	
@app.route('/predict', methods = ["GET","POST"])
def predict():
	if request.method == 'POST':
	  algo = request.form['algo']
	  review = request.form['review']
	  print(review)
	  print(algo)
	  if(algo == '1'):
	  	pred_bow = knn_bow(review)
	  	pred_tfidf = knn_tfidf(review)
	  	print("BOW : "+str(pred_bow[0])+" TFIDF : "+str(pred_tfidf[0]))
	  	return "BOW : "+str(pred_bow[0])+" TFIDF : "+str(pred_tfidf[0])
	  elif(algo == '2'):
	  	pred_bow = naive_bow(review)
	  	pred_tfidf = "Model Not yet deployed`" #naive_tfidf(review)
		if(int(pred_bow[0]) == 1):
	  		bow_output = "Positive"
	  	else:
	  		bow_output = "Negative"
		return render_template('upload.html', pred_bow = bow_output, pred_tf = pred_tfidf)
	  elif(algo == '3'):
	  	pred_bow = lr_bow(review)
	  	pred_tfidf = lr_tfidf(review)
	  elif(algo == '4'):
	  	pred_bow = dt_bow(review)
	  	pred_tfidf = dt_tfidf(review)	  	
	  elif(algo == '5'):
	  	pred_bow = svm_bow(review)
	  	pred_tfidf = svm_tfidf(review)
	  else:
	  	return "Somthing went wrong"
		 
	  if(int(pred_bow[0]) == 1):
	  	bow_output = "Positive"
	  else:
	  	bow_output = "Negative"
	  if(int(pred_tfidf[0]) == 1):
	  	tfidf_output = "Positive"
	  else:
	  	tfidf_output = "Negative"
	  	
	  return render_template('upload.html', pred_bow = bow_output, pred_tf = tfidf_output)
		  
if __name__ == '__main__':
	port = int(os.environ.get('PORT', 5000))
	app.run(host='0.0.0.0', port=port, debug = True)	


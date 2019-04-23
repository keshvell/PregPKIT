from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	url = "https://raw.githubusercontent.com/keshvell/PregPKIT/master/cleanedflask.csv"
	df= pd.read_csv(url)
	df_data = df[["Cleantext","Target"]]
	# Features and Labels
	df_x = df_data['Cleantext']
	df_y = df_data.Target
    # Extract Feature With CountVectorizer
	corpus = df_x
	cv = TfidfVectorizer()
	X = cv.fit_transform(corpus) # Fit the Data
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, df_y, test_size=0.30, random_state=42)
	#Naive Bayes Classifier
	#from sklearn.linear_model import LogisticRegression
	#clf = LogisticRegression()
	#clf.fit(X_train,y_train)
	#clf.score(X_test,y_test)
	#Alternative Usage of Saved Model
	model = open("Logistic_pkmodel.pkl","rb")
	clf = joblib.load(model)
	
	if request.method == 'POST':
		result={}
		comment = request.form['comment']
		data=comment.split('. ')
		for val in data:
			val1=[val]
			temp = compute(val1,cv,clf)
			result[val]=temp
	print(result)
	return render_template('result.html',prediction = result)


def compute(data,cv,clf):
	vect = cv.transform(data).toarray()
	my_prediction = clf.predict(vect)
	return my_prediction
if __name__ == '__main__':
	app.run(debug=True)

from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np



app = Flask(__name__)

model = load_model('final_iris_model.h5')

scaler = MinMaxScaler()


@app.route('/home', methods = ['GET','POST'])
def home():
	if request.method == "POST":
		s_len = float(request.form['Sepal Length'])
		s_wid = float(request.form['Sepal Width'])
		p_len = float(request.form['Petal Length'])
		p_wid = float(request.form['Petal width'])
		flower = [[s_len,s_wid,p_len,p_wid]]
		scaled_flower = scaler.fit_transform(flower)
		
		predict = model.predict(flower)
		y = np.argmax(predict, axis=1)[0]
		return render_template('result.html',result = y)
		
		#eturn redirect(url_for('prediction', flower = scaled_flower))

	else:
		return render_template('index.html')
		

#@app.route('/home/prediction/<flower>',methods = ['GET', 'POST'])
#def prediction(flower):
#	print(flower)
#	predicted_value = model.predict(flower)
#	
#	y = np.argmax(predict, axis=1)[0]
#	return render_template('result.html',result = y)


if __name__ == '__main__':
	app.run(debug = True)

from flask import Flask,render_template,request
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
app=Flask(__name__)
ritz=pickle.load(open("ritz.pkl",'rb'))
@app.route('/')
def hello_world():
    return render_template('front.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features)]
    prediction=ritz.predict_proba(final)
    output='{0:.{1}f}'.format(prediction[0][1],2)

    if(output>str(0.5)):
        return render_template('front.html',pred='Your Forest is in Danger.\n Probability of fire occuring is {}'.format(output))
    else:
        return render_template('front.html',pred='Your Forest is in safe.\n Probability of fire occuring is {}'.format(output))


if __name__=='__main__':
    app.run(debug=True)
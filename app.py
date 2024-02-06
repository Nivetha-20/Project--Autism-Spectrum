import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
  return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
  input_features = [float(x) for x in request.form.values()]
  features_value = [np.array(input_features)]

  features_name = ['age', 'gender', 'ethnicity', 'jundice', 'austim', 'contry_of_res',
       'used_app_before', 'result', 'relation']

  df = pd.DataFrame(features_value, columns=features_name)
  output = model.predict(df)

  if output == 0:
      res_val = "Found"
  elif output == 1:
      res_val = "Not Found"
  
  return render_template('index.html', prediction_text='Autism spectrum disorder {}'.format(res_val))

if __name__ == "__main__":
  app.run()

##host='0.0.0.0',debug=False, port = 4566

import joblib
import pandas as pd
import numpy as np
import pickle
import json

def combine_features():
	infile = open("Metric_0_36.pkl",'rb')
	df = pickle.load(infile)
	infile.close()
	
	for i in range(1, 7):
		filename = "Metric_"+str(i)+"_36.pkl"
		infile = open(filename, 'rb')
		x = pickle.load(infile)
		infile.close()

		df = df.join(x.set_index('t'), on='t')

	del df['t']
	with open("X.pkl", "wb") as outfile:
		pickle.dump(df, outfile)

	return df

def predict_alc_level():
    ##loading the model from the saved file
    pkl_filename = "alcohol_model.pkl"
    alcohol_model = joblib.load(pkl_filename)
    with open("X.pkl", "rb") as infile:
        df = pickle.load(infile)
    df = np.array(df)
    
    
    y_pred = alcohol_model.predict(df)
    
    return y_pred

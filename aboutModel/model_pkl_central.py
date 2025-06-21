model_list=[
    'decission_tree',
]
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
import joblib

for modelName in model_list:
    if modelName=='decission_tree':
        # load model
        model = joblib.load('saved_model/'+modelName+'.pkl')
        print(rfc2.predict(X[0:1,:]))

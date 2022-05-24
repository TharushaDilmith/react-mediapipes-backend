import pandas as pd
import pickle 
from sklearn.model_selection import train_test_split
from flask_restful import Resource

class TrainData(Resource):

    def get(self):
        df = pd.read_csv('dataset/coords.csv')

        x = df.drop('class', axis=1) #features
        y = df['class'] #target value

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1234)

        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression, RidgeClassifier
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

        pipelines = {
            'lr':make_pipeline(StandardScaler(), LogisticRegression()),
            'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
            'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),
            'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier()),
        }

        fit_models = {}
        for algo, pipeline in pipelines.items():
            model = pipeline.fit(x_train, y_train)
            fit_models[algo] = model

        from sklearn.metrics import accuracy_score
        import pickle

        for algo, model in fit_models.items():
            yhat = model.predict(x_test)
            print(algo, accuracy_score(y_test, yhat))
            
        with open('body_language.pkl', 'wb') as f:
            pickle.dump(fit_models['rf'], f)



        return 'success'


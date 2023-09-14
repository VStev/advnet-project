from xgboost import XGBClassifier

class GradientBoosting():
    model = any

    def create_model(self, inputs):
        n, o, r, d = inputs
        self.model = XGBClassifier(n_estimators = n, objective = o, learning_rate = r, max_depth = d)
    
    def train_model(self, trainset):
        data, label = trainset
        self.model.fit(data, label)

    def do_prediction(self, data):
        return self.model.predict(data)

    def __init__(self, inputs) -> None:
        self.create_model(inputs)
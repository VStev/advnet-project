from sklearn.ensemble import RandomForestClassifier as RF

class RandomForest():
    model = any

    def create_model(self):
        self.model = RF(
            n_estimators = 10,
            criterion = 'gini',
            max_depth = 100
        )
    
    def train_model(self, trainset):
        data, label = trainset
        self.model.fit(data, label)

    def do_prediction(self, data):
        return self.model.predict(data)
    
    def __init__(self) -> None:
        self.create_model()
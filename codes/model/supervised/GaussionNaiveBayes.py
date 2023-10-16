from sklearn.naive_bayes import GaussianNB as GNB

class GaussionNaiveBayes():
    model = any

    def create_model(self):
        self.model = GNB()
    
    def train_model(self, trainset):
        data, label = trainset
        self.model.fit(data, label)

    def do_prediction(self, data):
        return self.model.predict(data)
    
    def __init__(self, inputs) -> None:
        self.create_model(inputs)
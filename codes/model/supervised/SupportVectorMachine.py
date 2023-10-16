from sklearn.svm import SVC

class SupportVectorMachine():
    model = any

    def create_model(self):
        self.model = SVC(
            C = 1,
            kernel = 'linear',
            decision_function_shape = 'ovr',
            max_iter = 1000
        )
    
    def train_model(self, trainset):
        data, label = trainset
        self.model.fit(data, label)

    def do_prediction(self, data):
        return self.model.predict(data)
    
    def __init__(self, inputs) -> None:
        self.create_model(inputs)
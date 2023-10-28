from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

class LinearDiscriminantAnalysis():
    model = any

    def create_model(self):
        self.model = LDA(
            n_components = None,
            priors = None,
            shrinkage = None,
            solver = 'svd',
            store_covariance = False,
            tol = 0.0001
        )
    
    def train_model(self, trainset):
        data, label = trainset
        self.model.fit(data, label)

    def do_prediction(self, data):
        return self.model.predict(data)
    
    def __init__(self) -> None:
        self.create_model()
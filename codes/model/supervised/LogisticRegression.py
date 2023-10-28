from sklearn.linear_model import LogisticRegression as LR

class LogisticRegression():
    model = any

    def create_model(self):
        self.model = LR(
            penalty = 'l2',
            dual = False,
            tol = 0.0001,
            C = 1.0,
            fit_intercept = True,
            intercept_scaling = 1,
            solver = 'sag',
            max_iter = 100,
            multi_class = 'ovr',
            verbose = 0,
            warm_start = False,
            l1_ratio = None
        )
    
    def train_model(self, trainset):
        data, label = trainset
        self.model.fit(data, label)

    def do_prediction(self, data):
        return self.model.predict(data)
    
    def __init__(self) -> None:
        self.create_model()
from sklearn.linear_model import LogisticRegression as lr

class LogisticRegression():
    model = any

    def create_model(self, inputs):
        p, s, l1, r, i, m = inputs
        self.model = lr(
            penalty = p,
            solver = s,
            l1_ratio = l1,
            random_state = r,
            max_iter = i,
            multi_class = m
        )
    
    def train_model(self, trainset):
        data, label = trainset
        self.model.fit(data, label)

    def do_prediction(self, data):
        return self.model.predict(data)
    
    def __init__(self, inputs) -> None:
        self.create_model(inputs)
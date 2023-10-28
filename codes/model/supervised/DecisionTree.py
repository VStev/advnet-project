from sklearn.tree import DecisionTreeClassifier as DT

class DecisionTree():
    model = any

    def create_model(self):
        self.model = DT(
            criterion = 'gini',
            splitter = 'best',
            max_depth = 100,
            min_samples_split = 2,
            min_samples_leaf = 1
        )
    
    def train_model(self, trainset):
        data, label = trainset
        self.model.fit(data, label)

    def do_prediction(self, data):
        return self.model.predict(data)
    
    def __init__(self) -> None:
        self.create_model()
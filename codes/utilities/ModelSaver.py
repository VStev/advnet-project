import pickle

def save_model(model, filename):
    pickle.dump(model, open(f'${filename}.pkl', 'wb'))

def load_model(model, filename):
    pickled_model = pickle.load(open(f'${filename}.pkl', 'rb'))
    return pickled_model
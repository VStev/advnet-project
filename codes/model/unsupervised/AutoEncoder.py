import tensorflow as tf

class AutoEncoder():
    model = any

    def create_model(self, input_dim):
        input_layer = tf.keras.layers.Input(shape=(input_dim,))
        hidden_layer = tf.keras.layers.Dense(40, activation='relu')(input_layer)
        output_layer = tf.keras.layers.Dense(input_dim, activation='sigmoid')(hidden_layer)
        autoencoder = tf.keras.models.Model(input_layer, output_layer)
        # Compile the model
        model = autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    
    def train_model(self, data):
        self.model.fit(data, data, epochs=50, batch_size=32)

    def do_prediction(self, data):
        return self.model.predict(data)
    
    def __init__(self, input_dim) -> None:
        self.create_model(input_dim)
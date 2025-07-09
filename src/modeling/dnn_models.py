from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

def build_simple_dnn(input_dim, hidden_units=[64, 32], dropout_rate=0.3, learning_rate=0.001):
    """
    Build and compile a simple dense neural network for classification.
    """
    model = Sequential()
    model.add(Dense(hidden_units[0], input_dim=input_dim, activation='relu'))
    model.add(Dropout(dropout_rate))
    for units in hidden_units[1:]:
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))  # binary classification
    
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=learning_rate),
                  metrics=['accuracy'])
    return model

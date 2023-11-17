# LIBRARIES
from preprocess import generate_training_sequences, SEQUENCE_LENGTH
from tensorflow import keras


# CONSTANTS
OUTPUT_UNITS = 18 # Number of vocabulary (Check the mapping.json file)
NUM_UNITS = [256]
LOSS = 'sparse_categorical_crossentropy'
LEARNING_RATE = 0.001
EPOCHS = 30
BATCH_SIZE = 64

SAVE_MODEL_PATH = 'model.h5'

# FUNCTIONS

def build_model(output_units, num_untis, loss, learning_rate):
    
    # Create the model architecture
    input = keras.layers.Input(shape=(None, output_units))
    x = keras.layers.LSTM(num_untis[0])(input)
    x = keras.layers.Dropout(0.2)(x)

    output = keras.layers.Dense(output_units, activation='softmax')(x)

    model = keras.Model(input, output)

    # Compile the model
    model.compile(loss=loss, optimizer=keras.optimizers.Adam(learning_rate=learning_rate), metrics=['accuracy'])
    
    model.summary()

    return model




def train(output_units=OUTPUT_UNITS, num_untis=NUM_UNITS, loss=LOSS, learning_rate=LEARNING_RATE):

    # Generate training sequences
    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)

    # Build the network
    model = build_model(output_units, num_untis, loss, learning_rate)

    # Train the model
    model.fit(inputs, targets, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # Save the model
    model.save(SAVE_MODEL_PATH)

# MAIN
if __name__ == '__main__':
    train()
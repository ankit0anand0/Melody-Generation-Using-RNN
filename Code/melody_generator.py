from tensorflow import keras
import json
from preprocess import SEQUENCE_LENGTH, MAPPING_PATH
import numpy as np
import music21 as m21



class MelodyGenerator:

    def __init__(self, model_path='model.h5'):

        self.model_path = model_path
        self.model = keras.models.load_model(model_path)

        with open(MAPPING_PATH, 'r') as fp:
            self._mappings = json.load(fp)
        
        self._start_symbols = ['/'] * SEQUENCE_LENGTH
    
    def generate_melody(self, seed, num_steps, max_sequence_length, temparature):
        
        # Create seed with start symbol
        seed = seed.split()
        melody = seed
        seed = self._start_symbols + seed

        # Map seeds to int
        seed = [self._mappings[symbol] for symbol in seed]

        for _ in range(num_steps):

            # Limit the seed to the max sequence length
            seed = seed[-max_sequence_length:]

            # One-hot encode the seed
            onehot_seed = keras.utils.to_categorical(seed, num_classes=len(self._mappings))
            onehot_seed = onehot_seed[np.newaxis, ...]

            # Make prediction
            probabilities = self.model.predict(onehot_seed)[0]

            output_int = self._sample_with_temperature(probabilities, temparature)

            # Update seed
            seed.append(output_int)

            # Map int to our encoding
            output_symbol = [k for k, v in self._mappings.items() if v == output_int][0]

            # Check whether we are at the end of a melody
            if output_symbol == '/':
                break

            # Update the melody
            melody.append(output_symbol)
        
        return melody    
    
    def _sample_with_temperature(self, probabilities, temperature): 

        predictions = np.log(probabilities) / temperature
        probabilities = np.exp(predictions) / np.sum(np.exp(predictions))

        choices = range(len(probabilities))
        index = np.random.choice(choices, p=probabilities)

        return index

    def save_melody(self, melody, step_duration=0.25, format='midi', file_name='mel.mid'):

        # Create a music21 stream
        stream = m21.stream.Stream()

        # Parse all the symbols in the melody and create note/rest objects
        start_symbol = None
        step_counter = 1

        for i, symbol in enumerate(melody):

            # Handle case in which we have a note/rest
            if symbol != '_' or i+1 == len(melody):
                
                # Ensure we are dealing with note/rest beyond the first one
                if start_symbol is not None:
                    quarter_length_duration = step_duration * step_counter

                    # Handle rest
                    if start_symbol == 'r':
                        m21_event = m21.note.Rest(quarterLength=quarter_length_duration)

                    # Handle note
                    else:
                        m21_event =m21.note.Note(int(start_symbol), quarterLength=quarter_length_duration)
                    stream.append(m21_event)

                    # Reset the step counter
                    step_counter = 1
                start_symbol = symbol

            # Handle case in which we have a prolongation sign '_'
            else:
                step_counter += 1

        # write the m21 stream to midi file
        stream.write(format, file_name)
            

if __name__ == '__main__':
   
    mg = MelodyGenerator()
    seed = '57 _ 60 _ 57 _ _ _ 62 _'
    melody = mg.generate_melody(seed, 500, SEQUENCE_LENGTH, 0.3)
    print(melody)
    mg.save_melody(melody=melody)

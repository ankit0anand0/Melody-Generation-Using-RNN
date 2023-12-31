# LIBRARIES
import music21 as m21
import os
import json
from tensorflow import keras
import numpy as np

# CONSTANTS
KERN_DATASET_PATH = '/Users/ankitanand/Documents/Research/Projects/Music Generation RNN-LSTM/Dataset/deutschl/test'
ENCODED_DATASET_PATH = '/Users/ankitanand/Documents/Research/Projects/Music Generation RNN-LSTM/Encoded Dataset'
SINGLE_FILE_DATASET = 'file_dataset'
MAPPING_PATH = 'mapping.json'

SEQUENCE_LENGTH = 64

ACCEPTABLE_DURATIONS = [0.25, 0.50, 0.75, 1, 1.5, 2, 3, 4] # 0.75 -> dotted 8th note and 1.5 -> dotted quarter note

# FUNCTIONS

def load_songs_in_kern(dataset_path):

    songs = []

    # Go through all the path and load songs
    for path, subdirs, files in os.walk(dataset_path):
        for file in files:
            if file[-3:] == 'krn':
                song = m21.converter.parse(os.path.join(path, file))
                songs.append(song)

    return songs

def has_acceptable_durations(song, acceptable_durations):
    for note in song.flat.notesAndRests:
        if note.duration.quarterLength not in acceptable_durations:
            return False
    
    return True

def transpose(song):

    # Get key from a song
    parts = song.getElementsByClass(m21.stream.Part)
    measure_part0 = parts[0].getElementsByClass(m21.stream.Measure)
    key = measure_part0[0][4]

    # Estimate key using music21 incase it is not mentioned
    if not isinstance(key, m21.key.Key):
        key = song.analyze('key')

    # Get the interval for transposition
    if key.mode == 'major':
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch('C'))
    elif key.mode == 'minor':
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch('A'))

    # Transpose by the calculated interval
    transposed_song = song.transpose(interval)

    return transposed_song

def encode_song(song, time_step=0.25):

    encoded_song = []

    for event in song.flat.notesAndRests:

        # Handle notes
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi
        
        # Handle rests
        elif isinstance(event, m21.note.Rest):
            symbol = 'r'

        # Convert notes and rests into time series representation
        steps = int(event.duration.quarterLength / time_step)

        for step in range(steps):
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append('_')
    
    # Cast encoded songs to string
    encoded_song = ' '.join(map(str, encoded_song))

    return encoded_song



def preprocess(dataset_path):
    # Load the folk songs
    print('Loading song...')
    songs = load_songs_in_kern(KERN_DATASET_PATH)
    print(f'Loaded {len(songs)} songs.')

    for i, song in enumerate(songs):

        # Filter songs with non-acceptable durations
        if not has_acceptable_durations(song=song, acceptable_durations=ACCEPTABLE_DURATIONS):
            continue
        # Transpose to CM/Am
        song = transpose(song=song)

        # Encode songs with music time series representations
        encoded_song = encode_song(song)

        # Save songs to a text file
        save_path = os.path.join(ENCODED_DATASET_PATH,str(i))
        with open(save_path, 'w') as fp:
            fp.write(encoded_song)

def load(file_path):
    with open(file_path, 'r') as fp:
        song = fp.read()
    return song

def create_single_file_dataset(dataset_path, file_dataset_path, sequence_length):

    new_song_delimiter = '/ ' * sequence_length
    songs = ''

    # Load encoded songs and add delimiters
    for path, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(path, file)
            song = load(file_path)

            songs = songs + song + ' ' + new_song_delimiter
    songs = songs[:-1]

    # Save string that contains all dataset
    with open(file_dataset_path, 'w') as fp:
        fp.write(songs)
    
    return songs

def create_mappings(songs, mapping_path):

    mappings = {}

    # Identify vocabulary
    songs = songs.split()
    vocabulary = list(set(songs))

    # create mappings
    for i, symbol in enumerate(vocabulary):
        mappings[symbol] = i

    # Save vocabulary to a json file
    with open(mapping_path, 'w') as fp:
        json.dump(mappings, fp, indent=4)

def convert_songs_to_int(songs):
    
    int_songs = []

    # Load mappings
    with open(MAPPING_PATH, 'r') as fp:
        mappings = json.load(fp)  

    # Cast songs string to a list
    songs = songs.split()
    
    # Map songs to int
    for symbol in songs:
        int_songs.append(mappings[symbol])
    
    return int_songs

def generate_training_sequences(sequence_length):

    inputs = []
    targets = []

    # Load songs and map them to integers
    songs = load(SINGLE_FILE_DATASET)
    int_songs = convert_songs_to_int(songs=songs)

    # Generate the training sequences
    num_sequences = len(int_songs) - sequence_length
    for i in range(num_sequences):
        inputs.append(int_songs[i:i+sequence_length])
        targets.append(int_songs[i+sequence_length])

    # One-hot encode the sequences
    # Input : (# of sequences, sequence length, vocabulary_size(due to hot encoding))
    vocabulary_size = len(set(int_songs))
    inputs = keras.utils.to_categorical(inputs, num_classes=vocabulary_size)
    targets = np.array(targets)

    return inputs, targets

def main():
    preprocess(KERN_DATASET_PATH)
    songs = create_single_file_dataset(ENCODED_DATASET_PATH, SINGLE_FILE_DATASET, SEQUENCE_LENGTH)
    create_mappings(songs=songs, mapping_path=MAPPING_PATH)
    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)
    a = 1
    

# MAIN FUNCTION
if __name__ == '__main__':
    main()
    

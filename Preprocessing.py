import os ,pretty_midi
import numpy as np

def pianoRollGen(midi_obj):
    #Function to get the transpose of the piano roll matrix of a MIDI file
    piano_midi = midi_obj.instruments[0] #Gets Piano channels from the object 
    piano_roll = piano_midi.get_piano_roll(fs=2) #Generates empty piano roll from the MIDI data
    
    #Replaces the velocity of each note with a 1 to represent when its been pressed
    for row in piano_roll:
      for j in range(len(row)):
        if row[j] > 0: row[j] = 1 
      
    piano_roll = piano_roll[21:21+88].T #Restricts to only notes within the piano space and inverts columns and rows

    #Removes empty rows from piano roll
    rows_to_del =[]
    for i in range(len(piano_roll)):
      if all(n == 0 for n in piano_roll[i]):
        rows_to_del.append(i) 

    piano_roll = np.delete(piano_roll, rows_to_del, axis=0) 

    return piano_roll

def inputsTargetGen(data_folder, sequence_length=16, num_files=100):
    #Function to generate input and target sequences
    inputs = []
    targets = []

    for x in range(num_files):
      file_name = np.random.choice(os.listdir(data_folder)) #Choose files at random
      midi_obj = pretty_midi.PrettyMIDI(os.path.join(data_folder, file_name)) #Creates a pretty midi object with the selected file
      piano_roll = pianoRollGen(midi_obj) #Generates a piano roll out of the MIDI object
      
      #Creates inputs and targets to input into the neural network
      for i in range(len(piano_roll) - sequence_length):
        inputs.append([piano_roll[i:i + sequence_length]])
        targets.append(piano_roll[i + sequence_length]) 

    #Reshape inputs and targets into shape that can be taken as inputs
    inputs = np.reshape(inputs, (len(inputs), sequence_length, 88))
    targets = np.reshape(targets, (len(targets), 88))

    return inputs, targets

data_folder = "maestro-v2.0.0"

inputsTargets = inputsTargetGen(data_folder)
inputs = inputsTargets[0]
targets = inputsTargets[1]
import pretty_midi, numpy as np
from preprocessing import inputs
from keras.models import load_model 

def predictedPianoRoll(input_notes, model):
  #Function to generate piano roll using the trained network
  generated_piano_roll = []

  for i in range(360):
      model_input = np.reshape(input_notes ,(1, len(input_notes), 88)) #Reshape inputs the network
      model_output = model.predict(model_input) #Generate probability of each note being played using the trained network
       
      #Decide whether or not the note is played based on its probability
      for i, prob in enumerate(model_output.flatten()):
        rand = np.random.random()
        model_output[0][i] = 1 if prob > rand else 0

      generated_piano_roll.append(model_output)

      #Drop first row in inputs and append the newly generated row. This new array is fed back into the network
      input_notes = np.append(input_notes, model_output, axis=0)
      input_notes = input_notes[1:]

  generated_piano_roll = np.reshape(generated_piano_roll, (len(generated_piano_roll), 88))
  
  return generated_piano_roll.T

def pianoRollToMidi(input_notes, model, fs=2, program=0):
  #Function to turn piano roll into a MIDI file
  piano_roll = predictedPianoRoll(input_notes, model)
  notes, frames = piano_roll.shape
  pm = pretty_midi.PrettyMIDI()
  instrument = pretty_midi.Instrument(program=program)

  # pad 1 column of zeros so we can acknowledge inital and ending events
  piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

  # use changes in velocities to find note on / note off events
  velocity_changes = np.nonzero(np.diff(piano_roll).T)

  # keep track on velocities and note on times
  prev_notes_played = np.zeros(notes, dtype=int)
  note_on_time = np.zeros(notes)

  for time, note in zip(*velocity_changes):
    # use time + 1 because of padding above
    note_played = piano_roll[note, time + 1]
    time = time / fs
    if note_played:
      if prev_notes_played[note] == 0:
        note_on_time[note] = time
        prev_notes_played[note] = note_played
      else:
        pm_note = pretty_midi.Note(
        velocity=64,
        pitch=note + 21,
        start=note_on_time[note], end=time)
        instrument.notes.append(pm_note)
        prev_notes_played[note] = 0
  pm.instruments.append(instrument)
  return pm

model = load_model("model.hdf5") #Load in the pretrained model

start_index = np.random.randint(0, len(inputs) - 1)
input_notes = inputs[start_index]

new_midi_obj = pianoRollToMidi(input_notes, model)
new_midi_obj.write("GeneratedMusic.mid")
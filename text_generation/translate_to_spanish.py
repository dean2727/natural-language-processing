from tensorflow import keras
from keras.layers import Input, LSTM, Dense
from keras.models import Model
import numpy as np
import re
from preprocessing import input_docs, target_docs, input_tokens, target_tokens, num_encoder_tokens, num_decoder_tokens, max_encoder_seq_length, max_decoder_seq_length

# If we wanted to load in a trained keras model as an HDF5 model:
# from keras.models import load_model
# training_model = load_model('training_model.h5')
# encoder_inputs = training_model.input[0]
# encoder_outputs, state_h_enc, state_c_enc = training_model.layers[2].output
# encoder_states = [state_h_enc, state_c_enc]

input_features_dict = dict(
    [(token, i) for i, token in enumerate(input_tokens)])
target_features_dict = dict(
    [(token, i) for i, token in enumerate(target_tokens)])

reverse_input_features_dict = dict(
    (i, token) for token, i in input_features_dict.items())
reverse_target_features_dict = dict(
    (i, token) for token, i in target_features_dict.items())

encoder_input_data = np.zeros(
    (len(input_docs), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(input_docs), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_docs), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

for line, (input_doc, target_doc) in enumerate(zip(input_docs, target_docs)):

  for timestep, token in enumerate(re.findall(r"[\w']+|[^\s\w]", input_doc)):

    print("Encoder input timestep & token:", timestep, token)
    print(input_features_dict[token])
    # Assign 1. for the current line, timestep, & word
    # in encoder_input_data:
    encoder_input_data[line, timestep, input_features_dict[token]] = 1

  for timestep, token in enumerate(target_doc.split()):

    # decoder_target_data is ahead of decoder_input_data by one timestep
    print("Decoder input timestep & token:", timestep, token)
    # Assign 1. for the current line, timestep, & word
    # in decoder_input_data:
    decoder_input_data[line, timestep, target_features_dict[token]] = 1
    if timestep > 0:
      # decoder_target_data is ahead by 1 timestep
      # and doesn't include the start token.
      print("Decoder target timestep:", timestep)
      # Assign 1. for the current line, timestep, & word
      # in decoder_target_data:
      decoder_target_data[line, timestep - 1, target_features_dict[token]] = 1


# Training the encoder
# Create the input layer:
encoder_inputs = Input(shape=(None, num_encoder_tokens))

# Create the LSTM layer:
encoder_lstm = LSTM(256, return_state=True)

# Retrieve the outputs and states:
encoder_outputs, state_hidden, state_cell = encoder_lstm(encoder_inputs)

# Put the states together in a list:
encoder_states = [state_hidden, state_cell]


# Training the decoder
# The decoder input and LSTM layers:
decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(256, return_sequences=True, return_state=True)

# Retrieve the LSTM outputs and states:
decoder_outputs, decoder_state_hidden, decoder_state_cell = decoder_lstm(decoder_inputs, initial_state=encoder_states)

# Build a final Dense layer:
decoder_dense = Dense(num_decoder_tokens, activation='softmax')

# Filter outputs through the Dense layer:
decoder_outputs = decoder_dense(decoder_outputs)


# training the deep learning seq2seq model
training_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

print("Model summary:\n")
training_model.summary()
print("\n\n")

# Compile the model:
training_model = training_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Choose the batch size
# and number of epochs:
batch_size = 50
epochs = 50

print("Training the model:\n")
# Train the model:
training_model.fit(
  [encoder_input_data, decoder_input_data], 
  decoder_target_data,
  epochs=epochs,
  batch_size=batch_size,
  validation_split=0.2
)


# testing the model
# Building the encoder test model:
encoder_model = Model(encoder_inputs, encoder_states)

latent_dim = 256
# Building the two decoder state input layers:
decoder_state_input_hidden = Input(shape=(latent_dim,))

decoder_state_input_cell = Input(shape=(latent_dim,))

# Put the state input layers into a list:
decoder_states_inputs = [decoder_state_input_hidden, decoder_state_input_cell]

# Call the decoder LSTM:
decoder_outputs, state_hidden, state_cell = decoder_lstm(decoder_inputs, 
    initial_state=decoder_states_inputs)
decoder_states = [state_hidden, state_cell]

# Redefine the decoder outputs:
decoder_outputs = decoder_dense(decoder_outputs)

# Build the decoder test model:
decoder_model = Model(
  [decoder_inputs] + decoder_states_inputs,
  [decoder_outputs] + decoder_states
)


# model now ready for testing!
def decode_sequence(test_input):
  # run a test sentence (spanish) through the encoder model
  # predict() takes the sentence (matrix) and gives output states to pass to the decoder
  encoder_states_value = encoder_model.predict(test_input)
  decoder_states_value = encoder_states_value

  # empty 3D zeros array for spanish translation, with <START> set to 1
  target_seq = np.zeros((1, 1, num_decoder_tokens))
  target_seq[0, 0, target_features_dict['<START>']] = 1.

  # our return value, the decoded sentence
  decoded_sentence = ''
  
  # using output state from encoder, decode the sentence word-by-word
  stop_condition = False
  while not stop_condition:
    # Run the decoder model to get possible output tokens (with probabilities) & states
    output_tokens, new_decoder_hidden_state, new_decoder_cell_state = decoder_model.predict([target_seq] + decoder_states_value)
    
    # Choose token with highest probability (argmax())
    # slicing [0, -1, :] gives us  specific token vector within the 3d NumPy matrix
    sampled_token_index = np.argmax(output_tokens[0, -1, :])
    # using reverse feature dict to get the word from index
    sampled_token = reverse_target_features_dict[sampled_token_index]

    decoded_sentence += " " + sampled_token

    # Exit condition: either hit max length
    # or find stop token.
    if (sampled_token == '<END>' or len(decoded_sentence) > max_decoder_seq_length):
      stop_condition = True

    # Update the target sequence (of length 1).
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # this one-hot vector now represents the token we just sampled
    target_seq[0, 0, sampled_token_index] = 1

    # Update decoder hidden state so that previously decoded words can help in decoding new ones
    decoder_states_value = [
    new_decoder_hidden_state,
    new_decoder_cell_state]

  return decoded_sentence

for seq_index in range(10):
  test_input = encoder_input_data[seq_index: seq_index + 1]
  decoded_sentence = decode_sequence(test_input)
  print('-')
  print('Input sentence:', input_docs[seq_index])
  print('Decoded sentence:', decoded_sentence)
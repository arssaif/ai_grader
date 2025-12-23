import joblib
import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Embedding, Concatenate, BatchNormalization, Dropout, Add, GRU, AveragePooling2D
import pandas as pd
import numpy as np
import cv2
import re
from nltk.translate.bleu_score import sentence_bleu
from PIL import Image
tf.compat.v1.enable_eager_execution()
'''
ALL PATHS NEEDED
'''

chexnet_weights = 'static/models/captioning/captioning_CheXNet_Keras_0.3.0_weights.h5'
tokenizer = joblib.load('static/models/captioning/tokenizer.pkl')
model_filename = 'static/models/captioning/Encoder_Decoder_global_attention.h5'

def create_chexnet(chexnet_weights=chexnet_weights, input_size=(224, 224)):
    """
    chexnet_weights: weights value in .h5 format of chexnet
    creates a chexnet model with preloaded weights present in chexnet_weights file
    """
    model = tf.keras.applications.DenseNet121(include_top=False, input_shape=input_size + (
    3,))  # importing densenet the last layer will be a relu activation layer

    # we need to load the weights so setting the architecture of the model as same as the one of the chexnet
    x = model.output  # output from chexnet
    x = GlobalAveragePooling2D()(x)
    x = Dense(14, activation="sigmoid", name="chexnet_output")(
        x)  # here activation is sigmoid as seen in research paper

    chexnet = tf.keras.Model(inputs=model.input, outputs=x)
    chexnet.load_weights(chexnet_weights)
    chexnet = tf.keras.Model(inputs=model.input, outputs=chexnet.layers[
        -3].output)  # we will be taking the 3rd last layer (here it is layer before global avgpooling)
    # since we are using attention here
    return chexnet


class Image_encoder(tf.keras.layers.Layer):
    """
    This layer will output image backbone features after passing it through chexnet
    """

    def __init__(self,
                 name="image_encoder_block"
                 ):
        super().__init__()
        self.chexnet = create_chexnet(input_size=(224, 224))
        self.chexnet.trainable = False
        self.avgpool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))
        # for i in range(10): #the last 10 layers of chexnet will be trained
        #   self.chexnet.layers[-i].trainable = True

    def call(self, data):
        op = self.chexnet(data)  # op shape: (None,7,7,1024)
        op = self.avgpool(op)  # op shape (None,3,3,1024)
        op = tf.reshape(op, shape=(-1, op.shape[1] * op.shape[2], op.shape[3]))  # op shape: (None,9,1024)
        return op


def encoder(image1, dense_dim, dropout_rate):
    """
    Takes image1
    gets the final encoded vector of these
    """
    # image1
    im_encoder = Image_encoder()
    bkfeat1 = im_encoder(image1)  # shape: (None,9,1024)
    bk_dense = Dense(dense_dim, name='bkdense', activation='relu')  # shape: (None,9,512)
    bkfeat1 = bk_dense(bkfeat1)

    bn = BatchNormalization(name="encoder_batch_norm")(bkfeat1)
    dropout = Dropout(dropout_rate, name="encoder_dropout")(bn)
    return dropout


class global_attention(tf.keras.layers.Layer):
    """
    calculate global attention
    """

    def __init__(self, dense_dim):
        super().__init__()
        # Intialize variables needed for Concat score function here
        self.W1 = Dense(units=dense_dim)  # weight matrix of shape enc_units*dense_dim
        self.W2 = Dense(units=dense_dim)  # weight matrix of shape dec_units*dense_dim
        self.V = Dense(units=1)  # weight matrix of shape dense_dim*1
        # op (None,98,1)

    def call(self, encoder_output, decoder_h):  # here the encoded output will be the concatted image bk features shape: (None,98,dense_dim)
        decoder_h = tf.expand_dims(decoder_h, axis=1)  # shape: (None,1,dense_dim)
        tanh_input = self.W1(encoder_output) + self.W2(decoder_h)  # ouput_shape: batch_size*98*dense_dim
        tanh_output = tf.nn.tanh(tanh_input)
        attention_weights = tf.nn.softmax(self.V(tanh_output),
                                          axis=1)  # shape= batch_size*98*1 getting attention alphas
        op = attention_weights * encoder_output  # op_shape: batch_size*98*dense_dim  multiply all aplhas with corresponding context vector
        context_vector = tf.reduce_sum(op,
                                       axis=1)  # summing all context vector over the time period ie input length, output_shape: batch_size*dense_dim

        return context_vector, attention_weights


class One_Step_Decoder(tf.keras.layers.Layer):
    """
    Decodes a single token using GRUCell for manual stepping
    """

    def __init__(self, vocab_size, embedding_dim, max_pad, dense_dim, name="onestepdecoder"):
        super().__init__(name=name)
        self.dense_dim = dense_dim
        self.embedding = Embedding(input_dim=vocab_size+1,
                                   output_dim=embedding_dim,
                                   mask_zero=True,
                                   name='onestepdecoder_embedding')
        
        # FIX: Use GRUCell instead of GRU Layer. 
        # Cells are designed for single-step processing.
        self.gru_cell = tf.keras.layers.GRUCell(units=self.dense_dim, name='onestepdecoder_gru_cell')
        
        self.attention = global_attention(dense_dim=dense_dim)
        self.concat = Concatenate(axis=-1)
        self.final = Dense(vocab_size + 1, activation='softmax')

    @tf.function
    def call(self, input_to_decoder, encoder_output, decoder_h):
        '''
        input_to_decoder: (batch_size, 1)
        encoder_output: (batch_size, num_patches, dense_dim)
        decoder_h: (batch_size, dense_dim)
        '''
        
        # A. Embedding
        embedding_op = self.embedding(input_to_decoder) # shape: (batch_size, 1, embedding_dim)
        # Squeeze the time dimension for the Cell (Cells expect 2D input: batch, features)
        embedding_op = tf.squeeze(embedding_op, axis=1) # shape: (batch_size, embedding_dim)

        # B. Attention
        context_vector, attention_weights = self.attention(encoder_output, decoder_h)
        # context_vector shape: (batch_size, dense_dim)

        # C. Concat
        # We concatenate along the feature axis. 
        # Note: We do NOT need to expand dims for time axis because GRUCell takes 2D input.
        concat_input = self.concat([context_vector, embedding_op]) 
        # shape: (batch_size, dense_dim + embedding_dim)

        # D. LSTM/GRU Processing
        # GRUCell returns: (output, [new_state])
        output, states = self.gru_cell(concat_input, states=[decoder_h])
        
        # Update the hidden state
        decoder_h = states[0]

        # E. Final Dense Layer
        output = self.final(output) # shape: (batch_size, vocab_size)

        return output, decoder_h, attention_weights

class decoder(tf.keras.Model):
    """
    Decodes the encoder output and caption
    """

    def __init__(self, max_pad, embedding_dim, dense_dim, batch_size, vocab_size):
        super().__init__()
        self.onestepdecoder = One_Step_Decoder(vocab_size=vocab_size, embedding_dim=embedding_dim, max_pad=max_pad,
                                               dense_dim=dense_dim)
        self.output_array = tf.TensorArray(tf.float32, size=max_pad)
        self.max_pad = max_pad
        self.batch_size = batch_size
        self.dense_dim = dense_dim

    @tf.function
    def call(self, encoder_output,
             caption):  # ,decoder_h,decoder_c): #caption : (None,max_pad), encoder_output: (None,dense_dim)
        decoder_h, decoder_c = tf.zeros_like(encoder_output[:, 0]), tf.zeros_like(
            encoder_output[:, 0])  # decoder_h, decoder_c
        output_array = tf.TensorArray(tf.float32, size=self.max_pad)
        for timestep in range(self.max_pad):  # iterating through all timesteps ie through max_pad
            output, decoder_h, attention_weights = self.onestepdecoder(caption[:, timestep:timestep + 1],
                                                                       encoder_output, decoder_h)
            output_array = output_array.write(timestep, output)  # timestep*batch_size*vocab_size

        self.output_array = tf.transpose(output_array.stack(), [1, 0,
                                                                2])  # .stack :Return the values in the TensorArray as a stacked Tensor.)
        # shape output_array: (batch_size,max_pad,vocab_size)
        return self.output_array


def create_model():
    """
    creates the best model ie the attention model
    and returns the model after loading the weights
    and also the tokenizer
    """
    # hyperparameters
    input_size = (224, 224)
    max_pad = 29
    batch_size = 100
    vocab_size = len(tokenizer.word_index)
    print('vocab_size: ',vocab_size)
    embedding_dim = 300
    dense_dim = 512
    dropout_rate = 0.2

    tf.keras.backend.clear_session()
    image1 = Input(shape=(input_size + (3,)))  # shape = 224,224,3
    caption = Input(shape=(max_pad,))

    encoder_output = encoder(image1, dense_dim, dropout_rate)  # shape: (None,28,512)

    output = decoder(max_pad, embedding_dim, dense_dim, batch_size, vocab_size)(encoder_output, caption)
    model = tf.keras.Model(inputs=[image1, caption], outputs=output)
    model_save = model_filename
    model.load_weights(model_save)

    return model, tokenizer


def greedy_search_predict(image1, model, tokenizer, input_size=(224, 224)):
    """
    Given paths to two x-ray images predicts the impression part of the x-ray in a greedy search algorithm
    """
    image1 = tf.expand_dims(cv2.resize(image1, input_size, interpolation=cv2.INTER_NEAREST), axis=0)
    image1 = model.get_layer('image_encoder')(image1)
    image1 = model.get_layer('bkdense')(image1)

    enc_op = model.get_layer('encoder_batch_norm')(image1)
    enc_op = model.get_layer('encoder_dropout')(enc_op)  # this is the output from encoder

    decoder_h, decoder_c = tf.zeros_like(enc_op[:, 0]), tf.zeros_like(enc_op[:, 0])
    a = []
    max_pad = 29
    for i in range(max_pad):
        if i == 0:  # if first word
            caption = np.array(tokenizer.texts_to_sequences(['<cls>']))  # shape: (1,1)
        output, decoder_h, attention_weights = model.get_layer('decoder').onestepdecoder(caption, enc_op, decoder_h)  # ,decoder_c) decoder_c,

        # prediction
        max_prob = tf.argmax(output, axis=-1)  # tf.Tensor of shape = (1,1)
        caption = np.array([max_prob])  # will be sent to onstepdecoder for next iteration
        if max_prob == np.squeeze(tokenizer.texts_to_sequences(['<end>'])):
            break
        else:
            a.append(tf.squeeze(max_prob).numpy())
    return tokenizer.sequences_to_texts([a])[0]  # here output would be 1,1 so subscripting to open the array

def predict1(image1,  model_tokenizer=None):
    """given image1 and image 2 filepaths returns the predicted caption,
    the model_tokenizer will contain stored model_weights and tokenizer
    """

    if model_tokenizer == None:
        model, tokenizer = create_model()
    else:
        model, tokenizer = model_tokenizer[0], model_tokenizer[1]
    predicted_caption = greedy_search_predict(image1, model, tokenizer)

    return predicted_caption

def predict2(true_caption, image1,image2=None,model_tokenizer = None):
  """given image1 and image 2 filepaths and the true_caption
   returns the mean of cumulative ngram bleu scores where n=1,2,3,4,
  the model_tokenizer will contain stored model_weights and tokenizer
  """
  if image2 == None: #if only 1 image file is given
    image2 = image1

  try:
    image1 = cv2.imread(image1,cv2.IMREAD_UNCHANGED)/255
    image2 = cv2.imread(image2,cv2.IMREAD_UNCHANGED)/255
  except:
    return print("Must be an image")

  if model_tokenizer == None:
    model,tokenizer = create_model()
  else:
    model,tokenizer = model_tokenizer[0],model_tokenizer[1]
  predicted_caption = greedy_search_predict(image1,image2,model,tokenizer)

  _ = get_bleu(true_caption,predicted_caption)
  _ = list(_)
  return pd.DataFrame([_],columns = ['bleu1','bleu2','bleu3','bleu4'])

def function1(image1, model_tokenizer=None):
    """
    here image1 will be a list of image
    filepaths and outputs the resulting captions as a list
    """
    if model_tokenizer is None:
        model_tokenizer = list(create_model())
    predicted_caption = []
    caption = predict1(image1, model_tokenizer)
    predicted_caption.append(caption)

    return predicted_caption

def predict(image_1,model_tokenizer):
    if (image_1 is not None):
        image_1 = Image.open(image_1).convert("RGB") #converting to 3 channels
        image_1 = np.array(image_1)/255

        caption = function1(image_1,model_tokenizer)
        del image_1
        return caption
    else:
        print("No Image")

def caption_g(p1, user_id):
    """
    Generates and formats a caption for a specific patient image.
    """
    model_tokenizer = create_model()
    test_im ='static/Patient_images/'+ p1   # atelectasis| effusion| emphysema| infiltration
    c = predict(test_im, model_tokenizer)
    caption = '. '.join(map(lambda s: s.strip().capitalize(), c[0].split('.')))
    caption = re.split("<", caption)
    #print(caption[0])
    return caption[0]
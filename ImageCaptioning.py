# Importing various libraries
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
import numpy as np
from keras.optimizers import Adam
from keras.layers import Dense, Flatten,Input, Convolution2D, Add, Dropout, LSTM, Activation,TimeDistributed,Conv2D, Embedding, Bidirectional, Activation, RepeatVector,Concatenate,ZeroPadding2D,BatchNormalization,AveragePooling2D,MaxPooling2D,GlobalMaxPooling2D
from keras.models import Sequential, Model, load_model
from keras.utils import np_utils,layer_utils
from keras.initializers import glorot_uniform
import random
from keras.preprocessing import image, sequence
import matplotlib.pyplot as plt
import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)
from keras import layers
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from IPython.display import Image, display
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
import scipy.misc
from tensorflow.python.framework import ops
import tensorflow as tf


#Loading the files from Google colab

from google.colab import drive
drive.mount('/content/drive')

#File Paths
imagepath = '/content/drive/My Drive/Flickr_Data/Images/'
captionpath = '/content/drive/My Drive/Flickr_Data/Flickr_TextData/Flickr8k.token.txt'
trainpath = '/content/drive/My Drive/Flickr_Data/Flickr_TextData/Flickr_8k.trainImages.txt'
validationpath = '/content/drive/My Drive/Flickr_Data/Flickr_TextData/Flickr_8k.devImages.txt'
testpath = '/content/drive/My Drive/Flickr_Data/Flickr_TextData/Flickr_8k.testImages.txt'

caption = open(captionpath, 'r').read().split("\n")
Xtrain = open(trainpath, 'r').read().split("\n")
Xval = open(validationpath, 'r').read().split("\n")
Xtest = open(testpath, 'r').read().split("\n")

token = {}

for cap in range(len(caption)-1):
    temp = caption[cap].split("#")
    if temp[0] in token:
        token[temp[0]].append(temp[1][2:])
    else:
        token[temp[0]] = [temp[1][2:]]

traindataset = open('flickr_8k_train_dataset.txt','wb')
traindataset.write(b"image_id\tcaptions\n")

valdataset = open('flickr_8k_val_dataset.txt','wb')
valdataset.write(b"image_id\tcaptions\n")

testdataset = open('flickr_8k_test_dataset.txt','wb')
testdataset.write(b"image_id\tcaptions\n")

for img in Xtrain:
    if img == '':
        continue
    for capt in token[img]:
        caption = "<start> "+ capt + " <end>"
        traindataset.write((img+"\t"+caption+"\n").encode())
        traindataset.flush()
traindataset.close()

for img in Xtest:
    if img == '':
        continue
    for capt in token[img]:
        caption = "<start> "+ capt + " <end>"
        testdataset.write((img+"\t"+caption+"\n").encode())
        testdataset.flush()
testdataset.close()

for img in Xval:
    if img == '':
        continue
    for capt in token[img]:
        caption = "<start> "+ capt + " <end>"
        valdataset.write((img+"\t"+caption+"\n").encode())
        valdataset.flush()
valdataset.close()

#Writing Identityblock
def identityblock(X_val, f_val, filters, stages, blocks):

    conv_name_base = 'res' + str(stages) + blocks + '_branch'
    bn_name_base = 'bn' + str(stages) + blocks + '_branch'
    
    F1, F2, F3 = filters
    
    X_short = X_val
    
    X_val = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X_val)
    X_val = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X_val)
    X_val = Activation('relu')(X_val)
    
    X_val = Conv2D(F2,kernel_size=(f_val,f_val),strides=(1,1),padding='same',kernel_initializer = glorot_uniform(seed=0))(X_val)
    X_val = BatchNormalization(axis=-1)(X_val)
    X_val = Activation('relu')(X_val)

    X_val = Conv2D(F3,(1,1),strides=(1,1),padding='valid',kernel_initializer = glorot_uniform(seed=0))(X_val)
    X_val = BatchNormalization(axis=-1)(X_val)

    X_val = Add()([X_val,X_short])
    X_val = Activation('relu')(X_val)

    return X_val

tf.reset_default_graph()

with tf.Session() as test:
    np.random.seed(1)
    Aprev = tf.placeholder("float", [3, 4, 4, 6])
    X = np.random.randn(3, 4, 4, 6)
    A = identityblock(Aprev, f_val = 2, filters = [2, 4, 6], stages = 1, blocks = 'a')
    test.run(tf.global_variables_initializer())
    out = test.run([A], feed_dict={Aprev: X, K.learning_phase(): 0})

def convolutionalblock(X_val, f_val, filters, stages, blocks, st = 2):
    conv_name_base = 'res' + str(stages) + blocks + '_branch'
    bn_name_base = 'bn' + str(stages) + blocks + '_branch'
    
    F1, F2, F3 = filters
    
    X_short = X_val

    X_val = Conv2D(filters =F1, kernel_size =(1, 1), strides = (st,st), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X_val)
    X_val = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X_val)
    X_val = Activation('relu')(X_val)

    X_val= Conv2D(filters =F2, kernel_size =(f_val, f_val), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X_val)
    X_val = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X_val)
    X_val = Activation('relu')(X_val)

    X_val = Conv2D(filters =F3, kernel_size =(1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X_val)
    X_val = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X_val)

    X_short = Conv2D(filters =F3, kernel_size =(1, 1), strides = (st,st), padding = 'valid', name = conv_name_base + '1', kernel_initializer = glorot_uniform(seed=0))(X_short)
    X_short = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_short)

    X_val = Add()([X_val, X_short])
    X_val = Activation('relu')(X_val)

    return X_val

tf.reset_default_graph()

with tf.Session() as test:
    np.random.seed(1)
    Aprev = tf.placeholder("float", [3, 4, 4, 6])
    X = np.random.randn(3, 4, 4, 6)
    A = convolutionalblock(Aprev, f_val = 2, filters = [2, 4, 6], stages = 1, blocks = 'a')
    test.run(tf.global_variables_initializer())
    out = test.run([A], feed_dict={Aprev: X, K.learning_phase(): 0})

def ImageCaptioning(inputshape = (224,224, 3), classes = 2048):

    Xinput = Input(inputshape)

    Xval = ZeroPadding2D((3, 3))(Xinput)
    
    Xval = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(Xval)
    Xval = BatchNormalization(axis = 3, name = 'bn_conv1')(Xval)
    Xval = Activation('relu')(Xval)
    Xval = MaxPooling2D((3, 3), strides=(2, 2))(Xval)

    Xval = convolutionalblock(Xval, f_val = 3, filters = [64, 64, 256], stages = 2, blocks='a', st = 1)
    Xval = identityblock(Xval, 3, [64, 64, 256], stages=2, blocks='b')
    Xval = identityblock(Xval, 3, [64, 64, 256], stages=2, blocks='c')

    Xval = convolutionalblock(Xval, f_val = 3, filters = [128, 128, 512], stages = 3, blocks='a', st = 2)
    Xval = identityblock(Xval, 3, [128, 128, 512], stages=3, blocks='b')
    Xval = identityblock(Xval, 3, [128, 128, 512], stages=3, blocks='c')
    Xval = identityblock(Xval, 3, [128, 128, 512], stages=3, blocks='d')

    Xval = convolutionalblock(Xval, f_val = 3, filters = [256, 256, 1024], stages = 4, blocks='a', st = 2)
    Xval = identityblock(Xval, 3, [256, 256, 1024], stages=4, blocks='b')
    Xval = identityblock(Xval, 3, [256, 256, 1024], stages=4, blocks='c')
    Xval = identityblock(Xval, 3, [256, 256, 1024], stages=4, blocks='d')
    Xval = identityblock(Xval, 3, [256, 256, 1024], stages=4, blocks='e')
    Xval = identityblock(Xval, 3, [256, 256, 1024], stages=4, blocks='f')

    Xval = convolutionalblock(Xval, f_val = 3, filters = [512, 512, 2048], stages = 5, blocks='a', st = 2)
    Xval = identityblock(Xval, 3, [512, 512, 2048], stages=5, blocks='b')
    Xval = identityblock(Xval, 3, [512, 512, 2048], stages=5, blocks='c')

    Xval = AveragePooling2D(pool_size=(2, 2), name = 'avg_pool')(Xval)
    
    Xval = Flatten()(Xval)
    Xval = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(Xval)
    
    model = Model(inputs = Xinput, outputs = Xval, name='ImageCaptioning')

    return model

model = ImageCaptioning(inputshape = (224, 224, 3), classes = 2048)

def pre_processing(img_path):
    im = image.load_img(img_path, target_size=(224,224,3))
    im = image.img_to_array(im)
    im = np.expand_dims(im, axis=0)
    return im

training_data = {}
ct=0
for x in Xtrain:
    if x == "":
        continue
    if ct >= 6000:
        break
    ct+=1
    impath = imagepath + x
    img = pre_processing(impath)
    prediction = model.predict(img).reshape(2048)
    training_data[x] = prediction

pddataset = pd.read_csv("flickr_8k_train_dataset.txt", delimiter='\t')
ds = pddataset.values
sentence = []
for vals in range(ds.shape[0]):
    sentence.append(ds[vals, 1])

words_list = [i.split() for i in sentence]
unique_list = []
for i in words_list:
    unique_list.extend(i)
unique_list = list(set(unique_list))

word_2indices = {val:index for index, val in enumerate(unique_list)}
indices_2word = {index:val for index, val in enumerate(unique_list)}

word_2indices['UNK'] = 0
word_2indices['raining'] = 8253

indices_2word[0] = 'UNK'
indices_2word[8253] = 'raining'

vocab_size = len(word_2indices.keys())

max_len = 0

for vals in sentence:
    vals = vals.split()
    if len(vals) > max_len:
        max_len = len(vals)

padded_sequence, subsequent_word = [], []

for ex in range(ds.shape[0]):
    partial_seq = []
    next_word = []
    text = ds[ex, 1].split()
    text = [word_2indices[i] for i in text]
    for i in range(1, len(text)):
        partial_seq.append(text[:i])
        next_word.append(text[i])
    padded_partial_seq = sequence.pad_sequences(partial_seq, max_len, padding='post')

    next_wordshot = np.zeros([len(next_word), vocab_size], dtype=np.bool)
    
    for i,nextword in enumerate(next_word):
        next_wordshot[i, nextword] = 1
        
    padded_sequence.append(padded_partial_seq)
    subsequent_word.append(next_wordshot)
    
padded_sequence = np.asarray(padded_sequence)
subsequent_word = np.asarray(subsequent_word)

num_of_images = 2000

captions = np.zeros([0, max_len])
nextwords = np.zeros([0, vocab_size])

for x in range(num_of_images):
    captions = np.concatenate([captions, padded_sequence[x]])
    nextwords = np.concatenate([nextwords, subsequent_word[x]])

np.save("captions.npy", captions)
np.save("nextwords.npy", nextwords)

with open('/content/drive/My Drive/train_encoded_images.p', 'rb') as f:
    encodedimages = pickle.load(f, encoding="bytes")

imgs_list = []

for x in range(ds.shape[0]):
    if ds[x, 0].encode() in encodedimages.keys():
        imgs_list.append(list(encodedimages[ds[x, 0].encode()]))

imgs_list = np.asarray(imgs_list)

images = []

for x in range(num_of_images):
    for y in range(padded_sequence[x].shape[0]):
        images.append(imgs_list[x])
        
images = np.asarray(images)

np.save("images.npy", images)

imagenames = []

for x in range(num_of_images):
    for iy in range(padded_sequence[x].shape[0]):
        imagenames.append(ds[x, 0])
        
imagenames = np.asarray(imagenames)

np.save("imagenames.npy", imagenames)

captions = np.load("captions.npy")
nextwords = np.load("nextwords.npy")

images = np.load("images.npy")

imag = np.load("imagenames.npy")
        
embedding_size = 128
max_len = 40

imagemodel = Sequential()

imagemodel.add(Dense(embedding_size, input_shape=(2048,), activation='relu'))
imagemodel.add(RepeatVector(max_len))

imagemodel.summary()

languagemodel = Sequential()

languagemodel.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_len))
languagemodel.add(LSTM(256, return_sequences=True))
languagemodel.add(TimeDistributed(Dense(embedding_size)))

languagemodel.summary()

concat = Concatenate()([imagemodel.output, languagemodel.output])
x = LSTM(128, return_sequences=True)(concat)
x = LSTM(512, return_sequences=False)(x)
x = Dense(vocab_size)(x)
out = Activation('softmax')(x)
model = Model(inputs=[imagemodel.input, languagemodel.input], outputs = out)


model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
model.summary()

hist = model.fit([images, captions], nextwords, epochs=5,batch_size=512)

model.save_weights("model_weights.h5")

def getencoding(model, img):
    image = pre_processing(img)
    pred = model.predict(image).reshape(2048)
    return pred

resnet = ImageCaptioning(inputshape=(224,224,3))

def predictcaptions(image):
    start_word = ["<start>"]
    while True:
        par_caps = [word_2indices[i] for i in start_word]
        par_caps = sequence.pad_sequences([par_caps], maxlen=max_len, padding='post')
        preds = model.predict([np.array([image]), np.array(par_caps)])
        word_pred = indices_2word[np.argmax(preds[0])]
        start_word.append(word_pred)
        
        if word_pred == "<end>" or len(start_word) > max_len:
            break
            
    return ' '.join(start_word[1:-1])
for filename in os.listdir(imagepath):
  img = imagepath + filename

  test_img = getencoding(resnet, img)

  Argmax_Search = predictcaptions(test_img)

  z = Image(filename=img)
  display(z)
  print(Argmax_Search)


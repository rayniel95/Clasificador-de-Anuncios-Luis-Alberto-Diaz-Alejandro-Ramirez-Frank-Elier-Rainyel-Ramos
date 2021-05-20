import json
import os
import pickle
import random

import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.utils import Sequence

from feature_vecs import compute_input, compute_output


class Generator:
	def __init__(self, dsp, vocab, categories, max_sequence, batch_size):
		self._dsp = dsp
		self._v = vocab
		self._c = categories
		self._ms = max_sequence
		self._batch_size = batch_size

	def getGenerators(self):
		ds = os.listdir(self._dsp)
		shuffle = []
		c = len(ds) - 1
		while c >= 0:
			i = random.randint(0, c)
			shuffle.append(ds.pop(i))
			c -= 1

		shuffle = [os.path.join(self._dsp, w) for w in shuffle]

		trainingGenerator = DataGenerator(shuffle[: -5000], self._v, self._c, self._ms, self._batch_size)
		validationGenerator = DataGenerator(shuffle[-5000:], self._v, self._c, self._ms, self._batch_size)
		return trainingGenerator, validationGenerator

class  DataGenerator(Sequence):
	def __init__(self, data_path, vocab, categories, max_sequence, batch_size):
		self._dp = data_path
		self._v = vocab
		self._c = categories
		self._ms = max_sequence

		self._batch_size = batch_size

	def __len__(self):
		return int(np.ceil(len(self._dp) / float(self._batch_size)))

	def __getitem__(self, index):
		batch = self._dp[index * self._batch_size: (index + 1) * self._batch_size]
		inputs = []
		outputs = []
		for p in batch:
			with open(p) as fd:
				js = json.load(fd)
			inputs.append(compute_input(js['text'], self._v, self._ms))
			outputs.append(compute_output(js['category'], self._c))
		
		return np.array(inputs), np.array(outputs)

def cnn(input_size, vocab_size, MAX_SEQUENCE_LENGTH, pretrained_weights = None, t = 0):
	inputs = Input(input_size)
	embb1 = Embedding(vocab_size, 32, input_length = MAX_SEQUENCE_LENGTH)(inputs)

	convs = []
	if t:
		filters = [2,3,4]
	else:
		filters = [2,3,4,5]
	for fsz in filters:
		conv = Conv1D(64, fsz, activation='relu')(embb1)
		pool = MaxPooling1D(5)(conv)
		pool = Dropout(0.3)(pool)
		convs.append(pool)
	concat1 = Concatenate(axis=1)(convs)

	flat1 = Flatten()(concat1)

	dense1 = Dense(50, activation='relu')(flat1)
	dense1 = Dropout(0.3)(dense1)
	dense2 = Dense(41, activation='softmax')(dense1)

	model = Model(inputs = [inputs], outputs = [dense2])

	model.compile(optimizer = Adam(lr = 1e-4), loss = 'categorical_crossentropy', metrics = ['accuracy'])
	
	model.summary()

	if(pretrained_weights):
		model.load_weights(pretrained_weights)

	return model

def rnn(input_size, vocab_size, MAX_SEQUENCE_LENGTH, pretrained_weights = None):
	inputs = Input(input_size)
	embb1 = Embedding(vocab_size, 32, input_length = MAX_SEQUENCE_LENGTH)(inputs)
	bidir1 = Bidirectional(LSTM(32))(embb1)
	dense1 = Dense(50, activation='relu')(bidir1)
	dense2 = Dense(41, activation='softmax')(dense1)

	model = Model(inputs = [inputs], outputs = [dense2])

	model.compile(optimizer = Adam(lr = 1e-4), loss = 'categorical_crossentropy', metrics = ['accuracy'])
	
	model.summary()

	if(pretrained_weights):
		model.load_weights(pretrained_weights)

	return model

BATCHSIZE = 32

if __name__ == '__main__':
	f = open('./dataset_properties', 'rb')
	p = pickle.load(f)
	f.close()

	v = list(p['vocab'])
	c = list(p['categories'])

	v.sort()
	c.sort()

	v = { w : i + 1 for i, w in  enumerate(v)}
	c = { w : i for i, w in enumerate(c)}
	ms = p['max_sequence']
	ms = 10 * int(ms / 10) + 10

	model = rnn((ms,), len(v) + 1, ms, pretrained_weights = './weights.04-2.93.hdf5')

	gen = Generator('/home/luis/Desktop/SI/dataset', v, c, ms, BATCHSIZE)
	t_gen, v_gen = gen.getGenerators()
	
	model_checkpoint = ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='loss',verbose=1, save_best_only=True, save_weights_only=True)
	history = model.fit_generator(t_gen, epochs = 30, shuffle=True, callbacks=[model_checkpoint], validation_data=v_gen)
	with open('./trainHistoryDict', 'wb') as fd:
	    pickle.dump(history.history, fd)

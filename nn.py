from typing import List

import joblib
from keras import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense, Flatten
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
from sklearn.metrics import precision_score, f1_score

import utils




train_text = [info['text'] for info in joblib.load('train_set_cleaned3')]
test_text = [info['text'] for info in joblib.load('test_set_cleaned3')]

train_cats = [info['category'] for info in joblib.load('train_set_cleaned3')]
test_cats = [info['category'] for info in joblib.load('test_set_cleaned3')]


all_words = set()
for text in train_text:
	for word in text:
		all_words.add(word)
for text in test_text:
	for word in text:
		all_words.add(word)

all_tags = set()
for tag in train_cats:
	all_tags.add(tag)

for tag in test_cats:
	all_tags.add(tag)

n_cats = len(all_tags)
n_words = len(all_words)

max_len = max(max(len(text) for text in train_text),
			  max(len(text) for text in test_text))

word2dix = {w: i + 1 for i, w in enumerate(all_words)}
tag2dix = {t: i for i, t in enumerate(all_tags)}

joblib.dump(word2dix, 'word2index_cleaned3')
joblib.dump(tag2dix, 'tag2index_cleaned3')

x = [[word2dix[w] for w in text] for text in train_text]
x = pad_sequences(truncating='post', maxlen=max_len, sequences=x,
				  padding='post', value=0)

y = [tag2dix[cat] for cat in train_cats]

y = to_categorical(y, num_classes=n_cats)


x_test = [[word2dix[w] for w in text] for text in test_text]
x_test = pad_sequences(truncating='post', maxlen=max_len, sequences=x_test,
				  padding='post', value=0)

y_test = [tag2dix[cat] for cat in test_cats]

y_test = to_categorical(y_test, num_classes=n_cats)

model = Sequential()
model.add(Embedding(input_dim=n_words + 1, output_dim=5,
					input_length=max_len))
model.add(Bidirectional(LSTM(units=10, return_sequences=True,
							   recurrent_dropout=0.1)))
model.add(Flatten())
model.add(Dense(units=n_cats, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

history = model.fit(x, y, epochs=15, use_multiprocessing=True, workers=8,
					batch_size=100)

joblib.dump(model, 'nn_cleaned3')
joblib.dump(history, 'history_cleaned3')

pred = model.predict(x_test)

print(f1_score(y_test, pred, average='weighted'))



# model.compile('rmsprop', 'mse')
#
# train_embeddings = model.predict(x)
# means = train_embeddings.mean(1)


def test_nn(text: List[str]) -> str:
	word2index = joblib.load('word2index')
	new_sent = [word2dix[word] for word in text]

	x = pad_sequences(truncating='post', maxlen=max_len, sequences=new_sent,
					  padding='post', value=0)

	model = joblib.load('nn')
	cats = model.predict(x)

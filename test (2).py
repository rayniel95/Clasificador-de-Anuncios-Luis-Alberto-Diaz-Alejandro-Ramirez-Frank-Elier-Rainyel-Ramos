
import pickle
import matplotlib.pyplot as plt

# from nn import cnn
# from utils import preprocessing
# from feature_vecs import compute_input, compute_output, read_output
# import numpy as np

# f = open('./dataset_properties', 'rb')
# p = pickle.load(f)
# f.close()

# v = list(p['vocab'])
# c = list(p['categories'])

# v.sort()
# c.sort()

# v = { w : i + 1 for i, w in  enumerate(v)}
# c = { w : i for i, w in enumerate(c)}
# ms = p['max_sequence']
# ms = 10 * int(ms / 10) + 10

# model = cnn((ms,), len(v) + 1, ms, pretrained_weights='./weights.30-0.10.hdf5')

# x = 'vendo laptop'
# x = preprocessing(x)
# x = compute_input(x, v, ms)
# x = np.reshape(x, (1, 5430))


# y = model.predict_on_batch(x)

# print(read_output(y, c))

f = open('./cnn_v1/trainHistoryDict', 'rb')

h = pickle.load(f)
f.close()

plt.plot(h['loss'])
plt.plot(h['val_loss'])
plt.plot(h['accuracy'])
plt.plot(h['val_accuracy'])
plt.xlabel("Epochs")
plt.legend(['loss', 'val_loss', 'accuracy', 'val_accuracy'])
plt.show()
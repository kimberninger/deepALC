import csv, re, os, sys
import numpy as np
import matplotlib
from matplotlib2tikz import save as tikz_save
from keras.models import Sequential, Graph
from keras.layers.core import Activation, Dense
from keras.layers.recurrent import GRU, LSTM
from keras.callbacks import Callback, EarlyStopping
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.noise import GaussianDropout
from sklearn.metrics import accuracy_score, precision_score, recall_score

on_cluster = True
use_graph = sys.argv[1] == 'graph'

num_conv = int(sys.argv[2])
num_gru = int(sys.argv[3])
num_dense = int(sys.argv[4])
batch_size = int(sys.argv[5])
patience = int(sys.argv[6])

optimizer = sys.argv[7]

offset = 0


if on_cluster:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt

features_folder = 'features_100_40_25'

def fetch_data(table_file):
    with open(table_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for row in csv_reader:
            wav_file = re.sub('block\d+\/', '', row[0].lower())
            npy_file = wav_file.replace('.wav', '.logspec.npy')
            yield (npy_file, row[1])

def extract_data(data, offset=0):
    d = list(data)
    xs = [np.load(os.path.join('..', '..', features_folder, npy_file)) for npy_file, _ in d]
    
    if offset > 0:
        xs = [x[offset:-offset] for x in xs]
    
    ys = [1 if label == 'N' else 0 for _, label in d]
    
    return (xs, ys)

def build_tensors(xs, ys, max_length):
    n = len(xs)
    X = np.zeros((n, max_length, np.shape(xs[0])[1]))
    Y = np.array(ys, dtype='int')
    for i in range(n):
        x = xs[i]
        m, _ = np.shape(x)
        X[i, 0:m, :] = x
    return (X, Y)

if __name__ == '__main__':
    train_path = os.path.join('..', '..', 'TRAIN.TBL')
    dev_path = os.path.join('..', '..', 'D1.TBL')
    test_path = os.path.join('..', '..', 'TEST.TBL')
    
    training = list(fetch_data(train_path))
    dev = list(fetch_data(dev_path))
    testing = list(fetch_data(test_path))
    
    xs_train, ys_train = extract_data(training, offset=offset)
    xs_dev, ys_dev = extract_data(dev, offset=offset)
    xs_test, ys_test = extract_data(testing, offset=offset)
    
    t1 = len(xs_train)
    t2 = len(xs_dev) + t1
    
    xs_all = xs_train + xs_dev + xs_test
    ys_all = ys_train + ys_dev + ys_test
    
    max_length = max([np.shape(x)[0] for x in xs_all])
    
    model = Graph() if use_graph else Sequential()
    
    if use_graph:
        model.add_input(name='input', input_shape=(max_length, 40))

        input_name = 'input'
        
        for i in range(num_conv):
            conv_name = 'conv' + str(i)
            maxpooling_name = 'maxpooling' + str(i)
            conv = Convolution1D(64, 3, activation='relu', init='he_normal')
            maxpooling = MaxPooling1D()
            model.add_node(conv, name=conv_name, input=input_name)
            model.add_node(maxpooling, name=maxpooling_name, input=conv_name)
            input_name = maxpooling_name
        
        for i in range(num_gru):
            gru_forward_name = 'fw_gru' + str(i)
            gru_backward_name = 'bw_gru' + str(i)
            dropout_name = 'dropout' + str(i)
            return_sequences = i < num_gru - 1
            forward = GRU(128, activation='relu', inner_activation='sigmoid', init='he_normal', return_sequences=return_sequences)
            backward = GRU(128, activation='relu', inner_activation='sigmoid', init='he_normal', return_sequences=return_sequences, go_backwards=True)
            model.add_node(forward, name=gru_forward_name, input=input_name)
            model.add_node(backward, name=gru_backward_name, input=input_name)
            model.add_node(GaussianDropout(0.4), name=dropout_name, inputs=[gru_forward_name,gru_backward_name], merge_mode='sum')
            input_name = dropout_name
        
        for i in range(num_dense):
            dense_name = 'dense' + str(i)
            dense = Dense(128, activation='relu', init='he_normal')
            model.add_node(dense, name=dense_name, input=input_name)
            input_name = dense_name
        
        model.add_node(Dense(1, activation='sigmoid', init='he_normal'), name='dense', input=input_name)
        model.add_output(name='output', input='dense')
        
        model.compile(loss={'output': 'binary_crossentropy'}, optimizer=optimizer)
        
    else:
        for i in range(num_conv):
            model.add(Convolution1D(64, 3, input_dim=40, activation='relu', init='he_normal'))
            model.add(MaxPooling1D())
        
        for i in range(num_gru):
            return_sequences = i < num_gru - 1
            go_backwards = sys.argv[1] != 'normal'
            if sys.argv[1] == 'lstm':
                model.add(LSTM(128, input_dim=40, activation='relu', inner_activation='sigmoid', init='he_normal', return_sequences=True))
                model.add(GaussianDropout(0.4))
                model.add(LSTM(128, input_dim=40, activation='relu', inner_activation='sigmoid', init='he_normal', go_backwards=go_backwards, return_sequences=return_sequences))
                model.add(GaussianDropout(0.4))
            else:
                if go_backwards:
                    model.add(GRU(128, input_dim=40, activation='relu', inner_activation='sigmoid', init='he_normal', go_backwards=True, return_sequences=True))
                    model.add(GaussianDropout(0.4))
                model.add(GRU(128, input_dim=40, activation='relu', inner_activation='sigmoid', init='he_normal', go_backwards=True, return_sequences=return_sequences))
                model.add(GaussianDropout(0.4))
        
        for i in range(num_dense):
            model.add(Dense(128, activation='relu', init='he_normal'))
        
        model.add(Dense(1, activation='sigmoid'))
        
        #model.load_weights('../../results/sequentialconv0gru1dense2batch128offset0adam/weights')
        
        model.compile(loss='binary_crossentropy', optimizer=optimizer, class_mode='binary')
    
    X, Y = build_tensors(xs_all, ys_all, max_length)
    
    for i in range(X.shape[2]):
        X[:,:,i] -= np.mean(X[:,:,i])
        X[:,:,i] /= np.std(X[:,:,i])
    
    class MeasurementsLogger(Callback):
        def on_train_begin(self, logs={}):
            self.i = 0
            self.training_measures = np.zeros((self.params['nb_epoch'], 3))
            self.dev_measures = np.zeros((self.params['nb_epoch'], 3))
            self.testing_measures = np.zeros((self.params['nb_epoch'], 3))
        
        def on_epoch_end(self, batch, logs={}):
            p = model.predict({'input': X}, verbose=1)['output'] if use_graph else model.predict(X, verbose=1)
            
            self.training_measures[self.i, 0] = accuracy_score(Y[:t1], np.round(p[:t1]))
            self.training_measures[self.i, 1] = recall_score(Y[:t1], np.round(p[:t1]))
            self.training_measures[self.i, 2] = precision_score(Y[:t1], np.round(p[:t1]))
            print('Training accuracy:', self.training_measures[self.i, 0])
            print('Training recall:', self.training_measures[self.i, 1])
            print('Training precision:', self.training_measures[self.i, 2])
            np.save('training_predictions', p[:t1])
            
            self.dev_measures[self.i, 0] = accuracy_score(Y[t1:t2], np.round(p[t1:t2]))
            self.dev_measures[self.i, 1] = recall_score(Y[t1:t2], np.round(p[t1:t2]))
            self.dev_measures[self.i, 2] = precision_score(Y[t1:t2], np.round(p[t1:t2]))
            print('Development accuracy:', self.dev_measures[self.i, 0])
            print('Development recall:', self.dev_measures[self.i, 1])
            print('Development precision:', self.dev_measures[self.i, 2])
            np.save('dev_predictions', p[t1:t2])
            
            self.testing_measures[self.i, 0] = accuracy_score(Y[t2:], np.round(p[t2:]))
            self.testing_measures[self.i, 1] = recall_score(Y[t2:], np.round(p[t2:]))
            self.testing_measures[self.i, 2] = precision_score(Y[t2:], np.round(p[t2:]))
            print('Testing accuracy:', self.testing_measures[self.i, 0])
            print('Testing recall:', self.testing_measures[self.i, 1])
            print('Testing precision:', self.testing_measures[self.i, 2])
            np.save('testing_predictions', p[t2:])
            
            np.save('training_measures', self.training_measures)
            np.save('dev_measures', self.dev_measures)
            np.save('testing_measures', self.testing_measures)
            
            model.save_weights('weights', overwrite=True)
            
            self.i += 1
            
            self.plot()
        
        def plot(self):
            plt.clf()
            lines = plt.plot(range(1, self.i+1), self.training_measures[:self.i])
            plt.legend(iter(lines), ('accuracy', 'recall', 'precision'), loc=4)
            plt.xlabel('Iteration')
            plt.ylabel('Measures')
            plt.axis([1, self.i+1, 0, 1])
            plt.title('Training measures over time')
            plt.savefig('training_measures.png')
            tikz_save('training_measures.tikz', show_info=False, figurewidth='0.5\\textwidth')
            
            plt.clf()
            lines = plt.plot(range(1, self.i+1), self.dev_measures[:self.i])
            plt.legend(iter(lines), ('accuracy', 'recall', 'precision'), loc=4)
            plt.xlabel('Iteration')
            plt.ylabel('Measures')
            plt.axis([1, self.i+1, 0, 1])
            plt.title('Development measures over time')
            plt.savefig('dev_measures.png')
            tikz_save('dev_measures.tikz', show_info=False, figurewidth='0.5\\textwidth')
            
            plt.clf()
            lines = plt.plot(range(1, self.i+1), self.testing_measures[:self.i])
            plt.legend(iter(lines), ('accuracy', 'recall', 'precision'), loc=4)
            plt.xlabel('Iteration')
            plt.ylabel('Measures')
            plt.axis([1, self.i+1, 0, 1])
            plt.title('Testing measures over time')
            plt.savefig('testing_measures.png')
            tikz_save('testing_measures.tikz', show_info=False, figurewidth='0.5\\textwidth')
            
    class AccuracyLogger(Callback):
        def on_epoch_end(self, batch, logs={}):
            p = model.predict({'input': X[t1:t2]}, verbose=1)['output'] if use_graph else model.predict(X[t1:t2], verbose=1)
            print('Development accuracy:', accuracy_score(Y[t1:t2], np.round(p)))
    
    measurements = MeasurementsLogger()
    
    if use_graph:
        model.fit({'input': X[:t1], 'output': Y[:t1]}, nb_epoch=100, batch_size=batch_size, validation_data={'input': X[t1:t2], 'output': Y[t1:t2]}, callbacks=[measurements, EarlyStopping(patience=patience)])
    else:
        model.fit(X[:t1], Y[:t1], nb_epoch=100, batch_size=batch_size, validation_data=(X[t1:t2], Y[t1:t2]), callbacks=[measurements, EarlyStopping(patience=patience)])
    

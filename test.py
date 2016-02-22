import csv, re, os, sys
import numpy as np
import matplotlib
from matplotlib2tikz import save as tikz_save
from keras.models import Sequential, Graph
from keras.layers.core import Activation, Dense, TimeDistributedDense
from keras.layers.recurrent import GRU, LSTM
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.noise import GaussianDropout
from sklearn.metrics import accuracy_score, precision_score, recall_score

on_cluster = True

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
    
    ys = [1 if label == 'A' else 0 for _, label in d]
    
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
    
    xs_train, ys_train = extract_data(training)
    xs_dev, ys_dev = extract_data(dev)
    xs_test, ys_test = extract_data(testing)
    
    t1 = len(xs_train)
    t2 = len(xs_dev) + t1
    
    xs_all = xs_train + xs_dev + xs_test
    ys_all = ys_train + ys_dev + ys_test
    
    max_length = max([np.shape(x)[0] for x in xs_all])
    
    model = Sequential()
    model.add(GRU(128, input_dim=40, activation='relu', inner_activation='sigmoid', init='he_normal', return_sequences=True))
    model.add(GaussianDropout(0.4))
    model.add(GRU(128, activation='relu', inner_activation='sigmoid', init='he_normal', go_backwards=True))
    model.add(GaussianDropout(0.4))
    model.add(Dense(128, activation='relu', init='he_normal'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', class_mode='binary')
    
    #model = Graph()
    #model.add_input(name='input', input_shape=(max_length, 40))
    #model.add_node(GRU(128, activation='relu', inner_activation='sigmoid', init='he_normal'), name='gru1', input='input')
    #model.add_node(GRU(128, activation='relu', inner_activation='sigmoid', init='he_normal', go_backwards=True), name='gru2', input='input')
    #model.add_node(GaussianDropout(0.4), name='dropout', inputs=['gru1', 'gru2'], merge_mode='sum')
    #model.add_node(Dense(128, activation='relu', init='he_normal'), name='dense1', input='dropout')
    #model.add_node(Dense(1, activation='sigmoid', init='he_normal'), name='dense2', input='dense1')
    #model.add_output(name='output', input='dense2')
    
    #model.compile(loss={'output': 'binary_crossentropy'}, optimizer='adam')
    
    X, Y = build_tensors(xs_all, ys_all, max_length)
    
    for i in range(X.shape[2]):
        X[:,:,i] -= np.mean(X[:,:,i])
        X[:,:,i] /= np.std(X[:,:,i])
        
    np.save('X', X)
    np.save('Y', Y)
    
    class MeasurementsLogger(Callback):
        def on_train_begin(self, logs={}):
            self.i = 0
            self.training_measures = np.zeros((self.params['nb_epoch'], 3))
            self.dev_measures = np.zeros((self.params['nb_epoch'], 3))
            self.testing_measures = np.zeros((self.params['nb_epoch'], 3))
        
        def on_epoch_end(self, batch, logs={}):
            p = model.predict(X, verbose=1)
            
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
            
            model.save_weights('weights.hdf5', overwrite=True)
            
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
    
    class ModelEvaluation(Callback):
        def on_epoch_end(self, batch, logs={}):
            loss, accuracy = model.evaluate(X[t2:], Y[t2:], show_accuracy=True)
            print('test_loss:', loss, '- test_acc:', accuracy)
    
    class RecallPrinter(Callback):
        def on_train_begin(self, logs={}):
            self.accuracy = np.zeros((self.params['nb_epoch'], 3))
            self.recall = np.zeros((self.params['nb_epoch'], 3))
        
        def on_epoch_end(self, epoch, logs={}):
            p = model.predict(X, verbose=1)
            #p = model.predict({'input': X}, verbose=1)['output']
            
            train_acc = accuracy_score(Y[:t1], np.round(p[:t1]))
            dev_acc = accuracy_score(Y[t1:t2], np.round(p[t1:t2]))
            test_acc = accuracy_score(Y[t2:], np.round(p[t2:]))
            print('Accuracy | train:', train_acc, 'dev:', dev_acc, 'test:', test_acc)
            
            train_recall = (recall_score(Y[:t1], np.round(p[:t1])) + recall_score(Y[:t1], np.round(p[:t1]), pos_label=0)) / 2
            dev_recall = (recall_score(Y[t1:t2], np.round(p[t1:t2])) + recall_score(Y[t1:t2], np.round(p[t1:t2]), pos_label=0)) / 2
            test_recall = (recall_score(Y[t2:], np.round(p[t2:])) + recall_score(Y[t2:], np.round(p[t2:]), pos_label=0)) / 2
            print('Recall   | train:', train_recall, 'dev:', dev_recall, 'test:', test_recall)
            
            self.accuracy[epoch, :] = np.array([train_acc, dev_acc, test_acc])
            self.recall[epoch, :] = np.array([train_recall, dev_recall, test_recall])
            
            plt.clf()
            plt.subplot(211)
            lines = plt.plot(range(1, epoch+2), self.accuracy[:epoch+1])
            plt.legend(iter(lines), ('train', 'dev', 'test'), loc=4)
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.axis([1, epoch+2, 0, 1])
            plt.subplot(212)
            lines = plt.plot(range(1, epoch+2), self.recall[:epoch+1])
            plt.legend(iter(lines), ('train', 'dev', 'test'), loc=4)
            plt.xlabel('Epoch')
            plt.ylabel('Average recall')
            plt.axis([1, epoch+2, 0, 1])
            plt.savefig('results.png')
    
    measurements = MeasurementsLogger()
    
    measures = ['val_acc', 'val_loss']
    #callbacks = [ModelCheckpoint('best_weights_' + m + '.hdf5', monitor=m, save_best_only=True) for m in measures]
    callbacks = [RecallPrinter()]
    
    indices = np.where(Y[:t1] == 0)
    for i in range(10):
        print('Iteration', i)
        print('only intoxicated')
        model.fit(X[indices,:,:], Y[indices,:,:], nb_epoch=1)
        print('sober and intoxicated')
        model.fit(X[:t1], Y[:t1], nb_epoch=1, callbacks=callbacks)
    #model.fit(X[:t1], Y[:t1], nb_epoch=50, batch_size=128, callbacks=callbacks)
    
    #model.fit({'input': X[:t1], 'output': Y[:t1]}, nb_epoch=50, batch_size=128, callbacks=callbacks)
    

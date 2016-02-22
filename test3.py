import csv, re, os, sys
import numpy as np
import matplotlib
from matplotlib2tikz import save as tikz_save
from keras.models import Sequential, Graph
from keras.layers.core import Activation, Dense, TimeDistributedDense, Dropout
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

if __name__ == '__main__':
    t1 = 5400 + 3060
    t2 = 900 + t1
    
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
    
    X = np.load('../../X.npy')
    Y = np.load('../../Y.npy')
    
    class ModelEvaluation(Callback):
        def on_epoch_end(self, batch, logs={}):
            loss, accuracy = model.evaluate(X[t2:], Y[t2:], show_accuracy=True)
            print('test_loss:', loss, '- test_acc:', accuracy)
    
    class RecallPrinter(Callback):
        def on_train_begin(self, logs={}):
            self.bestrecall = 0
            
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
            tikz_save('results.tikz', show_info=False, figurewidth='0.5\\textwidth')
            
            if dev_recall > self.bestrecall:
                self.bestrecall = dev_recall
                model.save_weights('weights.hdf5', overwrite=True)
    
    model.fit(X[:t1], Y[:t1], nb_epoch=50, batch_size=128, callbacks=[RecallPrinter()])
    
    #model.fit({'input': X[:t1], 'output': Y[:t1]}, nb_epoch=50, batch_size=128, callbacks=callbacks)
    

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets,layers,Sequential,metrics,optimizers
batchsz=128
total_words=10000
max_review_length=80
embedding_len=100 #word vectsor len
(x_train,y_train),(x_test,y_test) = datasets.imdb.load_data(num_words=total_words)
print(x_train.shape,y_train.shape)
x_train=keras.preprocessing.sequence.pad_sequences(x_train,max_review_length)
x_test=keras.preprocessing.sequence.pad_sequences(x_test,max_review_length)
print(x_train.shape,x_test.shape)

db_train=tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(batchsz,drop_remainder=True)
db_test=tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(batchsz,drop_remainder=True)

class MyRNN(keras.Model):
    def __init__(self,units):
        super(MyRNN, self).__init__()
        #[b,80]=>[b,80,100]
        self.embedding=layers.Embedding(total_words,embedding_len,
                                        input_length=max_review_length)
        self.rnn_cell0=layers.SimpleRNNCell(units,dropout=0.2)
        self.rnn_cell1=layers.SimpleRNNCell(units,dropout=0.2)
        self.outlayer=layers.Dense(1)
        self.state0 = [tf.zeros([batchsz, units])]
        self.state1 = [tf.zeros([batchsz, units])]



    def __call__(self, inputs, training=None):
        x=inputs
        x=self.embedding(x)
        state0=self.state0
        state1=self.state1
        for word in tf.unstack(x,axis=1):#[b,80,100]=>[b,100] for each word
            out0,state0=self.rnn_cell0(word,state0)#循环80次
            out1,state1=self.rnn_cell1(out0,state1)
        #out:([b,64])
        x=self.outlayer(out1)
        prob=tf.sigmoid(x)
        

        return prob

def main():
    units=64
    epochs=4
    model=MyRNN(units)
    model.compile(optimizer=optimizers.Adam(0.001),
                  loss=tf.losses.BinaryCrossentropy(),
                  metrics=['Accuracy'],
                  experimental_run_tf_function = False)
    model.fit(db_train,epochs=epochs,validation_data=db_test)

if __name__=='__main__':
    main()




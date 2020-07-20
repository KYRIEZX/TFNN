import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets,layers,Sequential,metrics,optimizers
batchsz=128
total_words=10000
max_review_length=80
embedding_len=100 #word vector len
(x_train,y_train),(x_test,y_test) = datasets.imdb.load_data(num_words=total_words)
print(x_train.shape,y_train.shape)
x_train=keras.preprocessing.sequence.pad_sequences(x_train,max_review_length)
x_test=keras.preprocessing.sequence.pad_sequences(x_test,max_review_length)
print(x_train.shape,x_test.shape)

db_train=tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(batchsz,drop_remainder=True)
db_test=tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(batchsz,drop_remainder=True)

class MyLSTM(keras.Model):
    def __init__(self,units):
        super(MyLSTM, self).__init__()
        self.cell=layers.LSTMCell(units)
        self.embedding=layers.Embedding(total_words,embedding_len,input_length=max_review_length)
        self.state=[tf.zeros([batchsz,units]),tf.zeros([batchsz,units])]# h, c
        self.fc=layers.Dense(1)

    def __call__(self,inputs,training=None):
        x=inputs
        x=self.embedding(x)
        state=self.state
        for word in tf.unstack(x,axis=1):
            out,state=self.cell(x,state)
        x=self.fc(x)
        pred=tf.nn.sigmoid(x)
        return pred

def main():
    units=64
    epochs=4
    model=MyLSTM(units)
    for epoch in range(epochs):
        for step,(x,y) in enumerate(db_train):
           with tf.GradientTape() as tape:
               out = model(x)
               loss=tf.losses.BinaryCrossentropy(out,y)

           grads = tape.gradient(loss, model.trainable_variables)
           # w' = w - lr * grad
           keras.optimizer.apply_gradients(zip(grads, model.trainable_variables))

           if step % 100 == 0:
               print(epoch, step, 'loss:', loss.numpy())


if __name__ == '__main__':
    main()

import tensorflow as tf
from tensorflow.keras import Sequential,layers,datasets,optimizers,metrics
from tensorflow import keras
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def preprocess(x,y):
    x=tf.cast(x,dtype=tf.float32)/255
    y=tf.cast(y,dtype=tf.int32)
    return x,y

batchsz=128
(x,y),(x_val,y_val)=datasets.cifar10.load_data()
y=tf.squeeze(y)
y_val=tf.squeeze(y_val)
print('datasets:',x.shape,y.shape)
print('tset_set:',x_val.shape,y_val.shape)
db_train=tf.data.Dataset.from_tensor_slices((x,y))
db_train=db_train.map(preprocess).shuffle(50000).batch(batchsz)
db_test=tf.data.Dataset.from_tensor_slices((x_val,y_val))
db_test=db_test.map(preprocess).shuffle(10000).batch(batchsz)

class MyDense(layers.Layer):
    def __init__(self,input_dim,output_dim):
        super(MyDense,self).__init__()
        self.kernel=self.add_weight('w',[input_dim,output_dim])
        #self.bias=self.add_variable('b',[output_dim])

    def call(self,inputs,training=None):
        x=inputs@self.kernel
        return x

class MyNetwork(keras.Model):
    def __init__(self):
        super(MyNetwork,self).__init__()
        self.fc1=MyDense(32*32*3,256)
        self.fc2=MyDense(256,256)
        self.fc3 = MyDense(256, 256)
        self.fc4 = MyDense(256, 256)
        self.fc5 = MyDense(256, 10)

    def call(self, inputs, training=None):
         x=tf.reshape(inputs,[-1,32*32*3])

         x = tf.nn.relu(self.fc1(x))s
         x = tf.nn.relu(self.fc2(x))
         x = tf.nn.relu(self.fc3(x))
         x = tf.nn.relu(self.fc4(x))
         x = self.fc5(x)
         return x

network=MyNetwork()
network.compile(optimizer=optimizers.Adam(lr=0.01),
                loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
network.fit(db_train,epochs=5,validation_data=db_test,validation_freq=2)
network.evaluate(db_test)
network.save_weights('./weights')




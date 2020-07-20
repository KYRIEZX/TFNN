import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets,layers,optimizers,Sequential,metrics
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def preprocess(x,y):
    x=tf.cast(x,dtype=tf.float32)
    y=tf.cast(y,dtype=tf.int32)
    return x,y

(x,y),(x_val,y_val)=datasets.cifar10.load_data()
print('datasets:',x.shape,y.shape)
print('tset_set:',x_val.shape,y_val.shape)

db_train=tf.data.Dataset.from_tensor_slices((x,y))
db_test=tf.data.Dataset.from_tensor_slices((x_val,y_val))

db_train=db_train.map(preprocess).shuffle(50000).batch(128)
db_test=db_test.map(preprocess).shuffle(10000).batch(128)

conv_layers=[ # 5 units of conv and max pooling
    # unit 1
    layers.Conv2D(64,[3,3],padding="same",activation=tf.nn.relu),
    layers.Conv2D(64,kernel_size=[3,3],padding="same",activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2,2],strides=2,padding="same"),
    # unit 2
    layers.Conv2D(128,kernel_size=[3,3],padding="same",activation=tf.nn.relu),
    layers.Conv2D(128,kernel_size=[3,3],padding="same",activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2,2],strides=2,padding="same"),
    # unit 3
    layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),
    # unit 4
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),
    # unit 5
    layers.Conv2D(512, kernel_size=[1, 1], padding="same", activation=tf.nn.relu),
    layers.Conv2D(512, kernel_size=[1, 1], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same")
]

fc_layers=[
    layers.Dense(256,activation="relu"),
    layers.Dense(128,activation="relu"),
    layers.Dense(10,activation="relu")
]


def main():
    conv_net=Sequential(conv_layers)
    fc_net=Sequential(fc_layers)
    conv_net.build(input_shape=[None,32,32,3])
    fc_net.build(input_shape=[None,512])
    variables=conv_net.trainable_variables+fc_net.trainable_variables
    optimizer=optimizers.Adam(lr=1e-4)
    for epoch in range(50):
        for step,(x,y) in enumerate(db_train):
            with tf.GradientTape() as tape:
                #[b,32,32,3]=>[b,1,1,512]=>[b,512]
                out = conv_net(x)
                out=tf.reshape(out,[-1,512])
                #[b,512]=>[b,100]
                logits=fc_net(out)

                y_onehot=tf.one_hot(y,depth=10)
                loss=tf.losses.categorical_crossentropy(y_onehot,logits,from_logits=True)
                loss=tf.reduce_mean(loss)

            grads=tape.gradient(loss,variables)
            optimizer.apply_gradients(zip(grads,variables))

            if step%20==0:
                print(epoch,step,'loss:',float(loss))

if __name__=='__main__':
    main()

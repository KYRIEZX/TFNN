import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets,optimizers,layers,metrics,Sequential

def preprocessor(x,y):
    x=tf.cast(x,dtype=tf.float32)/255
    y=tf.cast(y,dtype=tf.int32)
    x=tf.reshape(x,[-1,28*28])
    y=tf.one_hot(y,depth=10)
    return x,y

(x,y),(x_test,y_test)=datasets.mnist.load_data()
print('datasets:', x.shape, y.shape, x.min(), x.max())
print('data_test',x_test.shape,y_test.shape)
db=tf.data.Dataset.from_tensor_slices((x,y))
db=db.map(preprocessor).shuffle(60000).batch(64)
db_test=tf.data.Dataset.from_tensor_slices((x_test,y_test))
db_test=db_test.map(preprocessor).shuffle(10000).batch(64)
sample = next(iter(db))
print(sample[0].shape, sample[1].shape)

network=keras.Sequential(
    [
        keras.layers.Dense(512,activation='relu'),
        keras.layers.Dense(256,activation='relu'),
        keras.layers.Dense(128,activation='relu'),
        keras.layers.Dense(64,activation='relu'),
        keras.layers.Dense(10)
    ]
)
network.build(input_shape=[None,28*28])
network.summary()

network.compile(optimizer=optimizers.Adam(lr=0.1),
                loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
network.fit(db,epochs=3,validation_data=db_test,validation_freq=1)
network.evaluate(db_test)


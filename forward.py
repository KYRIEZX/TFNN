import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets

def preprocess(x,y):
    x=tf.cast(x,dtype=tf.float32)/255.
    y=tf.cast(y,dtype=tf.int32)
    return x,y

def mnist_train():
    #x=([60k,28,28]),y=([60k])
    (x,y),(x_val,y_val) = datasets.mnist.load_data()
    print(x.shape,y.shape,x_val.shape,y_val.shape)
    #每次取128张图片
    db_train=tf.data.Dataset.from_tensor_slices((x,y))
    db_train=db_train.map(preprocess)
    db_train=db_train.shuffle(60000).batch(128)
    db_test = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    db_test = db_test.map(preprocess)
    db_test = db_test.shuffle(10000).batch(128)

    #[b,784]=>[b,256]=>[b,128]=>[b,10]
    w1=tf.Variable(tf.random.truncated_normal([784,256],stddev=0.1))
    b1=tf.Variable(tf.zeros([256]))
    w2=tf.Variable(tf.random.truncated_normal([256,128],stddev=0.1))
    b2=tf.Variable(tf.zeros([128]))
    w3=tf.Variable(tf.random.truncated_normal([128,10],stddev=0.1))
    b3=tf.Variable(tf.zeros([10]))
    lr=0.001

    total_correct, total_num=0,0
    for epoch in range(10):
        for step,(x,y) in enumerate(db_train):
            x=tf.reshape(x,[-1,28*28])
            with tf.GradientTape() as tape:
                #需要求导的部分：
                #h1=x@w1+b1
                h1=tf.matmul(x,w1)+tf.broadcast_to(b1,[x.shape[0],256])
                h1=tf.nn.relu(h1)
                h2=h1@w2+b2
                h2=tf.nn.relu(h2)
                out=h2@w3+b3
                out=tf.nn.relu(out)
                #计算误差 compute_loss
                y_onehot=tf.one_hot(y,depth=10)
                #平方和/N
                loss=tf.reduce_mean(tf.square(out-y_onehot))
            #计算梯度
            grads=tape.gradient(loss,[w1,b1,w2,b2,w3,b3])
            #梯度回归步骤,原地更新assign
            w1.assign_sub(lr * grads[0])
            b1.assign_sub(lr * grads[1])
            w2.assign_sub(lr * grads[2])
            b2.assign_sub(lr * grads[3])
            w3.assign_sub(lr * grads[4])
            b3.assign_sub(lr * grads[5])

            if step%100==0:
                print(epoch,step,'loss',float(loss))

        for step, (x,y) in enumerate(db_test):
            x=tf.reshape(x,[-1,28*28])
            h1=tf.nn.relu(x@w1+b1)
            h2=tf.nn.relu(h1@w2+b2)
            out=tf.nn.relu(h2@w3+b3)
            #out:([b,10])
            #prob:([b,10]) pred:([b])
            prob=tf.nn.softmax(out,axis=1)
            pred=tf.argmax(prob,axis=1)
            pred=tf.cast(pred,dtype=tf.int32)
            correct=tf.cast(tf.equal(pred,y),dtype=tf.int32)
            correct=tf.reduce_sum(correct)
            total_correct += int(correct)
            total_num += x.shape[0]

            if step % 100 == 0:
                print(epoch, step, 'accuracy:', total_correct/total_num)


if __name__=='__main__':
    mnist_train()

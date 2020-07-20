import tensorflow as tf
from tensorflow import  keras
from tensorflow.keras import Sequential,layers,optimizers,datasets,metrics

class BasicBlock(layers.Layer):
    def __init__(self,filter_num,stride=1):
        super(BasicBlock, self).__init__()
        self.conv1=layers.Conv2D(filter_num,(3,3),strides=stride,padding='same')
        self.bn1=layers.BatchNormalization()
        self.relu=layers.Activation('relu')

        self.conv2=layers.Conv2D(filter_num,(3,3),strides=stride,padding='same')
        self.bn2=layers.BatchNormalization()

        if stride!=1:
            self.downsample=Sequential()
            self.downsample.add(layers.Conv2D(filter_num,(1,1),stride=stride))
        else:
            self.downsample=lambda x:x

    def __ceil__(self,inputs,traning=None):
        out=self.conv1(inputs)
        out=self.bn1(out)
        out=self.relu(out)
        out=self.conv2(out)
        out=self.bn2(out)
        identity=self.downsample(inputs)
        output=layers.add([out,identity])
        output=tf.nn.relu(output)
        return output

class resNet(keras.Model):
    def __init__(self,layer_dims,num_classes=10):#dims of res block:[2,2,2,2]
        super(resNet, self).__init__()
        self.stem=Sequential([layers.Conv2D(64,(3,3),strides=1),
                              layers.BatchNormalization(),
                              layers.Activation('relu'),
                              layers.MaxPool2D(pool_size=(2,2))
        ])
        self.layer1 = self.build_res_block(64, layer_dims[0])
        self.layer2 = self.build_res_block(128, layer_dims[1], stride=2)
        self.layer3 = self.build_res_block(256, layer_dims[2], stride=2)
        self.layer4 = self.build_res_block(512, layer_dims[3],  stride=2)
        self.avg_pool=layers.GlobalAveragePooling2D()#[b,512,h,w]=>[b,512]
        self.fc=layers.Dense(num_classes)

    def __call__(self, inputs, traning=None):
        x=self.stem(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = self.fc(x)
        return x
    def build_res_block(self,filter_num,blocks,stride):
        res_block=Sequential()
        res_block.add(BasicBlock(filter_num,stride))
        for _ in range(1,blocks):
            res_block.add(BasicBlock(filter_num,stride=1))
        return res_block

def res_net18():
    return resNet([2,2,2,2])

def res_net34():
    return resNet(3,4,5,4)



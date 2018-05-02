from __future__ import print_function, division
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
# from keras.layers.advanced_activations import ReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt
import sys
import numpy as np


# def Activation('relu')(x, leak=0.2, name="lrelu"):
#     with tf.variable_scope(name):
#         f1 = 0.5 * (1 + leak)
#         f2 = 0.5 * (1 - leak)
#         return f1 * x + f2 * abs(x)

class GAN():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows,self.img_cols,self.channels)
        # optimizer = Adam(0.0002,0.5)
        optimizer = SGD(lr=0.0002, decay=1e-6, momentum=0.9, nesterov=True)
        #构建和编译判别器
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])
        #构建和编译生成器
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy',optimizer=optimizer)
        #生成器将噪声作为输入并生成图片
        z = Input(shape=(100,))
        img = self.generator(z)
        #对于组合模型，只训练生成器
        self.discriminator.trainable = False
        #判别器将生成的图像作为输入并确定其有效性
        valid = self.discriminator(img)
        #组合模型（叠加生成器和判别器）将噪声作为输入=》产生图像=》确定有效性
        self.combined = Model(z,valid)
        self.combined.compile(loss='binary_crossentropy',optimizer=optimizer)
    def build_generator(self):
        noise_shape = (100,)
        model = Sequential()
        #layer1 (None,100)>>(None,256)
        model.add(Dense(256,input_shape=noise_shape))
        model.add(Activation('relu'))
        model.add(BatchNormalization(momentum=0.8))
        #layer2 (None,256)>>(None,512)
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(BatchNormalization(momentum=0.8))
        #layer3 (None,512)>>(None,1024)
        model.add(Dense(1024))
        model.add(Activation('relu'))
        model.add(BatchNormalization(momentum=0.8))
        #layer4 (None,1024)>>(None,784)
        model.add(Dense(np.prod(self.img_shape),activation='tanh'))#np.prod返回数组内元素的乘积，（28，28，1）=》784
        model.add(Reshape(self.img_shape))#784=>(28,28,1)
        model.summary()
        noise = Input(shape=noise_shape)
        img = model(noise)
        return Model(noise,img)
    def build_discriminator(self):
        img_shape = (self.img_rows,self.img_cols,self.channels)
        model = Sequential()
        model.add(Flatten(input_shape=img_shape))#（None，28，28，1）拉伸为（None,784)
        #layer1 (None,784)>>(None,512)
        model.add(Dense(512))
        model.add(Activation('relu'))
        #layer2 (None,512)>>(None,256)
        model.add(Dense(256))
        model.add(Activation('relu'))
        #layer3 (None,256)>>(None,1)
        model.add(Dense(1,activation='sigmoid'))
        model.summary()
        img = Input(shape=img_shape)
        validity = model(img)
        return Model(img,validity)
    def train(self,epochs,batch_size=128,sample_interval=50):
        #加载数据集,训练集矩阵，训练集标签，测试集矩阵，测试集标签, uint8型 0-255 x_train=(60000,28,28)
        (x_train,_),(_,_) = mnist.load_data()
        #将像素值归整到-1到1之间
        x_train = (x_train.astype(np.float32) - 127.5) / 127.5
        x_train = np.expand_dims(x_train,axis=3)#扩展维度(60000,28,28)>>(60000,28,28,1)
        half_batch = int(batch_size/2)#64 = 128/2
        for epoch in range(epochs):
            #训练判别器，随机选择半批图像
            #在0-60000之间，随机生成64个数构成列表索引
            idx = np.random.randint(0,x_train.shape[0],half_batch)
            #随机选择图片
            imgs = x_train[idx]
            noise = np.random.normal(0,1,(half_batch,100))
            #生成半批新的图像
            gen_imgs = self.generator.predict(noise)
            #训练判别器
            #手动将一个个batch的数据送入网络中训练
            d_loss_real = self.discriminator.train_on_batch(imgs,np.ones((half_batch,1)))
            print(d_loss_real)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs,np.zeros((half_batch,1)))
            d_loss = 0.5 * np.add(d_loss_real,d_loss_fake)
            #训练生成器
            noise = np.random.normal(0,1,(batch_size,100))
            #生成器希望判别器将生成的样本标记为1
            valid_y = np.array([1] * batch_size)#[1,1,1....1]batch_size个
            g_loss = self.combined.train_on_batch(noise,valid_y)
            print(d_loss , '\n')
            print(d_loss[0],'\n')
            print(d_loss[1],'\n')
            print('epoch : %d ,[D loss: %f, acc.: %.2f%%] ,[G loss : %f]' % (epoch,d_loss[0],100*d_loss[1],g_loss))
            #保存生成的图片
            if epoch % sample_interval == 0:
                self.sample_images(epoch)
    def sample_images(self,epoch):
        r,c = 5,5
        noise = np.random.normal(0,1,(r*c,100))
        gen_imgs = self.generator.predict(noise)
        #-1到1归整到0-1
        gen_imgs = 0.5 * gen_imgs + 0.5
        fig,axs = plt.subplots(r,c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt,:,:,0],cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig('images/mnist_%d.png' % epoch)
        plt.close()
    
if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=300000,batch_size=32,sample_interval=200)
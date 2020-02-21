from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D 
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense 
from keras.utils import to_categorical
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model

(xtrain,ytrain),(xtest,ytest)=fashion_mnist.load_data()
(xtrain1,ytrain1),(xtest1,ytest1)=fashion_mnist.load_data()
#print(xtrain.shape)

xtrain = xtrain.astype('float32')
xtest = xtest.astype('float32')
xtrain /= 255
xtest /= 255

xtrain1 = xtrain1.astype('float32')
xtest1 = xtest1.astype('float32')
xtrain1/= 255
xtest1 /= 255

xtest1 = xtest1.reshape(xtest.shape[0], 28, 28, 1)
print(xtest.shape)
print(xtest1.shape)
model_fully_conn=load_model('fashion_mnist_fully_conn.h5')
model_cnn = load_model('fashion_mnist_cnn.h5')

y_pred1 = model_fully_conn.predict_classes(xtest)
y_pred2= model_cnn.predict_classes(xtest1)

output1=[]
output2=[]
loss1, accuracy1 = model_fully_conn.evaluate(xtest, ytest, verbose=0)
loss2, accuracy2 = model_cnn.evaluate(xtest1, ytest, verbose=0)
output1.append('Loss on Test Data : '+str(loss1))
output1.append('Accuracy on Test Data : '+str(accuracy1))
output2.append('Loss on Test Data : '+str(loss2))
output2.append('Accuracy on Test Data : '+str(accuracy2))

output1.append('gt_label,pred_label')
output2.append('gt_label,pred_label')
for i in range(len(xtest)):
	output1.append(str(ytest[i])+','+str(y_pred1[i]))
	output2.append(str(ytest[i])+','+str(y_pred2[i]))

 
np.savetxt('multi-layer-net.txt', output1, delimiter =" ", fmt="%s") 
np.savetxt('convolution-neural-net.txt', output2, delimiter =" ", fmt="%s")



import numpy as np
import tensorflow as tf

#this is the dataset import
(x_train0, y_train0),(x_test0, y_test) = tf.keras.datasets.mnist.load_data()
#x_train is the training data(grayscale) and y_train is the classification label(correct)
#do check the data structures to gain complete understanding of the operations done further
#also to make sense of the matrix equalities and dimensions of the transormation marices

#i used a simple linear to map the values from  (a,a,a) --> (b,b,b) , where a is in [0,256] and b is in the range [0,1]   
x_train0=[1/256]*x_train0
x_test0=[1/256]*x_test0

#flatten the data sets
#u can simply use reshape too here, the below written was done to catch a bug...
x_train=list()
for i in x_train0 : 
    x_train.append(np.reshape(i, 784))
x_train=np.array(x_train)
x_test=list()
for i in x_test0 : 
    x_test.append(np.reshape(i, 784))
x_test=np.array(x_test)  

#construct y_train and test which are in shape (10.1) where entry at index i is 1 iff label is i else 0
y_train=list()
for y in y_train0 : 
    z=np.zeros(10)
    z[y]=1
    y_train.append(z)
y_train=np.array(y_train)
#print the y_train 0 and the newer ones to make senses of the above shapings

nl1=32  #number of neurons in layer 1
nl2=10  #possibilities in output layer

#the code for forward propagation, i'll use one hidden layer for my nn
# w1 and w2 are the transformation matrices from input to l1 and l1 to l2 respectively
#b1 and b2 are the respective biases which appear in the fitting, say, for the first iter : y(forward) = w1.x_train + b1
#here let the number of layers be two, nl1 = 20 and nl2 = 10 then,
#as nl1 has 20 neurons the weight matrix w1 has the shape (20,784). i.e. the first layer must reduce the 784 features to nl1, rhus the matrix of transforemation has the shape.
# b1 then also has the shape (20,1)
#lets then initialize w1, w2, b1, b2
w1=np.random.uniform(-0.5, 0.5, (nl1, 784)) 
w2=np.random.uniform(-0.5, 0.5, (nl2, nl1))
b1=np.zeros((nl1))
b2=np.zeros((nl2))
#print(w1@x_train[0])

#Try to add a normalization constraint i.e. the row vectors of wxs are normalized at all times

e=0.025 #the learning rate
r=0 #total right classifications
t=0 #total classifications
epochs=6 # number of times we will go over the data

#it is very important to get a theoretical understanding how does the code actually updates the weights over iterations
#up going over a record the algo uses w1, w2 , b1, b2 to get the output matrix which is used to classify the pattern to a number
#after clasification, the algo then in backpropogation step adjusts the weights for the next iteration... 
#the accuracy shown is the percentage of records correctly classified in the forward propagation step not the testing ! 
#thus if  there is a pattern as we are trying to figure out with the algorithm, i.e. if there exists a matrice w1* and the 
# matrix after the nth iteration is w1(n), for every delta > 0 there exists a N in naturals
# s.t. || w1(n) - w1* || < delta for all n>N...

for time in range(epochs) :
    for pattern, number in zip(x_train, y_train) : 
        #y1 = b1 + w1 @ pattern
        y1 = b1 + np.matmul(w1, pattern)
        # b1 is 20, and as w1 is 20,784 and as pattern is 784, so, y1 is 20,
        #first transformation
        y1_box = 1/ (1 + np.exp(-y1)) # (20,1)
        #boxit, i.e. activation using sigmoid
        #y2 = b2 + w2 @ y1_box
        y2 = b2 + np.matmul(w2,y1_box)
        out = 1/ (1 + np.exp(-y2))
        
        #cost and error
        error= 1/len(out)*np.sum((out-number)**2, axis=0)
        r+= int(np.argmax(out)==np.argmax(number))

        #feedback, backpropogation
        delta_y2=out-number
        w2+= -e*np.matmul(np.transpose(delta_y2[np.newaxis]),y1_box[np.newaxis])
        b2+= -e*delta_y2
        delta_y1=np.transpose(w2)@delta_y2*(y1_box*(1-y1_box))
        w1+= -e*np.matmul(np.transpose(delta_y1[np.newaxis]),pattern[np.newaxis])
        b1+= -e*delta_y1
        t+=1
    print(f"train{time+1} -> accuracy : {round((r/t)*100, 2)}%")
    r=0
    t=0

for test, ans in zip(x_test, y_test) : 
    y1 = b1 + np.matmul(w1, test)
    y1_box = 1/ (1 + np.exp(-y1))
    y2 = b2 + np.matmul(w2,y1_box)
    out = 1/ (1 + np.exp(-y2))
    r+= int(np.argmax(out)==ans)
    t+=1
print(f"\nTest Performance : `was able to classify {r} samples out of {t} thus the accuracy {round((r/t)*100, 2)}%")

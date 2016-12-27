import tensorflow as tf
import scipy.misc
import scipy.io
import numpy as np
import os
import sys


#image width, height and number of channels
imWidth = 800
imHeight = 600
nchannels = 3

dest_path = './final'
vgg19_path = 'imagenet-vgg-verydeep-19.mat'


#alpha is the weight given to the content loss and beta to the style loss
alpha = 5
beta = 1000

#get the weights and biases from vgg model
def getWeights(vgg19_layers, layerid):
    weight = vgg19_layers[layerid][0][0][0][0][0]
    bias = vgg19_layers[layerid][0][0][0][0][1]
    return weight, bias
    
#create vgg model using weights which are loaded from pretrained vgg19 model mat file
def create_tfvgg():   
    vgg19 = scipy.io.loadmat(vgg19_path)
    
    #three classes in struct: 'classes', 'layers' and 'normalization'
    #43 structs in 'layers' (1x43 struct)
    vgg19_layers = vgg19['layers'][0]
    
    #vggnet 
    vggnet = {}
    vggnet['inputimage'] = tf.Variable(np.zeros((1, imHeight, imWidth, 3)).astype('float32'))
    
    weights = getWeights( vgg19_layers, 0 )
    vggnet['conv1_1'] = tf.nn.relu(tf.nn.conv2d(vggnet['inputimage'], weights[0], strides=[1, 1, 1, 1], padding='SAME')+ weights[1])
    weights = getWeights( vgg19_layers, 2 )
    vggnet['conv1_2'] = tf.nn.relu(tf.nn.conv2d(vggnet['conv1_1'], weights[0], strides=[1, 1, 1, 1], padding='SAME')+ weights[1])
    vggnet['pool1'] = tf.nn.avg_pool(vggnet['conv1_2'], ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
    weights = getWeights( vgg19_layers, 5 )
    vggnet['conv2_1'] = tf.nn.relu(tf.nn.conv2d(vggnet['pool1'], weights[0], strides=[1, 1, 1, 1], padding='SAME')+ weights[1])
    weights = getWeights( vgg19_layers, 7 )
    vggnet['conv2_2'] = tf.nn.relu(tf.nn.conv2d(vggnet['conv2_1'], weights[0], strides=[1, 1, 1, 1], padding='SAME')+ weights[1])
    vggnet['pool2'] = tf.nn.avg_pool(vggnet['conv2_2'], ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
    weights = getWeights( vgg19_layers, 10 )
    vggnet['conv3_1'] = tf.nn.relu(tf.nn.conv2d(vggnet['pool2'], weights[0], strides=[1, 1, 1, 1], padding='SAME')+ weights[1])
    weights = getWeights( vgg19_layers, 12 )
    vggnet['conv3_2'] = tf.nn.relu(tf.nn.conv2d(vggnet['conv3_1'], weights[0], strides=[1, 1, 1, 1], padding='SAME')+ weights[1])
    weights = getWeights( vgg19_layers, 14 )
    vggnet['conv3_3'] = tf.nn.relu(tf.nn.conv2d(vggnet['conv3_2'], weights[0], strides=[1, 1, 1, 1], padding='SAME')+ weights[1])
    weights = getWeights( vgg19_layers, 16 )
    vggnet['conv3_4'] = tf.nn.relu(tf.nn.conv2d(vggnet['conv3_3'], weights[0], strides=[1, 1, 1, 1], padding='SAME')+ weights[1])
    vggnet['pool3'] = tf.nn.avg_pool(vggnet['conv3_4'], ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
    weights = getWeights( vgg19_layers, 19 )
    vggnet['conv4_1'] = tf.nn.relu(tf.nn.conv2d(vggnet['pool3'], weights[0], strides=[1, 1, 1, 1], padding='SAME')+ weights[1])
    weights = getWeights( vgg19_layers, 21 )
    vggnet['conv4_2'] = tf.nn.relu(tf.nn.conv2d(vggnet['conv4_1'], weights[0], strides=[1, 1, 1, 1], padding='SAME')+ weights[1])
    weights = getWeights( vgg19_layers, 23 )
    vggnet['conv4_3'] = tf.nn.relu(tf.nn.conv2d(vggnet['conv4_2'], weights[0], strides=[1, 1, 1, 1], padding='SAME')+ weights[1])
    weights = getWeights( vgg19_layers, 25 )
    vggnet['conv4_4'] = tf.nn.relu(tf.nn.conv2d(vggnet['conv4_3'], weights[0], strides=[1, 1, 1, 1], padding='SAME')+ weights[1])
    vggnet['pool4'] = tf.nn.avg_pool(vggnet['conv4_4'], ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
    weights = getWeights( vgg19_layers, 28 )
    vggnet['conv5_1'] = tf.nn.relu(tf.nn.conv2d(vggnet['pool4'], weights[0], strides=[1, 1, 1, 1], padding='SAME')+ weights[1])
    weights = getWeights( vgg19_layers, 30 )
    vggnet['conv5_2'] = tf.nn.relu(tf.nn.conv2d(vggnet['conv5_1'], weights[0], strides=[1, 1, 1, 1], padding='SAME')+ weights[1])
    weights = getWeights( vgg19_layers, 32 )
    vggnet['conv5_3'] = tf.nn.relu(tf.nn.conv2d(vggnet['conv5_2'], weights[0], strides=[1, 1, 1, 1], padding='SAME')+ weights[1])
    weights = getWeights( vgg19_layers, 34 )
    vggnet['conv5_4'] = tf.nn.relu(tf.nn.conv2d(vggnet['conv5_3'], weights[0], strides=[1, 1, 1, 1], padding='SAME')+ weights[1])
    vggnet['pool5'] = tf.nn.avg_pool(vggnet['conv5_4'], ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
    return vggnet


#subroutine to save ouput image
def saveim(dest,im):
    im = im + np.array([103.939, 116.779, 123.68]).reshape((1,1,1,3))
    im = np.clip(im[0], 0, 255).astype('uint8')
    scipy.misc.imsave(dest, im)


def main():
    print('enter main')
	
	#sys args: 1) number of iterations  2) source image path   3) style image path
    niterations = int(sys.argv[1])
    sourcePath = sys.argv[2]
    stylePath = sys.argv[3]
	
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    print('1')
    vggnet = create_tfvgg();
	
    #generate uniform random white noise
    xImage = np.random.uniform(-50, 50, (1, imHeight,imWidth, 3)).astype('float32')
    
    # read source and style image and perform mean subtraction - 
    #The input images should be zero-centered by mean pixel (rather than mean image) subtraction
    source = scipy.misc.imread(sourcePath)
    source = scipy.misc.imresize(source, (imHeight, imWidth))
    source = np.reshape(source,((1,)+source.shape))
    #source = source[::-1]
    source = source - np.array([103.939, 116.779, 123.68]).reshape((1,1,1,3))

    style = scipy.misc.imread(stylePath)
    style = scipy.misc.imresize(style, (imHeight, imWidth))
    style = np.reshape(style,((1,)+style.shape))
    #style = style[::-1]
    style = style - np.array([103.939, 116.779, 123.68]).reshape((1,1,1,3))

    

    #calculate source loss
    sess.run([vggnet['inputimage'].assign(source)])
    p = sess.run(vggnet['conv4_2'])
    x = vggnet['conv4_2']
    M = p.shape[1]*p.shape[2]
    N = p.shape[3]
    sourceloss = (0.5) * tf.reduce_sum(tf.pow((x - p),2))
    
    
    #calculate style loss
    sess.run([vggnet['inputimage'].assign(style)])
	#uniform contirbution of style from all the images, (wl = 0.2)
    convlayers = [('conv1_1',0.2),('conv2_1',0.2),('conv3_1',0.2),('conv4_1',0.2),('conv5_1',0.2)]
    styleloss = 0
    for i in range(len(convlayers)):
		a = sess.run(vggnet[convlayers[i][0]])
		M = a.shape[1]* a.shape[2]
		N = a.shape[3]
		aMat = np.reshape(a, (M,N))
		A = np.dot(aMat.T,aMat)
		
		g = vggnet[convlayers[i][0]]
		greshaped = tf.reshape(g, (M,N))
		G = tf.matmul(tf.transpose(greshaped), greshaped)
		
		loss = (1./(4*N*N*M*M)) * tf.reduce_sum(tf.pow(G - A, 2))
		styleloss = styleloss + convlayers[i][1] * loss

	#total loss as a weighted sum of source and style losses
    totalloss = alpha * sourceloss + beta * styleloss
    
    train = tf.train.AdamOptimizer(2.0).minimize(totalloss)
    
    #Initialise with white noise
    sess.run(tf.initialize_all_variables())
    sess.run(vggnet['inputimage'].assign(xImage))

    if not os.path.exists(dest_path):
      os.mkdir(dest_path)

    for i in range(niterations):
        sess.run(train)
        if i%100 ==0 or i==niterations-1:
            result_img = sess.run(vggnet['inputimage'])
            print sess.run(totalloss)
  
    saveim(os.path.join(dest_path,'final.png'),result_img)

if __name__ == '__main__':
  main()
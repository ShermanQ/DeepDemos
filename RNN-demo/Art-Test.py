# %导入几个需要用到的库
import tensorflow as tf
import numpy as np
import scipy.io
import scipy.misc
import os

# %设置图像的路径以及需要调整到的宽度值和高度值
IMAGE_W = 800
IMAGE_H = 600
CONTENT_IMG =  './images/Taipei101.jpg'
STYLE_IMG = './images/StarryNight.jpg'
OUTOUT_DIR = './results'
OUTPUT_IMG = 'results.png'
VGG_MODEL = 'imagenet-vgg-verydeep-19.mat'
# %这里采用在imagenet中训练好的VGG19模型，需要下载
INI_NOISE_RATIO = 0.7
# %对输入图像噪声干扰的比例
STYLE_STRENGTH = 500
# %风格强度
ITERATION = 5000
# %迭代次数
# 1
# 2
# 3
# 4
# 5
# 6
# 7
# 8
# 9
# 10
# 11
# 12
# 13
# 14
# 15
# 16
# 17
# 18
# 19
# 20
# 21
# 22
#
#
# 图6 训练中采用的训练图像
# %对VGG19网络定义内容层和风格层
CONTENT_LAYERS =[('conv4_2',1.)]
STYLE_LAYERS=[('conv1_1',1.),('conv2_1',1.),('conv3_1',1.),('conv4_1',1.),('conv5_1',1.)]

MEAN_VALUES = np.array([123, 117, 104]).reshape((1,1,1,3))
#
# %定义建立池化层和卷积层网络的函数
def build_net(ntype, nin, nwb=None):
  if ntype == 'conv':
    return tf.nn.relu(tf.nn.conv2d(nin, nwb[0], strides=[1, 1, 1, 1], padding='SAME')+ nwb[1])
  elif ntype == 'pool':
    return tf.nn.avg_pool(nin, ksize=[1, 2, 2, 1],
                  strides=[1, 2, 2, 1], padding='SAME')

# %获取VGG19模型参数的函数
def get_weight_bias(vgg_layers, i,):
  weights = vgg_layers[i][0][0][0][0][0]
  weights = tf.constant(weights)
  bias = vgg_layers[i][0][0][0][0][1]
  bias = tf.constant(np.reshape(bias, (bias.size)))
  return weights, bias

# %建立训练网络模型
def build_vgg19(path):
  net = {}
  vgg_rawnet = scipy.io.loadmat(path)
  vgg_layers = vgg_rawnet['layers'][0]
  net['input'] = tf.Variable(np.zeros((1, IMAGE_H, IMAGE_W, 3)).astype('float32'))
  net['conv1_1'] = build_net('conv',net['input'],get_weight_bias(vgg_layers,0))
  net['conv1_2'] = build_net('conv',net['conv1_1'],get_weight_bias(vgg_layers,2))
  net['pool1']   = build_net('pool',net['conv1_2'])
  net['conv2_1'] = build_net('conv',net['pool1'],get_weight_bias(vgg_layers,5))
  net['conv2_2'] = build_net('conv',net['conv2_1'],get_weight_bias(vgg_layers,7))
  net['pool2']   = build_net('pool',net['conv2_2'])
  net['conv3_1'] = build_net('conv',net['pool2'],get_weight_bias(vgg_layers,10))
  net['conv3_2'] = build_net('conv',net['conv3_1'],get_weight_bias(vgg_layers,12))
  net['conv3_3'] = build_net('conv',net['conv3_2'],get_weight_bias(vgg_layers,14))
  net['conv3_4'] = build_net('conv',net['conv3_3'],get_weight_bias(vgg_layers,16))
  net['pool3']   = build_net('pool',net['conv3_4'])
  net['conv4_1'] = build_net('conv',net['pool3'],get_weight_bias(vgg_layers,19))
  net['conv4_2'] = build_net('conv',net['conv4_1'],get_weight_bias(vgg_layers,21))
  net['conv4_3'] = build_net('conv',net['conv4_2'],get_weight_bias(vgg_layers,23))
  net['conv4_4'] = build_net('conv',net['conv4_3'],get_weight_bias(vgg_layers,25))
  net['pool4']   = build_net('pool',net['conv4_4'])
  net['conv5_1'] = build_net('conv',net['pool4'],get_weight_bias(vgg_layers,28))
  net['conv5_2'] = build_net('conv',net['conv5_1'],get_weight_bias(vgg_layers,30))
  net['conv5_3'] = build_net('conv',net['conv5_2'],get_weight_bias(vgg_layers,32))
  net['conv5_4'] = build_net('conv',net['conv5_3'],get_weight_bias(vgg_layers,34))
  net['pool5']   = build_net('pool',net['conv5_4'])
  return net

# %定义内容损失，算法上面已经讲过
def build_content_loss(p, x):
  M = p.shape[1]*p.shape[2]
  N = p.shape[3]
  loss = (1./(2* N**0.5 * M**0.5 )) * tf.reduce_sum(tf.pow((x - p),2))
  return loss

# %定义Gram矩阵，针对网络中的图像
def gram_matrix(x, area, depth):
  x1 = tf.reshape(x,(area,depth))
  g = tf.matmul(tf.transpose(x1), x1)
  return g

# %定义Gram矩阵，针对风格图像
def gram_matrix_val(x, area, depth):
  x1 = x.reshape(area,depth)
  g = np.dot(x1.T, x1)
  return g

# %建立风格损失，算法上面已经介绍过
def build_style_loss(a, x):
  M = a.shape[1]*a.shape[2]
  N = a.shape[3]
  A = gram_matrix_val(a, M, N )
  G = gram_matrix(x, M, N )
  loss = (1./(4 * N**2 * M**2)) * tf.reduce_sum(tf.pow((G - A),2))
  return loss


# %读取图像的函数
def read_image(path):
  image = scipy.misc.imread(path)
  image = scipy.misc.imresize(image,(IMAGE_H,IMAGE_W))
  image = image[np.newaxis,:,:,:]
  image = image - MEAN_VALUES
  return image

# %保存图像的函数
def write_image(path, image):
  image = image + MEAN_VALUES
  image = image[0]
  image = np.clip(image, 0, 255).astype('uint8')
  scipy.misc.imsave(path, image)

# %主函数
def main():
  net = build_vgg19(VGG_MODEL)
  sess = tf.Session()
  sess.run(tf.initialize_all_variables())
  noise_img = np.random.uniform(-20, 20, (1, IMAGE_H, IMAGE_W, 3)).astype('float32')
  content_img = read_image(CONTENT_IMG)
  style_img = read_image(STYLE_IMG)

# %这里通过迁移学习，提取VGG19中的参数来搭建新的网络，而且新的网络也是提取网络
  sess.run([net['input'].assign(content_img)])
  cost_content = sum(map(lambda l,: l[1]*build_content_loss(sess.run(net[l[0]]) ,  net[l[0]])
    , CONTENT_LAYERS))

  sess.run([net['input'].assign(style_img)])
  cost_style = sum(map(lambda l: l[1]*build_style_loss(sess.run(net[l[0]]) ,  net[l[0]])
    , STYLE_LAYERS))

# %这里是把风格作为首要优化的对象
  cost_total = cost_content + STYLE_STRENGTH * cost_style
  optimizer = tf.train.AdamOptimizer(2.0)

  train = optimizer.minimize(cost_total)
  sess.run(tf.initialize_all_variables())
  sess.run(net['input'].assign( INI_NOISE_RATIO* noise_img + (1.-INI_NOISE_RATIO) * content_img))
# %构建输入图像，其中也有一部分内容图像
  if not os.path.exists(OUTOUT_DIR):
      os.mkdir(OUTOUT_DIR)

# %注意，如果笔记本性能不是太好，建议把迭代次数调小点
  for i in range(ITERATION):
    sess.run(train)
    if i%100 ==0:
      result_img = sess.run(net['input'])
      print(sess.run(cost_total))
      write_image(os.path.join(OUTOUT_DIR,'%s.png'%(str(i).zfill(4))),result_img)

  write_image(os.path.join(OUTOUT_DIR,OUTPUT_IMG),result_img)


if __name__ == '__main__':
  main()

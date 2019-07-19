import tensorflow as tf
import numpy as np
import os,sys,cv2
from my_model import hdrnet_model
from data_loader import generator2, get_img_path_from_folder
os.environ["CUDA_VISIBLE_DEVICES"] = '1' 

def print_schedule(string):
    sys.stdout.flush()
    sys.stdout.write(string + '\r')

class hdrnet_train():
    def __init__(self):
        self.batch_size = 32
        self.save_path = './result/'
        self.hdrnet = hdrnet_model()
    
    def build(self):
        self.fullres_input = tf.placeholder(tf.float32, (self.batch_size, None, None, 3))
        self.lowres_input = tf.placeholder(tf.float32, (self.batch_size, 224, 224, 3))
        output = self.hdrnet.model(self.lowres_input,self.fullres_input)
        self.output = tf.cast(tf.squeeze(tf.clip_by_value(output, 0, 1)), tf.float32)
        self.ill = tf.clip_by_value(self.output, self.fullres_input, 1)
        self.final_result = np.divide(self.fullres_input, self.ill + 0.00001) 
        # loss
        self.r_loss = self.l2_loss(self.fullres_input, self.final_result)
        self.s_loss = self.tv_loss(self.fullres_input, self.output, 2)
        self.c_loss = self.cos_loss(self.fullres_input, self.final_result)
        self.total_loss = self.r_loss + self.s_loss + self.c_loss
        # optimizer
        opt = tf.train.AdamOptimizer(0.0001)
        self.opt_op = opt.minimize(self.total_loss)

    def l2_loss(self, target, prediction, name=None):
        with tf.name_scope(name, default_name='l2_loss', values=[target, prediction]):
            loss = tf.reduce_mean(tf.square(target-prediction))
            #loss = tf.reduce_mean(target-prediction)
        return loss

    def tv_loss(self, input_, output, weight):
        I = tf.image.rgb_to_grayscale(input_)
        L = tf.log(I+0.0001)
        dx = L[:, :-1, :-1, :] - L[:, :-1, 1:, :]
        dy = L[:, :-1, :-1, :] - L[:, 1:, :-1, :]

        alpha = tf.constant(1.2)
        lamda = tf.constant(1.5)  
        dx = tf.divide(lamda, tf.pow(tf.abs(dx),alpha)+ tf.constant(0.0001))
        dy = tf.divide(lamda, tf.pow(tf.abs(dy),alpha)+ tf.constant(0.0001))
        shape = output.get_shape()
        x_loss = dx *((output[:, :-1, :-1, :] - output[:, :-1, 1:, :])**2)
        y_loss = dy *((output[:, :-1, :-1, :] - output[:, 1:, :-1, :])**2)
        tvloss = tf.reduce_mean(x_loss + y_loss)/2.0
        return tvloss*weight
        
    def cos_loss(self, target, prediction):
        len_a = tf.sqrt(tf.reduce_sum(target*target, axis = -1))
        len_b = tf.sqrt(tf.reduce_sum(prediction*prediction, axis = -1))
        cosine_distence = tf.reduce_sum(target*prediction, axis = -1)/(len_a*len_b + 0.0001)
        return 1-tf.reduce_mean(cosine_distence)

    def train(self):
        '''image = np.random.uniform(0,1,(128,256,256,3))
        image2 = np.random.uniform(0,1,(128,100,100,3))'''
        x_folder = '/home/ben85824/hardnet2/DeepUPE/main/data/dped/blackberry/training_data/blackberry'
        y_folder = '/home/ben85824/hardnet2/DeepUPE/main/data/dped/blackberry/training_data/canon'
        x_path_list, y_path_list =  get_img_path_from_folder(x_folder, y_folder)
        g = generator2(x_path_list[:3000], y_path_list[:3000], 32, 3)
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            step = 30000
            for i in range(step):
                x,y = next(g)
                l,r,c,s,_ = sess.run([self.total_loss, self.r_loss, self.c_loss, self.s_loss,self.opt_op],feed_dict = {self.lowres_input:x, self.fullres_input:y})
                print_schedule('step:%d, total:%f, r_loss:%f, s_loss:%f, c_loss:%f'%(i,l,r,s,c))
                if (i+1)%30 == 0:
                    result = sess.run(self.final_result, feed_dict = {self.lowres_input:x, self.fullres_input:y})
                    input_img = x[0]
                    result_img = result[0]
                    #input_img = cv2.cvtColor(input_img,cv2.COLOR_BGR2RGB)
                    #result_img = cv2.cvtColor(result_img,cv2.COLOR_BGR2RGB)
                    img = np.concatenate((input_img, result_img), axis = 1) * 255.0
                    img = img.astype('uint8')
                    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                    cv2.imwrite(self.save_path + 'output_step_%d.jpg'%(i), img)


train_obj = hdrnet_train()
train_obj.build()
train_obj.train()



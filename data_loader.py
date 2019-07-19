import tensorflow as tf
import numpy as np
import cv2, os, sys
import matplotlib.pyplot as plt

def generator(x_path, y_path, batchsize, depth, shuffle = True):
    def load_data(data_path1, data_path2):
        x_image = tf.read_file(data_path1)
        x_image = tf.image.decode_jpeg(x_image, channels=3)
        x_image = tf.image.resize_images(x_image, (224, 224))
        #image = tf.image.random_flip_left_right(image)
        x_image = tf.cast(x_image, tf.float32) / 255.0
        y_image = tf.read_file(data_path2)
        y_image = tf.image.decode_jpeg(y_image, channels=3)
        y_image = tf.cast(y_image, tf.float32) / 255.0
        return x_image, y_image
    with tf.Session() as sess:
        # create a dataset
        train_dataset = tf.data.Dataset().from_tensor_slices((x_path, y_path))
        train_dataset = train_dataset.map(load_data, num_parallel_calls=4)
        train_dataset = train_dataset.repeat()
        if shuffle:
            train_dataset = train_dataset.shuffle(buffer_size=1000)
        train_dataset = train_dataset.batch(batchsize)
        train_iterator = train_dataset.make_initializable_iterator()
        train_batch = train_iterator.get_next()
        sess.run(train_iterator.initializer)
        while True:
            try:
                X_batch, y_batch = sess.run(train_batch)
                yield (X_batch, y_batch)
            except:
                train_iterator = train_dataset.make_initializable_iterator()
                sess.run(train_iterator.initializer)
                train_batch = train_iterator.get_next()
                X_batch, y_batch = sess.run(train_batch)
                yield (X_batch, y_batch)
def print_schedule(string):
    sys.stdout.flush()
    sys.stdout.write(string + '\r')
def generator2(x_path, y_path, batchsize, depth, shuffle = True):
    def load_data(data_path1, data_path2):
        x_image = cv2.imread(data_path1)
        
        x_image = cv2.resize(x_image, (224, 224))
        x_image = cv2.cvtColor(x_image,cv2.COLOR_BGR2RGB)
        #image = tf.image.random_flip_left_right(image)
        x_image = x_image / 255.0
        y_image = cv2.imread(data_path2)
        y_image = cv2.resize(y_image, (224, 224))
        y_image = cv2.cvtColor(y_image,cv2.COLOR_BGR2RGB)
        y_image = y_image / 255.0
        
        return x_image, y_image
    total = len(x_path)
    index = list(range(total))
    np.random.shuffle(index)
    x,y = [],[]
    for i in range(total):
        print_schedule(str(i)+str('/')+str(total))
        x_img, y_img = load_data(x_path[i],y_path[i])
        x.append(x_img)
        y.append(y_img)
    x = np.array(x)
    y = np.array(y)
    counter = 0
    while True:
        if counter + batchsize>total:
            np.random.shuffle(index)
            counter = 0
        x_batch = x[index[counter:counter+batchsize]]
        y_batch = y[index[counter:counter+batchsize]]
        counter += batchsize
        yield x_batch,y_batch

def get_img_path_from_folder(folder1, folder2):
    img_list1 = [folder1+'/'+x for x in os.listdir(folder1)]
    img_list2 = [folder2+'/'+x for x in os.listdir(folder1)]
    return img_list1, img_list2

    

if __name__ == '__main__':
    x_folder = '/home/ben85824/hardnet2/DeepUPE/main/data/dped/blackberry/training_data/blackberry'
    y_folder = '/home/ben85824/hardnet2/DeepUPE/main/data/dped/blackberry/training_data/canon'
    x_path_list, y_path_list =  get_img_path_from_folder(x_folder, y_folder)
    



    

    
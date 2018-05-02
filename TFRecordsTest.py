import tensorflow as tf
import matplotlib.pyplot as plt

data_path = './tfdata/train0.tfrecords'

with tf.Session() as sess:
    # feature key and its data type for data restored in tfrecords file
    feature = {'height': tf.FixedLenFeature([], tf.int64),
               'width': tf.FixedLenFeature([], tf.int64),
               'depth': tf.FixedLenFeature([], tf.int64),
               'image_raw': tf.FixedLenFeature([], tf.string),
               'label':tf.FixedLenFeature([], tf.int64)}
    # define a queue base on input filenames
    filename_queue = tf.train.string_input_producer([data_path])
    # define a tfrecords file reader
    reader = tf.TFRecordReader()
    # read in serialized example data
    _, serialized_example = reader.read(filename_queue)
    # decode example by feature
    features = tf.parse_single_example(serialized_example, features=feature)
    image = tf.image.decode_jpeg(features['image_raw'])
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)  # convert dtype from unit8 to float32 for later resize
    label = tf.cast(features['label'], tf.int64)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    # restore image to [height, width, 3]
    image = tf.reshape(image, [height, width, 3])
    # resize
    # image = tf.image.resize_images(image, [224, 224])
    # create bathch
    data1 = [image, label]
    print(data1)
    images, labels = tf.train.batch([image, label], batch_size=10, capacity=300, num_threads=8) # capacity是队列的最大容量，num_threads是dequeue后最小的队列大小，num_threads是进行队列操作的线程数。

    # initialize global & local variables
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    # create a coordinate and run queue runner objects
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for batch_index in range(3):
        batch_images, batch_labels = sess.run([images, labels])
        for i in range(10):
            plt.imshow(batch_images[i, ...])
            plt.show()
            print ("Current image label is: %d" % (batch_labels[i]))
    # close threads
    coord.request_stop()
    coord.join(threads)
    sess.close()
#opens the trained neural net and feeds in images to classify

import tensorflow as tf
import numpy as np
import dataset

img_size = 128
train_batch_size = 1
classes = ['amps', 'congo drums', 'drum set', 'guitars', 'hand drum', 'keyboards']
answer = np.zeros((0,9))
test_path='Data\\Uncategorized'
train_path ='Data\\Uncategorized'
validation_size = 1

sess = tf.Session()
saver = tf.train.import_meta_graph('my_test_model.meta')
saver.restore(sess, tf.train.latest_checkpoint('./'))

graph = tf.get_default_graph()

predictor = graph.get_tensor_by_name("y_pred:0")
x = graph.get_tensor_by_name('x:0')
y_true = graph.get_tensor_by_name('y_true:0')


images_to_categorize = dataset.read_train_sets(train_path, img_size, classes, validation_size=validation_size)

file = open('output.csv','w')
file.write("Itm,Category,Certainty")
file.write("\n")

for z in range(0, 3200):
    x_batch, y_true_batch, ids, cls_batch = images_to_categorize.train.next_batch(train_batch_size)

    nn_thinks = sess.run(predictor, {x: x_batch.reshape(-1, 49152), y_true: answer})
    choiceIndexArray = sess.run(tf.argmax(nn_thinks,dimension=1))
    choiceIndex = choiceIndexArray[0];
    category = classes[choiceIndex]
    percent = nn_thinks[0][choiceIndex]
    print("Itm: " + str(ids[0]).replace(".jpg","") + " Category: " + str(category) + " Certainty: " + str(percent * 100))
    file.write(str(ids[0]).replace(".jpg","") + "," + str(category) + "," + str(percent * 100))
    file.write("\n")
file.close()

import tensorflow as tf
import numpy as np
from visualize import visualize_accuracy, visualize_loss
import san
import random
import config
from preprocess import preprocess_wo_valid, to_batch
# import utils


# # # # --------------------- Initialize Model --------------------- # # # #
model = san.san(sa_type=0, layers=(2, 1, 2, 4, 1), kernels=[3,5,5,5,5], num_classes=config.NUM_CLASSES)
model.build(input_shape=(config.BATCH_SIZE, config.channels, config.image_height, config.image_width))
model.summary()




# # # # --------------------- Load the data --------------------- # # # #
# # train_images, train_labels, test_images, test_labels = utils.read_train_test_data("sample_dataset")
# # train_images = utils.data_preprocess(train_images)
train_images, train_labels, test_images, test_labels = preprocess_wo_valid("whole_dataset")
train_images = train_images[:(len(train_images)//config.BATCH_SIZE)*config.BATCH_SIZE]
train_labels = train_labels[:(len(train_labels)//config.BATCH_SIZE)*config.BATCH_SIZE]
test_images = test_images[:(len(test_images)//config.BATCH_SIZE)*config.BATCH_SIZE]
test_labels = test_labels[:(len(test_labels)//config.BATCH_SIZE)*config.BATCH_SIZE]
# # train_labels = utils.one_hot_encoder(train_labels)
# train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=config.NUM_CLASSES, dtype='uint8') # to one-hot


# # # X_train, X_val, y_train, y_val = utils.validation_data(train_img, train_lab)


# num_train_examples = train_images.shape[0]
# num_train_batches = int(num_train_examples / config.BATCH_SIZE)
# indices = list(range(num_train_examples))
# random.shuffle(indices)
# train_images = tf.gather(train_images, indices)
# train_labels = tf.gather(train_labels, indices)
# train_in_batches = to_batch(train_images, train_labels)

# # train_generator, val_generator = utils.data_augmentation(X_train, y_train, X_val, y_val)
# train_DataGen = tf.keras.preprocessing.image.ImageDataGenerator(zoom_range=0.2, 
#                                                             width_shift_range=0.1, 
#                                                             height_shift_range = 0.1, 
#                                                             horizontal_flip=True)
# train_in_batches = train_DataGen.flow(train_images, train_labels, batch_size=config.BATCH_SIZE) # train_lab is categorical 
train_in_batches = to_batch(train_images, train_labels) # augmentation + batch
# test_DataGen = tf.keras.preprocessing.image.ImageDataGenerator(zoom_range=0.2, 
#                                                             width_shift_range=0.1, 
#                                                             height_shift_range = 0.1, 
#                                                             horizontal_flip=True)
# test_in_batches = test_DataGen.flow(test_images, test_labels, batch_size=config.BATCH_SIZE) # train_lab is categorical
test_in_batches = to_batch(test_images, test_labels) # augmentation + batch





# # # # --------------------- Loss Acuracy Calculators --------------------- # # # #
optimizer = tf.keras.optimizers.Adam(config.LEARNING_RATE)
loss_object = tf.keras.losses.CategoricalCrossentropy()
train_loss_calculator = tf.keras.metrics.Mean(name='train_loss')
train_accuracy_calculator = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
test_loss_calculator = tf.keras.metrics.Mean(name='test_loss')
test_accuracy_calculator = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')




# # # # --------------------- Train and Test Procedure --------------------- # # # #
@tf.function
def train_wo_valid(model, train_images, train_labels):        
    with tf.GradientTape() as tape:
        train_predictions = model(train_images, training=True) 
        # train_loss = model.loss(train_predictions, train_labels_in_batch) # check first: if loss is implemented under san
        loss = loss_object(y_true=train_labels, y_pred=train_predictions)
        
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    # train_accuracy = model.accuracy(train_predictions, train_labels_in_batch)
    train_loss_calculator(loss)
    train_accuracy_calculator(train_labels, train_predictions)
    # return train_loss, train_accuracy


@tf.function
def test(model, test_images, test_labels):
    # test_predictions = tf.reshape(test_predictions, (-1,config.NUM_CLASSES)) # reshape the result?
    # test_loss = model.loss(test_predictions, test_labels)
    # test_accuracy = model.accuracy(test_predictions, test_labels)
    test_predictions = model(test_images, training=False)
    # print("test pred shape before reshape: " + test_predictions.shape)
    # test_predictions = tf.reshape(test_predictions, (-1, config.NUM_CLASSES)) # check necessity by the previous line
    loss = loss_object(test_labels, test_predictions)

    test_loss_calculator(loss)
    test_accuracy_calculator(test_labels,test_predictions)
    # return test_loss, test_accuracy






# # # # --------------------- Training and Testing --------------------- # # # #
for epoch in range(config.EPOCHS):
    train_loss_calculator.reset_states()
    train_accuracy_calculator.reset_states()
    test_loss_calculator.reset_states()
    test_accuracy_calculator.reset_states()

    # for i in range(num_train_batches):
    #     start = i*config.BATCH_SIZE
    #     train_images_in_batch = train_images[start:(start+config.BATCH_SIZE)]
    #     train_labels_in_batch = train_labels[start:(start+config.BATCH_SIZE)]
    #     train_loss, train_accuracy = train_wo_valid(model, train_images_in_batch, train_labels_in_batch)
    #     print(f"batch: {i}/{num_train_batches}, loss: {train_loss}, accuracy: {train_accuracy}")
    #     model.loss_list.append(train_loss)
    #     model.accuracy_list.append(train_accuracy)

    for batch in range(len(train_in_batches)):
        train_images, train_labels = train_in_batches[batch]
        train_images = tf.transpose(train_images, [0, 3, 1, 2]) # channel first
        train_wo_valid(model, train_images, train_labels)
        print(f"batch: {batch+1}/{len(train_in_batches)}, train loss: {train_loss_calculator.result()}, train accuracy: {train_accuracy_calculator.result()}")
        model.train_loss_list.append(train_loss_calculator.result())
        model.train_accuracy_list.append(train_accuracy_calculator.result())

    for batch in range(len(test_in_batches)):
        test_images, test_labels = test_in_batches[batch]
        test_images = tf.transpose(test_images, [0, 3, 1, 2]) # channel first
        test(model, test_images, test_labels)
        # print(f"batch: {batch}/{len(train_in_batches)}, loss: {train_loss}, accuracy: {train_accuracy}")
        # model.loss_list.append(train_loss)
        # model.accuracy_list.append(train_accuracy)
    # test_loss, test_accuracy = test(model, test_images, test_labels)
    print(f"EPOCH SUMMARY - epoch: {epoch+1}/{config.EPOCHS}, test loss: {test_loss_calculator.result()}, test accuracy: {test_accuracy_calculator.result()}")
    model.test_loss_list.append(test_loss_calculator.result())
    model.test_accuracy_list.append(test_accuracy_calculator.result())
visualize_loss(model.train_loss_list,True)
visualize_loss(model.test_loss_list, False)
visualize_accuracy(model.train_accuracy_list, True)
visualize_accuracy(model.test_accuracy_list, False)



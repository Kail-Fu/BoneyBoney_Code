import tensorflow as tf
from visualize import visualize_accuracy, visualize_loss
import san
import config
from preprocess import preprocess_wo_valid, to_batch



# # # # --------------------- Initialize Model --------------------- # # # #
model = san.san(layers=(2, 1, 2, 4, 1), kernels=[3,5,5,5,5], num_classes=config.num_classes)
model.build(input_shape=(config.batch_size, config.channels, config.image_height, config.image_width))
model.summary()



# # # # --------------------- Load the data --------------------- # # # #
train_images, train_labels, test_images, test_labels = preprocess_wo_valid("sample_dataset")
train_images = train_images[:(len(train_images)//config.batch_size)*config.batch_size]
train_labels = train_labels[:(len(train_labels)//config.batch_size)*config.batch_size]
test_images = test_images[:(len(test_images)//config.batch_size)*config.batch_size]
test_labels = test_labels[:(len(test_labels)//config.batch_size)*config.batch_size]
train_in_batches = to_batch(train_images, train_labels) # augmentation + batch
test_in_batches = to_batch(test_images, test_labels) # augmentation + batch



# # # # --------------------- Loss Acuracy Calculators --------------------- # # # #
optimizer = tf.keras.optimizers.Adam(config.learning_rate)
loss_object = tf.keras.losses.CategoricalCrossentropy()
train_loss_calculator = tf.keras.metrics.Mean(name='train_loss')
train_accuracy_calculator = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
test_loss_calculator = tf.keras.metrics.Mean(name='test_loss')
test_accuracy_calculator = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')



# # # # --------------------- Train and Test Procedure --------------------- # # # #
@tf.function
def train(model, train_images, train_labels): 
    ''' train step '''      
    with tf.GradientTape() as tape:
        train_predictions = model(train_images, training=True) 
        loss = loss_object(y_true=train_labels, y_pred=train_predictions) # calculate loss
        
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss_calculator(loss)
    train_accuracy_calculator(train_labels, train_predictions)


@tf.function
def test(model, test_images, test_labels):
    ''' test step '''
    test_predictions = model(test_images, training=False)
    loss = loss_object(test_labels, test_predictions) # calculate loss

    test_loss_calculator(loss)
    test_accuracy_calculator(test_labels,test_predictions)    



# # # # --------------------- Training and Testing --------------------- # # # #
for epoch in range(config.num_epoch):

    # reset the cumulative loss/accuracy calculation at the start of each epoch
    train_loss_calculator.reset_states()
    train_accuracy_calculator.reset_states()
    test_loss_calculator.reset_states()
    test_accuracy_calculator.reset_states()

    # training
    for batch in range(len(train_in_batches)):
        train_images, train_labels = train_in_batches[batch]
        train_images = tf.transpose(train_images, [0, 3, 1, 2]) # channel first
        train(model, train_images, train_labels)
        print(f"batch: {batch+1}/{len(train_in_batches)}, train loss: {train_loss_calculator.result()}, train accuracy: {train_accuracy_calculator.result()}")
        # construct lists for visualization
        model.train_loss_list.append(train_loss_calculator.result())
        model.train_accuracy_list.append(train_accuracy_calculator.result())

    # run testing after training for one epoch
    # run in batches to prevent OOM
    for batch in range(len(test_in_batches)):
        test_images, test_labels = test_in_batches[batch]
        test_images = tf.transpose(test_images, [0, 3, 1, 2]) # channel first
        test(model, test_images, test_labels)
        
    print(f"EPOCH SUMMARY - epoch: {epoch+1}/{config.num_epoch}, test loss: {test_loss_calculator.result()}, test accuracy: {test_accuracy_calculator.result()}")
    # construct lists for visualization
    model.test_loss_list.append(test_loss_calculator.result())
    model.test_accuracy_list.append(test_accuracy_calculator.result())
visualize_loss(model.train_loss_list,True)
visualize_loss(model.test_loss_list, False)
visualize_accuracy(model.train_accuracy_list, True)
visualize_accuracy(model.test_accuracy_list, False)

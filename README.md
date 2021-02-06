# Cats-and-dogs-CNN-v1

The cats vs. dogs dataset that we will use isn't packaged with Keras. It was made available by Kaggle.com as part of a computer vision competition in late 2013, back when convnets weren't quite mainstream. You can download the original dataset at: https://www.kaggle.com/c/dogs-vs-cats/data

The pictures are medium-resolution color JPEGs. They look like this:

![alt text](https://r.resimlink.com/DAsk.png)

### Installing a small convnet for dogs vs. cats classification:

    model = models.Sequential()
    
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    
### Configuring the model for training:

    from keras import optimizers

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])
                  
### Data preprocessing


•Read the picture files, decode the JPEG content to RBG grids of pixels, convert these into floating point tensors, rescale the pixel values (between 0 and 255) to the [0, 1] interval.

Using ImageDataGenerator to read from directories

    from keras.preprocessing.image import ImageDataGenerator

    # All images will be rescaled by 1./255
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            # This is the target directory
            train_dir,
            # All images will be resized to 150x150
            target_size=(150, 150),
            batch_size=20,
            # Since we use binary_crossentropy loss, we need binary labels
            class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
            validation_dir,
            target_size=(150, 150),
            batch_size=20,
            class_mode='binary')
            
 
### Fitting the model

    history = model.fit(
          train_generator,
          steps_per_epoch=100,
          epochs=30,
          validation_data=validation_generator,
          validation_steps=50)
### Save model

    model.save('cats_and_dogs_small_1.h5')


### Displaying curves of loss and accuracy during training

![alt text](https://r.resimlink.com/FLN091a.jpg)

![alt text](https://r.resimlink.com/0L7Bi.jpg )

These plots are characteristic of overfitting. Our training accuracy increases linearly over time, until it reaches nearly 100%, while our validation accuracy stalls at 70-72%.

Our validation loss reaches its minimum after only five epochs then stalls, while the training loss keeps decreasing linearly until it reaches nearly 0.
Help mitigate overfitting, you can use  dropout and weight decay (L2 regularization). Also, you can data augmentation.

If you want to more:

https://www.amazon.com/Deep-Learning-Python-Francois-Chollet/dp/1617294438


from tensorflow import keras
from data_extraction import train_ds,test_ds,val_ds,class_names

from keras.callbacks import ModelCheckpoint,EarlyStopping




base_model = keras.applications.vgg16.VGG16(weights='imagenet',
                                            include_top=False,  
                                            # without dense part of the network
                                            input_shape=(250,250, 3))

# Set layers to non-trainable

for layer in base_model.layers:
    layer.trainable = False

# Add custom layers on top of the convolutional layers of VGG16

flatten = keras.layers.Flatten()(base_model.output)
dense_4096_1 = keras.layers.Dense(4096, activation='relu')(flatten)
dense_4096_2 = keras.layers.Dense(4096, activation='relu')(dense_4096_1)
output = keras.layers.Dense(2, activation='sigmoid')(dense_4096_2)

VGG16 = keras.models.Model(inputs=base_model.input,
                           outputs=output,
                           name='VGG16')


classifier = VGG16
classifier.summary()

name_to_save = f"models/classifier_VGG16.h5"

#initiating callbacks
checkpoint = ModelCheckpoint(name_to_save,
                             monitor="val_loss",
                             mode="min",
                             save_best_only=True,
                             verbose=1)

# EarlyStopping to find best model with a large number of epochs
earlystop = EarlyStopping(monitor='val_loss',
                          restore_best_weights=True,
                          patience=5, 
                          verbose=1)

callbacks = [earlystop, checkpoint]

classifier.compile(loss='categorical_crossentropy',
                   optimizer=keras.optimizers.Adam(learning_rate=0.01),
                   metrics=['accuracy'])

model = classifier.fit(
    train_ds,
    epochs=5,
    callbacks=callbacks,
    validation_data=val_ds
)

classifier.save(name_to_save)
import tensorflow as tf
import os
base_dir="/kaggle"
input_dir = os.path.join(base_dir, "input")
project_dir = os.path.join(input_dir, "binary-adjusted-pe")
dataset_dir = os.path.join(project_dir, "Binary_Adjusted")
batch_size=128
dataset = tf.keras.utils.image_dataset_from_directory(
  directory=dataset_dir,
  validation_split=0.2,
  subset="both",
  seed=123,
  image_size=(512, 512),
  color_mode="rgb",
  label_mode="binary",
  batch_size=batch_size)
AUTOTUNE=tf.data.AUTOTUNE
training_data = dataset[0].prefetch(buffer_size=AUTOTUNE)
valid_data= dataset[1].prefetch(buffer_size=AUTOTUNE)

tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
# instantiate a distribution strategy
tpu_strategy = tf.distribute.TPUStrategy(tpu)

### SETTING UP THE BASE MODEL

import keras
import tensorflow as tf
# instantiating the model in the strategy scope creates the model on the TPU
# here you can choose what model you want to train just by changing ___MODEL___ in tf.keras.applications.___MODEL____
with tpu_strategy.scope():
    base_model = tf.keras.applications.VGG19(input_shape = (512, 512, 3), include_top = False, weights = 'imagenet')
    base_model.trainable= False
    inputs = keras.Input(shape=(512, 512, 3))
    x=base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    #x = tf.keras.layers.Dropout(0.2)(x)
    layerinfo="base model to output no dropout"
    x = tf.keras.layers.Dense(1024,activation="relu")(x)
    outputs = tf.keras.layers.Dense(1,activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)



### MODEL FITTING #######
    lr=0.0005
    optimizer="Adam"
    model.compile(tf.keras.optimizers.Adam(learning_rate=lr),
              loss=tf.keras.losses.BinaryCrossentropy(), metrics=[
                  tf.keras.metrics.AUC(),
				  tf.keras.metrics.BinaryAccuracy(),
				  tf.keras.metrics.Precision(),
				  tf.keras.metrics.Recall(),
				  tf.keras.metrics.TruePositives(),
				  tf.keras.metrics.TrueNegatives(),
				  tf.keras.metrics.FalsePositives(),
				  tf.keras.metrics.FalseNegatives()])
    model.summary()

ep=20
model.summary()
working_dir=os.path.join(base_dir,"working")
MODEL_NAME="VGGnm19"
MODEL_SERIES="-1024-19FineTuning"
MODELS_PATH=os.path.join(working_dir,MODEL_NAME+MODEL_SERIES+"-ep-{epoch:04d}.h5")
working_dir
save_every_epoch=tf.keras.callbacks.ModelCheckpoint(
    MODELS_PATH,
    save_freq="epoch",
    verbose=1
)
### it set to save the model every each epoch, you can change that by removing callbacks
history = model.fit(training_data,validation_data=valid_data,epochs=ep,callbacks=[save_every_epoch], verbose=1)
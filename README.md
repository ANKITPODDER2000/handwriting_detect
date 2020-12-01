# handwriting_detect

### Model details

```python

def get_base_model():
    #input = tf.keras.layers.Input(shape = (28,28,1))
    #x = tf.keras.layers.Dense(64 , activation=tf.nn.relu)(input)
    #x = tf.keras.layers.MaxPool2D((2,2))(x)
    #x = tf.keras.layers.Dense(128 , activation=tf.nn.relu)(x)
    #x = tf.keras.layers.MaxPool2D((2,2))(x)

    #x = tf.keras.layers.Flatten()(x)
    #x = tf.keras.layers.Dense(256 , activation=tf.nn.relu)(x)
    #x = tf.keras.layers.Dense(26 , activation=tf.nn.sigmoid)(x)

    #model = tf.keras.Model(inputs = input ,outputs = x)
    #model.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001) ,
                  #loss = tf.keras.losses.categorical_crossentropy , metrics = ['acc'])
    #return model

    cls = Sequential()
    cls.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    cls.add(MaxPooling2D(pool_size=(2, 2)))
    cls.add(Dropout(0.3))
    cls.add(Flatten())
    cls.add(Dense(128, activation='relu'))
    cls.add(Dense(26, activation='softmax'))

    cls.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return cls
```

### Conver keras model into tensorflow.js model

```python
!pip install tensorflowjs
import tensorflowjs as tfjs
tfjs.converters.save_keras_model(model1 , 'model')
```


### UI of the web APP
***
<p align="center">
  <img width="600px" src="https://user-images.githubusercontent.com/50513363/100707335-873bf600-33d0-11eb-85c4-4009c5ffbf93.png" />
</p>

***

<p align="center">
  <img width="600px" src="https://user-images.githubusercontent.com/50513363/100707529-d7b35380-33d0-11eb-88d9-6515598f7cf2.png" />
</p>

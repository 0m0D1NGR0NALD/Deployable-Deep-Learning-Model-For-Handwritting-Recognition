import numpy as np
from keras.models import model_from_json
import tensorflow as tf

def init():
    json_file = open('model.json','r')
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = model_from_json(loaded_model_json)

    # Load weights into new model
    loaded_model.load_weights('model.h5')
    print('Model Loaded from Storage.')

    # Compile loaded model
    loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    # Obtain default graph
    graph = tf.get_default_graph()
    # Output model and graph
    return loaded_model, graph
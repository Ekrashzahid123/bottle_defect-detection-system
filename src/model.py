import tensorflow as tf
from tensorflow.keras import layers, models

def create_model(num_classes: int = 2, variant: str = "small", dropout: float = 0.25, input_shape=(224, 224, 3)):
    """
    Pure TensorFlow / Keras MobileNetV3 Transfer Learning Model.
    
    Why MobileNetV3 in TensorFlow?
    1. Optimized Depthwise Separable Convolutions & Squeeze-and-Excitation attention.
    2. Hard-Swish activations for rapid CPU inference in industrial deployment (<15ms).
    3. Seamless export to TFLite / SavedModel format.
    """
    if variant.lower() == "large":
        base_model = tf.keras.applications.MobileNetV3Large(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
            pooling="avg"
        )
    else:
        base_model = tf.keras.applications.MobileNetV3Small(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
            pooling="avg"
        )

    base_model.trainable = True

    inputs = layers.Input(shape=input_shape, name="image_input")
    x = tf.keras.applications.mobilenet_v3.preprocess_input(inputs)
    x = base_model(x)
    x = layers.Dropout(dropout, name="dropout_regularization")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="defect_probabilities")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=f"Bottle_MobileNetV3_{variant.capitalize()}")
    return model

import tensorflow as tf
from tensorflow.keras import layers, models

def create_tf_mobilenet_v3(num_classes: int = 2, variant: str = "small", dropout: float = 0.2, input_shape=(224, 224, 3)):
    """
    Creates a MobileNetV3 Transfer Learning model using TensorFlow / Keras.
    
    Why MobileNetV3 in TensorFlow?
    - Ultra-lightweight with Depthwise Separable Convolutions + Squeeze-and-Excitation.
    - Native TFLite export capability for real-time edge device / Raspberry Pi / Jetson deployment.
    - Pretrained on ImageNet.
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

    # Freeze base model initially or allow full fine-tuning
    base_model.trainable = True

    inputs = layers.Input(shape=input_shape)
    # MobileNetV3 expects inputs in [-1, 1] or [0, 255] depending on preprocessing;
    # tf.keras.applications.mobilenet_v3.preprocess_input handles this
    x = tf.keras.applications.mobilenet_v3.preprocess_input(inputs)
    x = base_model(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="classification_head")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=f"MobileNetV3_{variant.capitalize()}_Defect_Classifier")
    return model

import os
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'dogs_vs_cats.h5')
_model = None


def get_model():
    global _model
    if _model is None:
        try:
            import tf_keras as keras
            _model = keras.models.load_model(MODEL_PATH, compile=False)
        except ImportError:
            import tensorflow as tf
            from tensorflow.keras.layers import DepthwiseConv2D

            class CustomDepthwiseConv2D(DepthwiseConv2D):
                def __init__(self, *args, **kwargs):
                    kwargs.pop('groups', None)
                    super().__init__(*args, **kwargs)

                @classmethod
                def from_config(cls, config):
                    config = config.copy()
                    config.pop('groups', None)
                    return super().from_config(config)

            _model = tf.keras.models.load_model(
                MODEL_PATH,
                custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D},
                compile=False
            )
    return _model


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)

def predictor(img):
    model = get_model()
    predictions = model.predict(img)
    class_names = ['Cat', 'Dog']
    
    predictions_arr = np.array(predictions)
    
    # Handle single output (binary classification)
    if predictions_arr.ndim == 2 and predictions_arr.shape[1] == 1:
        raw_val = float(predictions_arr[0][0])
        # If not in [0, 1], apply sigmoid
        prob = raw_val if 0.0 <= raw_val <= 1.0 else sigmoid(raw_val)
        label = 'Dog' if prob >= 0.5 else 'Cat'
        confidence = prob if label == 'Dog' else (1.0 - prob)
    else:
        raw_logits = predictions_arr[0]
        # If values are not valid probabilities summing to ~1.0, apply softmax
        if np.any(raw_logits < 0.0) or not np.isclose(np.sum(raw_logits), 1.0, atol=1e-2):
            probs = softmax(raw_logits)
        else:
            probs = raw_logits
            
        index = int(np.argmax(probs))
        label = class_names[index] if index < len(class_names) else f"Class {index}"
        confidence = float(probs[index])
        
    return {
        "class": label,
        "confidence": f"{confidence * 100:.2f}%"
    }

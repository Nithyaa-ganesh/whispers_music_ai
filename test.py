import os
import tensorflow as tf

# Check if model files exist
print("Checking model files...")
print("genre_yamnet_nn.h5 exists:", os.path.exists("model/genre_yamnet_nn.h5"))
print("yamnet.h5 exists:", os.path.exists("model/yamnet.h5"))

# If files exist, try to load them
if os.path.exists("model/genre_yamnet_nn.h5"):
    try:
        print("Testing genre model load...")
        model = tf.keras.models.load_model("model/genre_yamnet_nn.h5")
        print("✅ Genre model loaded successfully")
    except Exception as e:
        print(f"❌ Genre model error: {e}")

if os.path.exists("model/yamnet.h5"):
    try:
        print("Testing YAMNet model load...")
        model = tf.keras.models.load_model("model/yamnet.h5")
        print("✅ YAMNet model loaded successfully")
    except Exception as e:
        print(f"❌ YAMNet model error: {e}")
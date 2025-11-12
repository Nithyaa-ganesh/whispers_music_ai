import tensorflow as tf
import tensorflow_hub as hub
import librosa
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ---------------- Paths ----------------
DATASET_PATH = r"C:\Users\DELL\Desktop\whispers_music_ai\dataset"  # Folder with 10 genre subfolders
MODEL_PATH = "model"
os.makedirs(MODEL_PATH, exist_ok=True)

GENRES = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]

# ---------------- Load YamNet ----------------
yamnet_model_handle = "https://tfhub.dev/google/yamnet/1"
yamnet_model = hub.load(yamnet_model_handle)

# ---------------- Helper Function ----------------
def extract_embedding(file_path):
    """Load audio and extract YAMNet embedding (mean over time)."""
    wav, sr = librosa.load(file_path, sr=16000)
    scores, embeddings, spectrogram = yamnet_model(wav)
    mean_embedding = tf.reduce_mean(embeddings, axis=0)
    return mean_embedding.numpy()

# ---------------- Prepare Dataset ----------------
X, y = [], []
print("Extracting embeddings from dataset...")

for genre in GENRES:
    genre_folder = os.path.join(DATASET_PATH, genre)
    for filename in os.listdir(genre_folder):
        if filename.endswith(".mp3"):
            file_path = os.path.join(genre_folder, filename)
            emb = extract_embedding(file_path)
            X.append(emb)
            y.append(genre)

X = np.array(X)
y = np.array(y)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_onehot = tf.keras.utils.to_categorical(y_encoded, num_classes=len(GENRES))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

# ---------------- Build Classifier ----------------
model = Sequential([
    Dense(256, activation='relu', input_shape=(X.shape[1],)),
    Dense(128, activation='relu'),
    Dense(len(GENRES), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ---------------- Train ----------------
model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test))

# ---------------- Save Trained Classifier ----------------
model.save(os.path.join(MODEL_PATH, "genre_yamnet_nn.h5"))
print(f"✅ Genre classifier saved: {os.path.join(MODEL_PATH, 'genre_yamnet_nn.h5')}")

# Note: YamNet does NOT need to be saved; just reload from TF Hub when needed:
# yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

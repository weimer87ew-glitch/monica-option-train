import tensorflow as tf
import numpy as np
import pandas as pd
import requests
import time
import json

BOT_API_URL = "https://api.telegram.org/bot" + "<DEIN_BOT_TOKEN>"

def send_message(chat_id, text):
    url = f"{BOT_API_URL}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print("Fehler beim Senden:", e)

def train_model():
    send_message("<DEINE_CHAT_ID>", "🚀 KI-Training gestartet...")
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    X = np.random.rand(100, 10)
    y = np.random.rand(100, 1)
    model.fit(X, y, epochs=3, verbose=1)

    model.save("model.h5")
    send_message("<DEINE_CHAT_ID>", "✅ Training abgeschlossen! Modell gespeichert.")

if __name__ == "__main__":
    while True:
        try:
            train_model()
            time.sleep(3600)  # Wiederhole Training alle 60 Minuten
        except Exception as e:
            send_message("<DEINE_CHAT_ID>", f"⚠️ Fehler im Training: {e}")
            time.sleep(300)

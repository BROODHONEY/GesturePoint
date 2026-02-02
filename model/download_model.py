# download_model.py
import urllib.request

print("Downloading hand landmarker model...")
url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
urllib.request.urlretrieve(url, 'hand_landmarker.task')
print("Model downloaded successfully!")
#Library imports
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
import os
from gtts import gTTS

try:
    os.mkdir("temp")
except:
    pass

#Loading the Model
model = load_model('traffic_signs.h5')

#Name of Classes
CLASS_NAMES = ['Speed limit (20km/h)','Speed limit (30km/h)','Speed limit (50km/h)','Speed limit (60km/h)',
              'Speed limit (70km/h)','Speed limit (80km/h)','End of speed limit (80km/h)','Speed limit (100km/h)',
              'Speed limit (120km/h)','No passing','No passing for vechiles over 3.5 metric tons',
              'Right-of-way at the next intersection','Priority road','Yield','Stop','No vechiles',
              'Vechiles over 3.5 metric tons prohibited','No entry','General caution','Dangerous curve to the left',
              'Dangerous curve to the right','Double curve','Bumpy road','Slippery road','Road narrows on the right',
              'Road work','Traffic signals','Pedestrians','Children crossing','Bicycles crossing','Beware of ice/snow',
              'Wild animals crossing','End of all speed and passing limits','Turn right ahead','Turn left ahead',
              'Ahead only','Go straight or right','Go straight or left','Keep right','Keep left','Roundabout mandatory',
              'End of no passing','End of no passing by vechiles over 3.5 metric']

#Setting Title of App
st.title("Traffic sign Prediction")
st.markdown("Upload an image of the sign")

#Uploading the dog image
sign_image = st.file_uploader("Choose an image...", type="png")
submit = st.button('Predict')
#On predict button click


def convert_text_to_speech(text, output_language="en", tld="co.in"):

    tts = gTTS(text, lang=output_language, tld=tld, slow=False)
    try:
        my_file_name = text[0:20]
    except:
        my_file_name = "audio"
    tts.save(f"temp/{my_file_name}.mp3")
    return my_file_name


if submit:


    if sign_image is not None:

        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(sign_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)



        # Displaying the image
        st.image(opencv_image, channels="BGR")
        #Resizing the image
        opencv_image = cv2.resize(opencv_image, (50,50))
        #Convert image to 4 Dimension
        opencv_image.shape = (1,50,50,3)
        Y_pred = model.predict(opencv_image)

        st.title(str(CLASS_NAMES[np.argmax(Y_pred)]))


    # if st.button("Predict"):
        pred_caption = CLASS_NAMES[np.argmax(Y_pred)]

        # convert text to speech
        result = convert_text_to_speech(pred_caption)
        # print(result)
        audio_file = open(f"temp/{result}.mp3", "rb")
        audio_bytes = audio_file.read()
        st.markdown(f"## Your audio:")
        st.audio(audio_bytes, format="audio/mp3", start_time=0)


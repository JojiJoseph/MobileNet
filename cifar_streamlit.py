import streamlit as st
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
import cv2
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


categoris = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

from model import CIFARModel

model = CIFARModel()
model.build((1,32,32,3))
model.load_weights("./cifar.h5")
st.title("CIFAR10 Test")

uploaded_file = st.file_uploader("Choose a image file", type="jpg")

if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    # Now do something with the image! For example, let's display it:
    st.image(opencv_image, channels="BGR")

    img_small = cv2.resize(opencv_image, (32,32))
    img_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB).astype('float')/127.5-1

    prediction = model.predict(img_small[None,...])

    pred = np.argmax(prediction[0])

    st.title(categoris[pred])



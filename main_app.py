# %%
import numpy as np
import streamlit as st
import cv2 
from keras.models import load_model
from keras.models import load_model
# %%

#import sys
#print(sys.executable)
#import sys
#!{sys.executable} -m pip install streamlit tensorflow keras opencv-python

#loading the model
model = load_model(r'C:\Users\ayanr\Desktop\DataScienceML\projectplant\colabmodel\plant_disease.h5')


# name of classes
CLASS_NAMES = ['Corn-Common_rust', 'Potato-Early_blight', 'Tomato-Bacterial_spot']

#setting title of app
st.title("Plant Disease Detection")
st.markdown("Upload an image of the plant leaf")


#uploading the dog image
plant_image = st.file_uploader("Choose and image...", type="jpg")
submit = st.button('Predict')

# on predict button click

if submit:

  if plant_image is not None:
    # convert the file to an opencv image
    file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    # Displaying the image
    st.image(opencv_image, channels= "BGR")
    st.write(opencv_image.shape)

    #resizing the image 
    opencv_image = cv2.resize(opencv_image, (256,256))

    #convert image to 4 dimension because the model takes a 4d  images
    opencv_image.shape = (1, 256,256,3)

    #make predictions
    Y_pred = model.predict(opencv_image)
    result = CLASS_NAMES[np.argmax(Y_pred)]
    st.title(str("This is " +result.split('-')[0]+ " leaf with " + result.split('_')[1]))
    # pip install pipreqs
    # streamlit run main_app.py
# %%

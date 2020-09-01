
from yolo_utils import *
from gtts import gTTS
from playsound import playsound
import os

'''
This is main function where execution starts and it read the input image and pre-process .
'''
if __name__ == "__main__":
    # Path of the input image

    image_path = "3.JPG"
    # Read of the input images
    image = cv2.imread(image_path)
    # Resize of the input images
    image = cv2.resize(image, (608, 608))
    # Remove the noise from color images
    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 15)
    # Call the object_recog function to get output on the input-image
    full_frame, cropped_image = object_recog(image)

    if cropped_image is not {}:
        print("I am inside the If condition. . ")
        color_prediciton = color_recognition(cropped_image)
        print(color_prediciton)
        print("I Have reached here")
        exit()
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(full_frame, color_prediciton, (10, 100), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
        mytext = "The predicted color is " + color_prediciton
        language = 'en'
        myobj = gTTS(text=mytext, lang=language, slow=False)
        cv2.imshow("full_image", full_frame)
        cv2.waitKey(1)
        myobj.save("audio.mp3")
        playsound("audio.mp3")
        os.remove("audio.mp3")
    else:
        print("Couldn't detect the signal")
        quit()



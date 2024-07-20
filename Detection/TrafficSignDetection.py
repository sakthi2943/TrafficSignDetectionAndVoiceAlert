from gtts import gTTS
import numpy as np
import cv2
import time 
from tensorflow import keras
import pyttsx3

modelPath = '../Model\model.savedmodel'
model = keras.models.load_model(modelPath)
model = keras.models.load_model(modelPath+'../03-Classification\Models\TSModel5')


# ImagesFilePath='../Dataset/traffic_Data/TEST'
# ImageNamePath=os.listdir(ImagesFilePath)

# def readImage(imagePath):
#     img = cv2.imread(ImagesFilePath+'/'+imagePath,1)
#     img = cv2.resize(img,(500,400))
#     return img

def increaseContrast(img,alpha,beta):
	img=cv2.addWeighted(img,alpha,np.zeros(img.shape,img.dtype),0,beta)
	return img

def filteringImages(img):
    img=cv2.GaussianBlur(img,(11,11),0)
    return img


def returnRedness(img):
	yuv=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
	y,u,v=cv2.split(yuv)
	return v

def threshold(img,T=150):
	_,img=cv2.threshold(img,T,255,cv2.THRESH_BINARY)
	return img 

def show(img,name):
	cv2.imshow(name,img)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

def morphology(img,kernelSize=7):
	kernel = np.ones((kernelSize,kernelSize),np.uint8)
	opening = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
	return opening

def findContour(img):
	contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	return contours

def findBiggestContour(contours):
	# if not contours:
	# 	return np.array([])
	# areas =[cv2.contourArea(cnt) for cnt in contours]
	# return contours[np.argmax(areas)]
	m=0
	c=[cv2.contourArea(i) for i in contours]
	return contours[c.index(max(c))]

def boundaryBox(img,contours):
	x,y,w,h=cv2.boundingRect(contours)
	img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
	sign=img[y:(y+h) , x:(x+w)]
	return img,sign

def preprocessingImageToClassifier(image=None,imageSize=28,mu=89.77428691773054,std=70.85156431910688):
    image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image,(imageSize,imageSize))
    image = (image - mu) / std
    image = image.reshape(1,imageSize,imageSize,1)
    return image

def predict(sign):
	img=preprocessingImageToClassifier(sign,imageSize=28)
	return np.argmax(model.predict(img))

# def predict3(sign):
# 	img=preprocessingImageToClassifier(sign,imageSize=32)
# 	return np.argmax(model.predict(img))

def process_frame(frame):
    img = np.copy(frame)
    try:
        img = filteringImages(img)
        img = returnRedness(img)
        img = threshold(img, T=155)
        img = morphology(img, 11)
        contours = findContour(img)
        big = findBiggestContour(contours)
        img, sign = boundaryBox(img, big)
        tic = time.time()
        # print("Model4 says the sign in the frame:", labelToText[predict(sign)])
        # toc = time.time()
        # print("Running Time: ", (toc - tic) * 1000, 'ms')
        # print("--------------------------------------------------------")
    except Exception as e:
        print("Error processing frame:", e)

    show(img)
#__________________________________________________________________
	

# Initialize the text-to-speech engine
engine = pyttsx3.init()
	
# Function to convert text to speech
def text_to_speech(text, lang='en'):
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save("temp.mp3")
    engine.stop()  # Stop any ongoing speech
    engine.say(text)
    engine.runAndWait()
	
#__________________________________________________________________



# if __name__ == '__main__':
# 	for i in ImageNamePath:
# 		testCase=readImage(i)
# 		img=np.copy(testCase)
# 		try:
# 			img=filteringImages(img)
# 			img=returnRedness(img)
# 			img=threshold(img,T=155)
# 			img=morphology(img,11)
# 			contours=findContour(img)
# 			big=findBiggestContour(contours)
# 			testCase,sign=boundaryBox(testCase,big)
# 			tic=time.time()
# 			print("Model4 say The Sign in Image:",labelToText[predict(sign)])
# 			toc=time.time()
# 			print("Running Time of Model4",(toc-tic)*1000,'ms')
# 			"""
# 			tic=time.time()
# 			print("Model3 say The Sign in Image:",labelToText[predict3(sign)])
# 			toc=time.time()
# 			print("Running Time of Model3",(toc-tic)*1000,'ms')
# 			"""
# 			print("--------------------------------------------------------")
# 		except:
# 			pass
		# show(testCase)
		# show(img)
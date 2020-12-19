import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
#%matplotlib inline
import keras
from keras.layers import Dense,Conv2D,MaxPooling2D, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
import cv2

############
# arguments
############

# needs 2 arguments input
# 1 (required)- inputdir = path of folder containing "train" and "valid" subfolders
# 2 (optional)- outputdir = choose output directory or use current folder

###########
# usage
###########
#python plant_diseases_detection.py dir outfile


# retreive args
if (len(sys.argv)<2 or len(sys.argv)>3) :
	exit("USAGE: python explore_small_dataset.py <inputdir> <outputdir>")
elif len(sys.argv)==2 :
	print("1 argument")
	inputdir = sys.argv[1]
	cmd = Popen(['pwd'], stdout=PIPE)
	outputdir,err = cmd.communicate()
	outputdir = outputdir.decode("utf-8").rstrip('\n')
	print(outputdir)
elif len(sys.argv)== 3 :
	print("2 arguments")
	inputdir = sys.argv[1]
	outputdir = sys.argv[2]

################################
# definition des jeux de donnees
################################

#chemain train et valid
#train_path = r"D:\Project_final_version\Small_test\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\train"
train_path = inputdir+"/train"

#test_path = r"D:\Project_final_version\Small_test\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\valid"
test_path = inputdir+"/valid"

train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=25,horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory(train_path,
	target_size =[224,224],
	batch_size=32,
	class_mode='categorical' )

test_set = test_datagen.flow_from_directory(test_path,
	target_size =[224,224],
	batch_size=32,
	class_mode='categorical' )


#########################
# definition du modele
#########################

#VGG16 model
base_model=VGG16(include_top=False,input_shape=(224,224,3))
base_model.trainable=False
model = Sequential()

conv1 = Conv2D(filters = 32,kernel_size=(3,3),input_shape=(224,224,3),padding='same',activation='relu')#,padding='same')
maxpool1 = MaxPooling2D()#pool_size=(32,32))#,padding='same')
conv2=Conv2D(filters=32,kernel_size=(3,3),padding='same',activation='relu')#,padding='same')
maxpool2= MaxPooling2D()#pool_size=(15,15),padding='same')
flatten = Flatten()
dense1 = Dense(units=256)#,activation='relu')
dense2 = Dense(units=38,activation='softmax')

#first__test___
model.add(base_model)
model.add(flatten)
model.add(dense2)
#seconde_test
#model.add(conv1)
#model.add(maxpool1)
#model.add(conv2)
#model.add(maxpool2)
#model.add(flatten)
#model.add(dense1)
#model.add(dense2)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])

history = model.fit_generator(generator=train_set,
	validation_data=test_set,
	epochs=6,
	steps_per_epoch = train_set.samples//32,
	validation_steps= test_set.samples//32)

model.summary()

dic_differente_classes = train_set.class_indices
classes_name = list(dic_differente_classes.keys())


#################################
# visualisation des predictions
#################################

from keras.preprocessing import image
import matplotlib.pyplot as plt
import os
image_path = inputdir+"/test/test"
ii=1
for image_name in os.listdir(image_path):
	#image_path = "/media/PGM/bioinfo/Jennifer/DS/novembre/Small_test_data/test/test/AppleCedarRust2.jpg"
	new_img = image.load_img(os.path.join(image_path,image_name), target_size=(224, 224))
	img = image.img_to_array(new_img)
	img = np.expand_dims(img, axis=0)
	img = img/255
	# print("Following is our prediction:")
	prediction = model.predict(img)
	# decode the results into a list of tuples (class, description, probability)
	3
	# (one such list for each sample in the batch)
	d = prediction.flatten()
	j = prediction.max()
	for index,item in enumerate(d):
		if item == j:
			class_name = classes_name[index]
			proba = 100*item
	#ploting image with predicted class name
	plt.figure(figsize = (40,40))
	plt.subplot(33,1,ii)
	plt.imshow(new_img)
	plt.axis('off')
	plt.title('real name : '+image_name +' prediction : ' + class_name + '  '+str(proba))
	plt.show()
	ii+=1
plt.savefig(outputdir+"/VGG16_test_1.jpg",format='jpg')

print(model.predict(img))
print(model.predict(img).flatten().max())



from subprocess import *
import os.path, sys, re
import pandas as pd
import numpy as np
import matplotlib
#matplotlib.use('Agg')
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
# 2 (required)- inputcsv = path to csv file generated by explor_dataset.py script
# 3 (optional)- outputdir = choose output directory or use current folder


###########
# usage
###########
#python diseases_detection.py <inputcsv>

# retreive args
if (len(sys.argv)<3 or len(sys.argv)>4) :
	exit("USAGE: python disease_detection.py <inputdir> <inputcsv> <outputdir>")
elif len(sys.argv)==3 :
	print("2 arguments")
	inputdir = sys.argv[1]
	cmd = Popen(['pwd'], stdout=PIPE)
	outputdir,err = cmd.communicate()
	outputdir = outputdir.decode("utf-8").rstrip('\n')
	inputcsv = sys.argv[2]
else :
	print("3 arguments")
	inputdir = sys.argv[1]
	outputdir = sys.argv[3]
	inputcsv = sys.argv[2]


df = pd.read_csv(inputcsv,index_col=0,sep="\t")

#pour pouvoir utiliser 'healthy' comme une classe => conversion du type en string
df = df.astype({'healthy':'str'})


################################
# definition des jeux de donnees
################################

####
##chemin train et valid

train_path = inputdir+"/train"

test_path = inputdir+"/valid"

###
#small_test_data entier
#data_train = df[(df['dataset']=='train')]
#data_test = df[(df['dataset']=='valid')

#test fonctionnel sur pomme uniquement pour calculer plus vite
data_train = df[(df['dataset']=='train')&(df['specie']=='Apple')]
data_test = df[(df['dataset']=='valid')&(df['specie']=='Apple')]

data_train = data_train.reset_index(drop=True)
data_test = data_test.reset_index(drop=True)


#data_train["class"] = data_train['healthy']
#data_test["class"] = data_test['healthy']

train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=25,horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

####
#data generator
train_set = train_datagen.flow_from_dataframe(dataframe=data_train,
	target_size =[224,224],
	batch_size=32,
	x_col = 'file',
	#y_col = 'specie',
	y_col = 'healthy',
	#y_col = 'disease',
	#y_col = ('specie','disease'),
	class_mode='categorical')

test_set = train_datagen.flow_from_dataframe(dataframe=data_test,
	target_size =[224,224],
	batch_size=32,
	x_col = 'file',
	#y_col = 'specie',
	y_col = 'healthy',
	#y_col = 'disease',
	#y_col = ('specie','disease'),
	class_mode='categorical')


#########################
# definition du modele
#########################

#VGG16 model
base_model=VGG16(include_top=False,input_shape=(224,224,3))
base_model.trainable=False
model = Sequential()

conv1 = Conv2D(filters = 32,kernel_size=(3,3),input_shape=(224,224,3),padding='same',activation='relu')
maxpool1 = MaxPooling2D()#pool_size=(32,32))#,padding='same')
conv2=Conv2D(filters=32,kernel_size=(3,3),padding='same',activation='relu')
maxpool2= MaxPooling2D()#pool_size=(15,15),padding='same')
flatten = Flatten()
dense1 = Dense(units=256)#,activation='relu')

#all small_dataset
#dense2 = Dense(units=38,activation='softmax')

#only apple in small_dataset to predict disease class
#dense2 = Dense(units=4,activation='softmax')
#only apple in small_dataset to predict healthy class
dense2 = Dense(units=2,activation='softmax')

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
	epochs=1,
	steps_per_epoch = train_set.samples//32,
	validation_steps= test_set.samples//32)

model.summary()

dic_differente_classes = train_set.class_indices
classes_name = list(dic_differente_classes.keys())


#################################
# visualtisation des predictions
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
plt.savefig(outputdir+"/VGG16_flow_from_dataframe_test_1.jpg",format='jpg')

print(model.predict(img))
print(model.predict(img).flatten().max())


#########################
# evaluation du modele
#########################

#courbes pour voir loss et acc en fonction des epochs
train_acc = history.history['acc']
val_acc = history.history['val_acc']

train_loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(10,10))
plt.subplot(121)
plt.xlabel("Epochs")
plt.ylabel("loss")
plt.title("Model loss by epoch")
plt.plot(train_loss)
plt.plot(val_loss)
plt.legend(['train','test'])

plt.subplot(122)
plt.xlabel("Epochs")
plt.ylabel("acc")
plt.title("Model acc by epoch")
plt.plot(train_acc)
plt.plot(val_acc)
plt.legend(['train','test'])
plt.show();
plt.savefig(outputdir+"/VGG16_flow_from_dataframe_test_1_accuracy_curve.jpg",format='jpg')


def convert_image(X):
    X_img=[]
    for image in X:
        # Load image
        img=cv2.imread(image)
        # Resize image
        img=cv2.resize(img,(224,224))
        # for the black and white image
        if img.shape==(224, 224):
        	img = image.img_to_array(img)
        	img = np.expand_dims(img, axis=0)
        	img = img/255
        	#img=img.reshape([224,224,1])
            #img=np.concatenate([img,img,img],axis=2)  
        # cv2 load the image BGR sequence color (not RGB)
        X_img.append(img[...,::-1])
    return np.array(X_img)

X_train_img = convert_image(data_train.file)
y_train = data_train['disease']

X_test_img = convert_image(data_test.file)
y_test = data_test['disease']

#scores
#X_test= ? et Y_test ?
scores = model.evaluate(X_test_img,y_test)
print("Fonction de perte : %s" % scores[0])
print("Précision du modèle : %s" % scores[1])


test_pred = model.predict(X_test_img)
test_pred_class = test_pred.argmax(axis=1)
y_test_class = y_test.argmax(axis=1)

print(metrics.classification_report(y_test_class,test_pred_class))


################
# matrice de confusion

cnf_matrix = metrics.confusion_matrix(y_test_class, test_pred_class)
classes = range(0,3)

plt.figure()
plt.imshow(cnf_matrix, interpolation='nearest',cmap='Blues')
plt.title("Matrice de confusion")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)

for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
    plt.text(j, i, cnf_matrix[i, j],
             horizontalalignment="center",
             color="white" if cnf_matrix[i, j] > ( cnf_matrix.max() / 2) else "black")

plt.ylabel('Vrais labels')
plt.xlabel('Labels prédits')
plt.show()

plt.savefig(outputdir+"/VGG16_flow_from_dataframe_test_1_matrice_confusion.jpg",format='jpg')
################

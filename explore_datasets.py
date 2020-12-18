from subprocess import *
import os.path, sys, re
import pandas as pd
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

############
# arguments
############

# needs 3 arguments input
# 1 (required)- inputdir = path of folder containing "train" and "valid" subfolders
# 2 (required)- outfile = file name to create 
# 3 (optional)- outputdir = choose output directory or use current folder

###########
# usage
###########
#python explore_small_dataset.py dir outfile


# retreive args
if (len(sys.argv)<3 or len(sys.argv)>4) :
	exit("USAGE: python explore_small_dataset.py <inputdir> <outfile> <outputdir>")
elif len(sys.argv)==3 :
	print("2 arguments")
	inputdir = sys.argv[1]
	cmd = Popen(['pwd'], stdout=PIPE)
	outputdir,err = cmd.communicate()
	outputdir = outputdir.decode("utf-8").rstrip('\n')
	outfile = sys.argv[2]
	print(outputdir)
elif len(sys.argv)== 4 :
	print("3 arguments")
	inputdir = sys.argv[1]
	outputdir = sys.argv[3]
	outfile = sys.argv[2]


#dossier du jeu de donnees
#inputdir = "/home/jennifer/DataScience/projet_pyplant/OneDrive-2020-12-16/novembre/Small_test_data/"
try:
	os.path.isdir(inputdir)
except IOError:
	print("ERROR: couldn't open source directory " + source + "\n")

#fichier de sortie format csv 
#dataset_explore_file = "/home/jennifer/DataScience/projet_pyplant/decembre/New_Plant_Diseases_Dataset_Augmented_Small_Dataset.csv"
dataset_explore_file = outputdir+"/"+outfile

try:
	out_file = open(dataset_explore_file, "w")
except IOError:
	print("ERROR: couldn't open file " + dataset_explore_file + " for writing\n")


#header
out_file.write("ID\tspecie\thealthy\tdisease\tfile\tdataset\timg_size\n")
#out_file.write("ID,specie,healthy,disease,file,dataset\n")

#les 2 sous-jeux disponibles au meme format
dataset = ['train', 'valid']

#iterateur
i = 0

for d in dataset :
	path = inputdir+d
	print("jeu de donnees etudie : %s" % path)

	#regarde les dossiers dans train ou valid
	cmd = Popen(['ls',path], stdout=PIPE)
	out_dir, err = cmd.communicate()
	out_dir = out_dir.decode("utf-8")

	if len(out_dir) > 0: 
		print("OUT_DIR: %s" % out_dir)
		list_out_dir = out_dir.rstrip('\n').split('\n')
		#list_out contient la liste des dossiers dans le dossier train ou valid
		print(list_out_dir)
		for dirname in (list_out_dir) :
			#decoupe le nom du dossier et recupere le nom de l'espece + la maladie ou healthy
			print(dirname)
			specie = dirname.split('___')[0]
			disease = dirname.split('___')[1]
			if disease == "healthy" :
				healthy = 1
				disease = 'None'
			else :
				healthy = 0

			#regarde les images dans les dossiers
			sub_path = path+"/"+ dirname
			print("PATH : %s" % sub_path)
			#path = path.encode()
			print(type(path))
			print(str(len(path)))
			cmd = Popen(['ls',sub_path], stdout=PIPE)
			out_subdir, err = cmd.communicate()
			out_subdir = out_subdir.decode("utf-8")

			if len(out_subdir) > 0:
				list_subdir_img = out_subdir.rstrip("\n").split("\n")
				#list_subdir_img contient la liste des images dans un sous-dossier de train ou valid

				for image in list_subdir_img :
					i = i+1
					path_img = sub_path+"/"+image
					img_size = str(plt.imread(path_img).shape)
					out_file.write(str(i)+"\t"+specie+"\t"+str(healthy)+"\t"+disease+"\t"+path_img+"\t"+d+"\t"+img_size+"\n")
	else :
		sys.exit

#out_file.close()


#dossier test non organise par especes

test_dataset_explore_file = outputdir+"/test_dataset_"+outfile

try:
	out_file_test = open(test_dataset_explore_file, "w")
except IOError:
	print("ERROR: couldn't open file " + test_dataset_explore_file + " for writing\n")
#jeu de test
species = ["Apple","Corn","Potato","Tomato"]
d = "test"
path = inputdir+d
print("jeu de donnees etudie : %s" % path)

#regarde les dossiers dans train ou valid
cmd = Popen(['ls',path], stdout=PIPE)
out_dir, err = cmd.communicate()
out_dir = out_dir.decode("utf-8")

print("OUT_DIR: %s" % out_dir)
list_out_dir = out_dir.rstrip('\n').split('\n')
#list_out contient la liste des dossiers dans le dossier train ou valid
print(list_out_dir)
for dirname in (list_out_dir) :
	#decoupe le nom du dossier et recupere le nom de l'espece + la maladie ou healthy
	print(dirname)
	#regarde les images dans les dossiers
	path = path +"/"+ dirname
	print("PATH : %s" % path)

	cmd = Popen(['ls',path], stdout=PIPE)
	out_img, err = cmd.communicate()
	out_img = out_img.decode("utf-8")

	if len(out_img) > 0:
		list_subdir_img = out_img.rstrip("\n").split("\n")
		for image in list_subdir_img :
			print(image)
			for s in species :
				if re.match(s,image) :
					specie = s
					disease = image.replace(s,"").replace(".JPG","")

					if re.match("healthy",disease,re.IGNORECASE):
						healthy = 1
						disease = 'None'
					else :
						healthy = 0

					print("file:%s healthy=%s"%(image,str(healthy)))

			i = i+1
			path_img = path+"/"+image
			img_size = str(plt.imread(path_img).shape)
			out_file_test.write(str(i)+"\t"+specie+"\t"+str(healthy)+"\t"+disease+"\t"+path_img+"\t"+d+"\t"+img_size+"\n")

			
out_file.close()
out_file_test.close()


#dataviz
df = pd.read_csv(dataset_explore_file,sep="\t")

nb_images = len(df)
nb_img_per_specie = df.specie.value_counts()
nb_species = len(nb_img_per_specie)

print("Nombre total d'images = " + str(nb_images) + "\n")
print("Nombre d'especes = " + str(nb_species) + "\n")
print("Nombre d'images par espece : \n" + str(nb_img_per_specie))

print("\n\n Nombre de dataset = " + str(len(df.dataset.value_counts())) + "\n")
print("Nombre d'image par dataset : \n" + str(df.dataset.value_counts())+"\n")

#barplot : nombre d'images par jeu de donnees
plt.figure(figsize = (10, 10))
df.dataset.value_counts().plot(kind='bar',label=None,color='green')
plt.title("Images par jeu de donnees")
plt.legend()
plt.savefig(outputdir+"/img_per_dataset_bar.jpg",format='jpg')

#barplot : nombre d'images par specie
plt.figure(figsize = (10, 10))
df.specie.value_counts().plot(kind='bar', label=None, color='green')
plt.title("Images par espece")
plt.legend()
plt.savefig(outputdir+"/img_per_species_bar.jpg",format='jpg')


#barplot : nombre d'images par specie
plt.figure(figsize = (10, 10))
plt.bar(range(len(df.specie.value_counts())), df.specie.value_counts(), color = 'green');
plt.title("Nombre d'images par espece")
plt.xticks(range(len(df.specie.value_counts())),
	['Tomato','Apple','Corn_maize','Grape','Potato','Pepper_bell','Strawberry','Peach','Cherry','Soybean','Orange','Blueberry','Raspberry','Squash'],
	rotation='vertical')
plt.xlabel('Espece')
plt.ylabel("Nombre d'images")
plt.legend()
plt.savefig(outputdir+"/img_per_specie_bar2.jpg",format='jpg')

#barplot : nombre d'image par maladie groupe par espece
plt.figure(figsize = (40, 40))
print(str(df.disease.groupby(df.specie).value_counts()))
df.disease.groupby(df.specie).value_counts().plot(kind='bar',
	color=['yellow','yellow','yellow','yellow','darkslateblue','firebrick','firebrick','gold','gold','gold','gold','greenyellow','greenyellow','greenyellow','greenyellow','orange','peachpuff','peachpuff','brown','brown','khaki','khaki','khaki','lightcoral','sandybrown','darkolivegreen','deeppink','deeppink','red','red','red','red','red','red','red','red','red','red'])
#	hatch=['','/','','','/','/','','','','/','','','','','/','','','/','/','','','','/','/','/','','/','','','/','','','','','','','',''])
plt.title("Nombre d'image par maladie par espece")
plt.legend()
plt.savefig(outputdir+"/img_per_disease_per_specie_bar.jpg",format='jpg')

#barplot : nombre de maladie par espece
plt.figure(figsize = (20, 20))
print(str(df.disease.groupby(df.specie).value_counts()))
#df.specie.groupby(df.disease).len().plot(kind='bar')
#pd.crosstab(df.specie,df.disease).plot(kind='bar')
plt.title("Nombre de maladie par espece")
plt.legend()
plt.savefig(outputdir+"/nb_disease_per_specie_bar.jpg",format='jpg')


disease_specie_crosstab = pd.crosstab(df.specie,df.disease)
disease_specie_crosstab.to_csv(outputdir+"/crosstab_specie_disease.csv")
print(str(pd.crosstab(df.specie,df.disease)))


import shutil
import os


main = '/home/mb/Desktop/Computer_Vision/Kaggle_Carvana_Image_Segmentation/models-master/research/deeplab/datasets/Carvana/Imagesets/'
source = '/home/mb/Desktop/Computer_Vision/Kaggle_Carvana_Image_Segmentation/models-master/research/deeplab/datasets/Carvana/SegmentationClass/'
dest1 = '/home/mb/Desktop/Computer_Vision/Kaggle_Carvana_Image_Segmentation/models-master/research/deeplab/datasets/Carvana/Misc/'


files = os.listdir(source)

# Moving gif files to another folder
for f in files:
	print(f)	
	if (f.endswith('gif')):
        	shutil.move(source+f, dest1)


# Renaming the files to ensure the names match for input training files and their masks

for f in files:
	new_f = f.replace('_mask','')
	print ('Replacing ', f, 'with ', new_f, '...')
	os.rename(source+f, source+new_f)
    

# Creating text files with train and validation file names

text_file = open(main + "trainval.txt", "w")

for f in files:
	name = f[:-4]
	text_file.write(name + '\n')
	
text_file.close()




train = open(main + "train.txt", "w")
val = open(main + "val.txt", "w")

n = len(files)
train_samples = int(0.75*n)
counter = 0

for f in files:
	name = f[:-4]
	if (counter < train_samples):
		train.write(name + '\n')
		counter = counter + 1
	else:
		val.write(name + '\n')
	
train.close()
val.close()






















### Carvana Image Segmentation Challenge ###

This project attempts to tackle the Image Segmentation challenge posted on Kaggle by Carvana. For the purpose of this project, three architectures were chosen, U-Net, DeepLabV3+ and MobileNetV2. U-Net was chosen due to its popularity among the data science community for this type of task. DeepLabV3+, which is Google's state of the art architecture for semantic image segmentation task at the time of the project was also explored. Finally, a smaller and faster architecture suitable for mobile devices, MobileNetV2 was looked into. 


# Dataset #

The dataset for this project can be downloaded from the following link:
https://www.kaggle.com/c/carvana-image-masking-challenge/data


# Requirements #

- cv2
- h5py
- matplotlib
- numpy
- pandas
- pillow
- sklearn
- tensorflow
- tqdm


# U-Net Model #

Creating a U-Net model is pretty straightforward. 

1. First, place '*train*', '*train_masks*' and '*test*' data folders from the Kaggle dataset in the '*input*' folder. Also place the 'sample_submission.csv' and 'train_masks.csv' files into this folder. 

2. Next, we need to convert the training mask GIF files to PNG files. For this, go to the command prompt and to the train_mask folder and enter the following command:

` mogrify -format png *.gif` 

3. Next, go to the param.py file and add parameter values for input image size(into the model), epochs and batch size. Note that smaller the input size and lesser the number of epochs, faster will be the training though it will be at the expense of performance. For reference, it took around 27 hours to train a 128 size model with 10 epochs on my Intel i7 machine with 10GB RAM.


4. With these steps done, run the following command `python train.py` from the U-Net folder directory to train the model. 

5. Finally, to to make predictions on test data and generate submission, run the following command:

`python test_submit.py`


# DeepLabV3+ Model #

To create a DeepLab model, we need to import the core files from Google's research repository on Github. To do this, clone the folder from the following link:
https://github.com/tensorflow/models

1. To start with, create a new folder named Carvana to hold your data files as follows:
models-master/research/deeplab/datasets/Carvana

This folder should contain the following sub-folders:
- JPEGImages: For the training images. File Format - .jpg 
- SegmentationClass: For the masks. File Format - .png. Note that files in this folder should have the exact same name as files in the JPEG Images folder. 
- Imagesets: This folder contains 3 files: train.txt (Names of training set files), val.txt (Names of validation set files) and trainval.txt (Names of both train and validation set files)
- tfrecord: Contains tfrecord files which are used in building models. These will be generated in the next steps shortly.

2. a. If you are using the same mask folder as that used for U-Net model creation for SegmentationClass, some modifications like removing .gif images and renaming files will be required. For this go to the command prompt, change your directory to deeplab and then run the scripts available in deeplab_preprocess.py. This file will also help you create the text files for Imagesets folder. 

2. b. Copy the two files from DeepLabV3+ folder, namely, deeplab_preprocess.py and deeplab_test_submit.py, into the main deeplab folder.

3. The next task is to convert the raw data into tfrecords. For this, run the following command from the dataset directory:
`python build_voc2012_data.py \
  --image_folder="${IMAGE_FOLDER}" \
  --semantic_segmentation_folder="${SEMANTIC_SEG_FOLDER}" \
  --list_folder="${LIST_FOLDER}" \
  --image_format="jpg" \
  --output_dir="${OUTPUT_DIR}"`

where,
${IMAGE_FOLDER} - Path to the JPEGImages folder
${SEMANTIC_SEG_FOLDER} - Path to the SegmentationClass folder
${LIST_FOLDER} - Path to the Imagesets folder
${OUTPUT_DIR} - Path to the tfrecord folder


4. After creating the tfrecords, the next task is to make modifications to the data_generator.py file present in the dataset folder. Add the following to the file:

_CARVANA_INFORMATION = DatasetDescriptor(
                       splits_to_sizes={
                       'train': 3816,
                       'val': 1272,
                        },
                        num_classes=2,
                        ignore_label=255,
                        )

Also modify the code for the dataset information so that it looks something like this:

_DATASETS_INFORMATION = {
    'cityscapes': _CITYSCAPES_INFORMATION,
    'pascal_voc_seg': _PASCAL_VOC_SEG_INFORMATION,
    'ade20k': _ADE20K_INFORMATION,
    'carvana': _CARVANA_INFORMATION
}


5. Next, we need to make some modifications to the train.py file. If you want to re-use all the trained weights except the logits(since the num_classes may be different), set initialize_last_layer=False and last_layers_contain_logits_only=True in the file.


6. Next, we need to download the pre-trained model. For this, go to the model_zoo.md file inside the g3doc folder and download the tar file for the xception_65 model from the link provided. Extract the files to a folder named checkpoint in the deeplab directory. Also, create a file checkpoint.txt with the following contents inside this folder:

model_checkpoint_path: "model.ckpt"
all_model_checkpoint_paths: "model.ckpt"

7. We are finally ready to start the training of the model. Go to the command prompt and into the research directory and enter the following commands:

export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/deeplab
python deeplab/train.py \
       --logtostderr \
       --training_number_of_steps=300 \
       --train_split="train" \
       --model_variant="xception_65" \
       --atrous_rates=6 \
       --atrous_rates=12 \
       --atrous_rates=18 \
       --output_stride=16 \
       --decoder_output_stride=4 \
       --train_crop_size=513 \
       --train_crop_size=513 \
       --train_batch_size=2 \
       --dataset="carvana" \
       --tf_initial_checkpoint='/models-master/research/deeplab/checkpoint/deeplabv3_cityscapes_train/model.ckpt' \
       --train_logdir='/models-master/research/deeplab/logdir/train' \
       --dataset_dir='/models-master/research/deeplab/datasets/Carvana/tfrecord'

Note that complete paths must be provided for the three directories. Feel free to modify parameters like the number of steps and the train crop size. For reference, with the above parameters, it took around 7 hours on my machine to train the model. Higher values of parameters could be used to improve performance.

8. Once training is complete, the final step is to make predictions on test data and create the submission file. For this, create a folder deeplab_model inside the logdir/train folder with the following files:

- frozen_inference_graph.pb [Available in the checkpoint folder]
- model.ckpt-300.data-00000-of-00002
- model.ckpt-300.data-00001-of-00002
- model.ckpt-300.index
- model.ckpt-300.meta

Next create a tar file of this folder. Now, open the test_submit_deeplab.py and enter the complete path at the 3 places for the files mentioned. Finally, run the file from the command prompt using the following command:

` python test_submit_deeplab.py `


# MobileNetV2 #

The steps for MobileNetV2 are similar to the ones described for DeepLabV3+. If you didnot build the DeepLabV3+ model, follow the steps mentioned above and download the mobilenet model trained on Cityscape dataset in Step 6 instead of Xception_65. 

If you followed the steps above to create a DeepLabV3+ model, you just need to make the following changes to create a MobileNetV2 model:

1. In Step 6, download the MobileNetV2 model.

2. In step 7, enter the following command to train the model:

python deeplab/train.py \
       --logtostderr \
       --training_number_of_steps=1000 \
       --train_split="train" \
       --model_variant="mobilenet_v2" \
       --train_crop_size=513 \
       --train_crop_size=513 \
       --train_batch_size=2 \
       --dataset="carvana" \
       --tf_initial_checkpoint='/models-master/research/deeplab/checkpoint/deeplabv3_mnv2_cityscapes_train/model.ckpt' \
       --train_logdir='/models-master/research/deeplab/logdir/train/mobilenet' \
       --dataset_dir='/models-master/research/deeplab/datasets/Carvana/tfrecord' 

3. Complete step 8 using files from train/mobilenet folder. Rest as above.

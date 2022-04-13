# Chest X-Ray Image Classification with Deep Learning

**Author: Kamile Yagci**

**Blog URL: https://kamileyagci.github.io/**

## Overview

In this study, I analyze the Chest X-ray Images of pediatric patients in order to identify whether or
not they have pneumonia. I will apply Image Classification with Deep Learning using the
Convolutional Neural Networks (CNN).

## Business Problem

The Baylor Medical Center hired me to improve the accuracy in pneumonia diagnosis on pediatric
patients. The study will use the chest X-ray Images of pediatric patients and do the image
classification identifying whether the X-ray shows pneuomia or not. The outcome of this study will
not only be used at Baylor Centers, but also in partner medical clinics in Africa, where the medical
staff is limited. The automated identification system will provide early diagnosis of pediatric
patients, so the treatment can start as soon as possible. Moreover it will decrease the human errors
in pneumonia diagnosis.

## Outline


## Data

The dataset 'Chest X-Ray Images (Pneumonia)' is downloaded from Kaggle. 

The data contains the chest x-ray images of pedicatric patients from one to five years old, collected at Guangzhou Women and Children’s Medical Center.

The diagnosis on chest x-ray images have three types: Normal, Bacterial pneumonia and Viral pneumonia. The image below show a sample for each type.

<img src="/images/chestxray_images_samples.png" width=800/>

The dataset contains 5856 chest x-ray image files. They are labeled in two categories: NORMAL and PNEUMONIA. The number of NORMAL image samples is 1583, and the number of PNEUMONIA image samples is 4273. Bacterial pneumonia and Viral pneumonia samples are combined under label PNEUMONIA.

The original dataset downloaded from Kaggle distributed the data in three directories: Train, Validation and Test. However, the number of files in validation directory was small and insufficient. Thereofore, I redistributed the data with ~70% train, ~15% validation and 15% test. The table shows the number of images files in each directory per label.

| | Normal | PNEUMONIA | ALL |
| :- | -: | :-: | :-: |
| Train | 1107 | 2991 | 4098 
| Validation| 238 | 641 | 879 
| Test | 238 | 641 | 879
| All | 1583 | 4273 | 5856


## Methods

In this study, I applied Image Classification with Deep Learning using the Convolutional Neural Networks (CNN) on chest x-ray images. Since the data labels has two classes, NORMAL and PNEUMONIA, this is a binary image classification.


## Analysis and Results

My main challenge in this study is the computing power, since I am using an old Macbook Pro. When training with various models, I shouldn't overload my computer and need to keep the training time short. For this purpose, I train the model with the following settings: 

* Use a subset of data during training: 960 images (23%) for training and 320 images (36%) for validation. I controlled the number of images by 'steps_per_epoch' parameter while training. 

* Use small target_size when loading the images: (64x64)

* Use relatively small batch_size due to small dataset when loading the images: 20  (batch_size=20 yields better performance than batch_size=32 with subset data, according to my analysis)

* Run the training with small number of epochs: 30

With these parameter selections, model training takes about 10 minutes.

The whole training and testing data is used for model evaluations.

After final model is determined, I run the final model on whole dataset.


### Baseline Model

The features of my baseline model:
* 6 CNN layers
* 2 Dense layers
* Activation function (except out put layer): 'relu'
* Output layer activation faunction: 'sigmoid'

Compile with:
* loss='binary_crossentropy',
* optimizer= 'sgd'
* metrics='acc'

The evaluation results:

| Baseline Model Evaluation |


| | Accuracy | Loss |
| :- | -: | :-: |
| Train | 1107 | 2991 |
| Test | 238 | 641 |



## Conclusions


## Next Steps

* Even though, the model performance is good. There is always a room for improvement.

* In this study, my main limitation was computing power.
    * I did most of the model training in a subset of data. (~20% of training and validation)
    * Used (64x64) images instead of (128x128) or (256x256).
    * Used batch_size=20 instead of 32, beacuse of the small sample size.
    * Ran with a small epoch number(30).
    
* I would like run all steps of this analysis with whole dataset on a powerful computer, or grid
system. In my analysis, I did run on the whole data just once. And it didn't give the best
performance, mainly due to overfitting. There might also be other issues. I didn't have power to
investigate and tune the model with whole dataset.

* I would like to study augmentation more. Theoratically, it should improve the model
performance. However, it didn't on my subset data. Why? Would it yield better performance in
whole data? I would like to investigate this more.



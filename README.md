# Chest X-Ray Image Classification with Deep Learning

**Author: Kamile Yagci**

**Blog URL: https://kamileyagci.github.io/**

## Overview

In this study, I analyze the Chest X-ray Images of pediatric patients in order to identify whether or
not they have pneumonia. I will apply Image Classification wih Deep Learning using the
Convolutional Neural Networks (CNN).

## Business Problem

The Baylor Medical Center hired me to improve the accuracy in pneumonia diagnosis on pediatric
patients. The study will use the chest X-ray Images of pediatric patients and do the image
classification identifying whether the X-ray shows pneuomia or not. The outcome of this study will
not only be used at Baylor Centers, but also in partner medical clinics in Africa, where the medical
staff is limited. The automated identification system will provide early diagnosis of pediatric
patients, so the treatment can start as soon as possible. Moreover it will decrease the human errors
in pneumonia diagnosis.

## Data

The dataset 'Chest X-Ray Images (Pneumonia)' is downloaded from Kaggle. 

The data contains the chest x-ray images of pedicatric patients ages from one to five years old, collected at Guangzhou Women and Children’s Medical Center.

The diagnosis on chest x-ray images have three types: Normal, Bacterial pneumonia and Viral pneumonia. The image below show a sample for each type.

<img src="/images/chestxray_images_samples.png" width=800/>

The dataset contains 5856 chest x-ray image files. They are labeled in two categories: NORMAL and PNEUMONIA. The number of NORMAL image samples is 1583, and the number of PNEUMONIA image samples is 4273. Bacterial pneumonia and Viral pneumonia samples are combined under label PNEUMONIA.

The original dataset downloaded from Kaggle distributed the data in three directories: Train, Validation and Test. However, the number of files in validation directory was small and insufficient. Thereofore, I redistributed the data with ~70% train, ~15% validation and 15% test. 

| | Normal | PNEUMONIA | ALL |
| :- | -: | :-: | :-: |
| Train | 1107 | 2991 | 4098 
| Validation| 238 | 641 | 879 
| Test | 238 | 641 | 879
| All | 1583 | 4273 | 5856




Figure S6. Illustrative Examples of Chest X-Rays in Patients with Pneumonia, Related to Figure 6
The normal chest X-ray (left panel) depicts clear lungs without any areas of abnormal opacification in the image. Bacterial pneumonia (middle) typically exhibits a focal lobar consolidation, in this case in the right upper lobe (white arrows), whereas viral pneumonia (right) manifests with a more diffuse ‘‘interstitial’’ pattern in both lungs.

The dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (Pneumonia/Normal). There are 5,863 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal).

Chest X-ray images (anterior-posterior) were selected from retrospective cohorts of pediatric patients of one to five years old from Guangzhou Women and Children’s Medical Center, Guangzhou. All chest X-ray imaging was performed as part of patients’ routine clinical care.

For the analysis of chest x-ray images, all chest radiographs were initially screened for quality control by removing all low quality or unreadable scans. The diagnoses for the images were then graded by two expert physicians before being cleared for training the AI system. In order to account for any grading errors, the evaluation set was also checked by a third expert.


## Methods


## Results


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



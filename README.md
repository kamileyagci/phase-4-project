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



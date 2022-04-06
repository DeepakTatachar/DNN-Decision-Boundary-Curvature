# Deep Learning Decision Boundary Curvature Reference

This repository has code that calculates the decison boundary curvature for a deep neural net. The curvature calculation based on [Robustness via curvature regularization, and vice versa](https://openaccess.thecvf.com/content_CVPR_2019/papers/Moosavi-Dezfooli_Robustness_via_Curvature_Regularization_and_Vice_Versa_CVPR_2019_paper.pdf) .


Requirements
* Numpy
* Pytorch


How to run
----------
Create a folder "./pretrained/\<dataset name\>" and "./pretrained/\<dataset name\>/temp"
i.e. 
```
mkdir pretrained
mkdir pretrained/cifar10
mkdir pretrained/cifar10/temp
```

Update the dataset location in utils/load_dataset.py

Run the training program
``` 
python train.py --dataset=cifar10 --epochs=100 --loss=crossentropy --optimizer=adam --arch=resnet18
```

Features
--------
* Customizable train and validation split
* Resume training after stopping (keeps optimizer state, and best model)
* Model naming convention
* Saves models in "./pretrained/\<dataset name\>" and state in "./pretrained/\<dataset name\>/temp"


Run the test.py program to verify things work
``` 
python test.py --arch=resnet18 --suffix=1
```

Curvature Code
---------------
There are two programs, one on a toy example names "test_curvature_db.py" and another code to calculate the decision boundary curvature on real datasets called "test_crv_real_datasets.py"

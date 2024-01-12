# Ship-monitoring-systems
Practical tasks in ship monitoring, including Curriculum Learning, Full type classification, Adversarial attack training, and Visualization of result.

## Curriculum Learning
1.First you need to manually split the original dataset into subsets of data with increasing difficulty levels, e.g. subset 1, subset 2, etc.

<img src="https://github.com/axun2001/Ship-monitoring-systems/blob/main/images/1.png" width="600">

- Data Set Structure
    - Train Dataset
        - Subset 1
        - Subset 2
        - Subset N
    - Validation Dataset
        - Val
    - Test Dataset
        - Test

2.Modify the corresponding parameters in the [Train.py](https://github.com/axun2001/Ship-monitoring-systems/blob/main/Curriculum%20Learning/CreateDataset.py) such as train position and train dataloader

3.Use the optimal weights parameter file saved in [Train.py](https://github.com/axun2001/Ship-monitoring-systems/blob/main/Curriculum%20Learning/CreateDataset.py) and load it into [ModelEvaluate.py](https://github.com/axun2001/Ship-monitoring-systems/blob/main/Curriculum%20Learning/ModelEvaluate.py) with a picture of the test set (this example uses the validation set as the test set at the same time).

After the code is run, a .csv file of the following form will be obtained (vertical coordinate - category; horizontal coordinate - path image).

<img src="https://github.com/axun2001/Ship-monitoring-systems/blob/main/images/2.png" width="600">

4.At this point we use [CreateDataset.py](https://github.com/axun2001/Ship-monitoring-systems/blob/main/Curriculum%20Learning/CreateDataset.py) to generate the image path with the real label of the image.
(The former is the absolute path of each image, and the latter is the real label of the image)

<img src="https://github.com/axun2001/Ship-monitoring-systems/blob/main/images/3.png" width="600">

5.Load the previously obtained txt file and csv file into [compare.py](https://github.com/axun2001/Ship-monitoring-systems/blob/main/Visualization%20of%20Results/compare.py) and run it to get the Confusion Matrix, the ROC curve, the PR curve, and the Average PR curve.

<img src="https://github.com/axun2001/Ship-monitoring-systems/blob/main/images/4.png" width="600">
<img src="https://github.com/axun2001/Ship-monitoring-systems/blob/main/images/5.png" width="600">
<img src="https://github.com/axun2001/Ship-monitoring-systems/blob/main/images/6.png" width="600">

## Full Type Classification-Few shots Learning

1.The main focus of the task is on the training strategy, and the only parts that need to be adjusted are in the [Siamese-Train.py](https://github.com/axun2001/Ship-monitoring-systems/blob/main/Full%20Type%20Classification/Siamese-Train.py), the dataset paths, and the parameters (margin, optimizer, dimensionality of the output feature vector)

```
lr = 0.001

embedding_dim = 2048

Margin = 0.875

batch_size = 32

num_epochs = 30
```

After the run, the optimal weight parameter file is saved, as well as the confidence csv file for each of the output images.

2.At this point we use [CreateDataset.py](https://github.com/axun2001/Ship-monitoring-systems/blob/main/Curriculum%20Learning/CreateDataset.py) to generate the image path with the real label of the image.

3.Load the previously obtained txt file and csv file into [compare.py](https://github.com/axun2001/Ship-monitoring-systems/blob/main/Visualization%20of%20Results/compare.py) and run it to get the Confusion Matrix, the ROC curve, the PR curve, and the Average PR curve.

## Adversarial Attack Training

I'll add more later when I'm done...

## Visualization of Result

1.Regarding the scripts mentioned in the previous section, here are the [txt](https://github.com/axun2001/Ship-monitoring-systems/blob/main/Visualization%20of%20Results/test.txt) and [csv](https://github.com/axun2001/Ship-monitoring-systems/blob/main/Visualization%20of%20Results/resnet34.csv) files used as examples, which can be loaded directly into [compare.py](https://github.com/axun2001/Ship-monitoring-systems/blob/main/Visualization%20of%20Results/compare.py) for testing.

2.The three scripts ([plt-curve](https://github.com/axun2001/Ship-monitoring-systems/blob/main/Visualization%20of%20Results/plt.-curve.py). [plt-pr](https://github.com/axun2001/Ship-monitoring-systems/blob/main/Visualization%20of%20Results/plt-pr.py). [plt-roc](https://github.com/axun2001/Ship-monitoring-systems/blob/main/Visualization%20of%20Results/plt-roc.py)) are used to plot curves for various metrics, such as accuracy, average ROC curve, average PR curve, and so on. However, it should be noted that the corresponding csv files need to be loaded when using the scripts.

3.[Loss landscape](https://github.com/okn-yu/Visualizing-the-Loss-Landscape-of-Neural-Nets/tree/master):For visualizing the loss function, you can refer to another project for details on how to do this.We extracted the coordinate points from the obtained h5 file and plotted them to get the 2D as well as 3D loss function images respectively

<img src="https://github.com/axun2001/Ship-monitoring-systems/blob/main/images/7.png" width="400">
<img src="https://github.com/axun2001/Ship-monitoring-systems/blob/main/images/8.png" width="400">

The vtp file obtained through further operations can be imported into the software "[Paraview](https://www.paraview.org/download/)" for rendering. After rendering, the loss function is more detailed and realistic.

<img src="https://github.com/axun2001/Ship-monitoring-systems/blob/main/images/124123423.png" width="400">
<img src="https://github.com/axun2001/Ship-monitoring-systems/blob/main/images/2131314.png" width="400">

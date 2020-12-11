# ParameterFreeRCNs-MicroExpressionRec
Recurrent convolutional networks with parameter free modules for composite-database micro-expression recognition

#### Descriptions
These codes are used for micro-expression recognition on composite datasets (e.g., MEGC2019). The methods can be accessed by the paper "Revealing the Invisible With Model and Data Shrinking for Composite-Database Micro-Expression Recognition, IEEE TIP2020", which includes the RCN-A, RCN-S, RCN-W, RCN-P, RCN-C and RCN-F.

#### Dependencies
The code was written in Python 3.6, and tested on Windows 10 and CentOS 7. 
-  Pytorch: 1.1 or newer
-  Numpy: 1.16.3 or newer
-  Scikit-learn: 0.22.1 or newer

#### Instructions
1.  The optical flow should be extracted by your own tools before training the deep model.
2.  The data can be prepared by the script "PrepareData_LOSO_CD.py".
3.  At last, various models can be accessed by using the commond "--modelname". For training, the command like "python ModelEval_Final.py --dataset smic --dataversion 1 --epochs 20 --learningrate 0.0005 --modelname rcn_a --batchsize 64 --featuremap 32 --poolsize 5 --lossfunction crossentropy" can be used. 

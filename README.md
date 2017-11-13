# AudioAlert
eval_model.py - example of model evaluation, currently the model is loaded from CKPT so the same model have to used as the chekpoint was trained on. Later I plane to freeze the graphs so it won't be an issue.

train_module.py - a modul for training the Neural Network based on selected data feeder

neural_model.py - a class that construct the Neural Network with last layer based on max pooling

neural_model_scenario.py - same as above but with average pooling instead of max pooling, this is more useful for scene detection.


The Neural Networks I have used here are a one dimantional variation of resnet based on "Deep Residual Learning for Image Recognition
" https://arxiv.org/abs/1512.03385


referance for audio datasets:
http://www.cs.tut.fi/~heittolt/datasets.html

dataset of DCASE 2017:
http://www.cs.tut.fi/sgn/arg/dcase2017/challenge/download

dataset urbansound:
https://serv.cusp.nyu.edu/projects/urbansounddataset/download-urbansound.html

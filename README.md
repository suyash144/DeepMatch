# DNN tracking unit project
Train DNN to track the same units across days for Neuropixel recordings
General idea: use contrastive learning, helped with autoencoders to capture the waveform features

# data structure
data and code should be organized like this
./DNN_tracking_project/
    ---DATA_UnitMatch # original data for training [384, 82]
    ---R_DATA_UnitMatch  
    # preprocessed data for training. R means reduced, as we don't want to use all 384 channels, we just want interested channels
    ---test_Data_UnitMatch # original data for test [384, 82]
    ---test_R_DATA_UnitMatch 
    # preprocessed data for test. R means reduced, as we don't want to use all 384 channels, we just want interested channels
    ---Track_Units_NPXnet # the code
    ---UnitMatch #UnitMatch matlab code, to compare the prediction results
    ---Save_UnitMatch # where to save the results

# utils
AE_npdataset.py:    is used to pretrain the encoder under autoencoder framework
clip_demp.ipynb:    is a demo about do contrastive learning with 1D or 2D matrix output
demo.py:            where you want to test something
losses.py:          loss functions
metric.py:          functions to visualize training
myutil.py:          functions to preprocess the neuropixel data
npdataset.py:       is used to finetune the encoder with contrastive learning
param_fun.py:       functions to preprocess the neuropixel data
visualize.py:       functions to visualize the results

# preprocess
to slice the wanted waveform shape [channel, time] from [384, 82]

# ModelExp
where the training process is saved

# models
my DNN model

# track_analysis
post analysis using the trained DNN

# train
train_AE.py:        just train the encoder and decoder with AE loss
train_clip.py:      just train the encoder with clip loss
train_AEclip.py:    train the encoder and decoder with AE loss and clip loss simultaneously
train_finetune.py:  train the encoder with future coming data
# PersonLab

This is a Keras implementation of [PersonLab](https://arxiv.org/abs/1803.08225) for Multi-Person Pose Estimation and Instance Segmentation.
The model predicts heatmaps and various offsets which allow for computation of joint locations and connections as well as pixel instance ids. See the paper for more details.


## Training a model

If you want to use Resnet101 as the base, first download the imagenet initialization weights from [here](https://drive.google.com/open?id=1ulygah5BTWjhSGGpN20-eYV5NAozdE8Z) and copy it to your `~/.keras/models/` directory. (Over 100MB files cannot be hosted on github.)

First, construct the dataset in the correct format by running the [generate_hdf5.py](generate_hdf5.py) script. Before running, just set the `ANNO_FILE` and `IMG_DIR` constants at the top of the script to the paths to the COCO person_keypoints annotation file and the image folder respectively.

Edit the [config.py](config.py) to set options for training, e.g. input resolution, number of GPUs, whether to freeze the batchnorm weights, etc. More advanced options require altering the [train.py](train.py) script. For example, changing the base network can be done by adding an argument to the get_personlab() function, see the documentation [there](model.py#L162).

After eveything is configured to your liking, go ahead and run the train.py script.

## Testing a model

See the [demo.ipynb](demo.ipynb) for sample inference and visualizations.

## Technical Debts
Several parts of this codebase are borrowed from others. These include:

* The [Resnet-101 in Keras](https://gist.github.com/flyyufelix/65018873f8cb2bbe95f429c474aa1294)

* The augmentation code (which is different from the procedure in the PersonLab paper) and data iterator code is heavily borrowed from [this fork](https://github.com/anatolix/keras_Realtime_Multi-Person_Pose_Estimation) of the Keras implementation of CMU's "Realtime Multi-Person Pose Estimation". (The pose plotting function is also influenced by the one in that repo.)

* The Polyak Averaging callback is just a lightly modified version of the EMA callback from [here](https://github.com/alno/kaggle-allstate-claims-severity/blob/master/keras_util.py)


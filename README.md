# amygdalaGO-BOLT

This repo is a PyTorch-based framework for amygdala segmentation, whose goal is to provide an easy-to-use framework for academic researchers to develop and evaluate deep learning models. It provides fair evaluation and comparison on multiple center datasets. 

### Features

- Cover the whole process of model design, including dataset processing, model definition, model configuration, training and evaluation.
- Provide SOTA models as baseline for comparison. Model definition, training and evaluation code are simple with no complex code encapsulation.
- Provide models, losses, metrics, augmentation and etc. for 3D data, multiple centers and multiple tasks.
- Optimized training techniques for SOTA performance.


### Usage

We provide flexible usage. If you just want to use the models in your own framework, you can directly find the corresponding models in the `model/` folder and use them in your own framework. For the definition of specific models, please refer to `model/utils.py` in the `get_model` function. The models we provide do not have complex dependencies and encapsulation. The modules used by each model are defined in its own `xxx_utils.py` file. For example, the model definition of UNet only depend on `unet.py`, `unet_utils.py` and `conv_layers.py`.

If you want to use our framework, please follow below steps.

#### Install requirements

Create a new virtual environment and install all dependencies by:

```
pip install -r requirement.txt
```

#### Data preparation

Download the origin dataset from their corresponding official website.

Enter the `dataset_conversion` fold and find the dataset you want to use and the corresponding dimension (2d or 3d)

Edit the `src_path` and `tgt_path` the in `xxxdataset.py`, where the `src_path` is the path to the origin dataset, and `tgt_path` is the target path to store the processed dataset.

Then, `python xxxdataset.py`

After processing is finished, put the processed dataset into `dataset/` folder or use a soft link.

#### Configuration

Enter `config/xxxdataset/` and find the model and dimension (2d or 3d) you want to use. The training details, e.g. model hyper-parameters, training epochs, learning rate, optimizer, data augmentation, etc., can be altered here. You can try your own configuration or use the default configure, which should have a decent performance. The only thing to care is the `data_root`, make sure it points to the processed dataset directory.

#### Training

We can start training after the data and configuration is done. Several arguments can be parsed in the command line, see in the `get_parser()` function in the `train.py` and `train_ddp.py`. You need to specify the model, the dimension, the dataset, whether use pretrain weights, batch size, and the unique experiment name. Our code will find the corresponding configuration and dataset for training.

Here is an example to train with one gpu:

`python train.py --model medformer --dimension 3d --dataset Amygdala --batch_size 3 --unique_name Amygdala_3d_medformer --gpu 0`

This command will start the cross validation on Amygdala. The training loss and evaluation performance will be logged by tensorboard. You can find them in the `log/dataset/unique_name` folder. All the standard output, the configuration, and model weigths will be saved in the `exp/dataset/unique_name` folder. The results of cross validation will be saved in `exp/dataset/unique_name/cross_validation.txt`.

Besides training with a single GPU, we also provide distributed training (DDP) and automatic mixed precision (AMP) training in the `train_ddp.py`. The `train_ddp.py` is the same as `train.py` except it supports DDP and AMP. We recomend you to start with `train.py` to make sure the whole train and eval pipeline is correct, and then use `train_ddp.py` for faster training or larger batch size.

Example of using DDP:

`python train_ddp.py --model medformer --dimension 3d --dataset Amygdala --batch_size 16 --unique_name Amygdala_3d_medformer_ddp --gpu 0,1,2,3`

Example of using DDP and AMP:

`python train_ddp.py --model medformer --dimension 3d --dataset Amygdala --batch_size 32 --unique_name Amygdala_3d_medformer_ddp_amp --gpu 0,1,2,3 --amp`

We have not fully benchmark if AMP can speed up training, but AMP can reduce the GPU memory consumption a lot.

### To Do

Add more medical dataset support.

We'll continously maintain this repo to add more SOTA models, and add more dataset support. 

Performance comparison results of the supported models and dataset

Hope this repo can serves as a solid baseline for the future medical imaging model design.

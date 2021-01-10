# Training code 
The code for training is mainly finished by Xingjian Zhen together with Liu Yang.

## Pretrained model
The pretrained model can be found in [Google Drive](https://drive.google.com/file/d/1EkrzBANwr46OZ7anZ4QcM6KsYF7UJ09T/view?usp=sharing). 

Run the followings:
``` bash
cd Dual_Manifold_Glow
mkdir models
cd models
mv PATH/TO/DOWNLOAD/DTI2ODF-3D-pretrain.pth .
```

Also, the model can be trained from scratch with the default hyperparameters.

## Train
To train the model, run
``` bash
python train.py
```
Hyperparameters are 

## Inference
To generate the results, run
``` bash
python inference.py
```

## Model
Model is defined in [model.py](model.py)

The default number of layers and number of blocks are hard-coded in the class initialization. So the "C" in line 510 is not used. 
It needs to matche the number in [train.py](train.py) line 104, where "extra=8" is hard-coded. 

To change, the number of layers and the number of blocks, make sure that the "C" matchs "extra". 
An easy way to do so is to pass it once and print out the first "C" in the forward pass.

The reason to write it this way is to support multi-GPU training. There might be better and cleaner way to do so.

## Dataloader
Dataloader is defined in [dataset_HCP.py](dataset_HCP.py)

It will use "DataLoaderX" as the background loader so that it can fully use the CPU and I/O.

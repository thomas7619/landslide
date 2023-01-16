> cf https://docs.github.com/fr/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax 

Save environment here :
```
conda env export > environment.yml
```

Install environment here :
```
conda env create -f environment.yml
```
(first line : name of the env that can be changed)

Command to train : 

```
 python landslide_detect.py --h
 
Landslide detection with attention layers and fusion image/dem

options:
  -h, --help            show this help message and exit
  --path_img PATH_IMG   path to images
  --path_dem PATH_DEM   path to dem
  --path_mask PATH_MASK
                        path to classes
  --val_img VAL_IMG     path to validation images
  --val_dem VAL_DEM     path to validation dem
  --val_mask VAL_MASK   path to validation classes
  --path_model PATH_MODEL
                        path to save models
  --batch_size BATCH_SIZE
                        Size of the batch (default : 10)
  --epochs EPOCHS       Number of epochs (default : 50)
  --ncols NCOLS         numbers of columns of low res (default : 256)
  --nrows NROWS         numbers of rows of low res (default : 256)
  --ch_out CH_OUT       channels in output (default : 1)
  --ch_in CH_IN         channels in input (default : 3)
  --gpu_ids GPU_IDS     priority for GPU (default : 0)
  --pretrain PRETRAIN   pretrained model (default : )
  --gf GF               number of filter for the residual part (default : 32)
  --initial_lr INITIAL_LR
                        Initial learning rate (default : 0.001000)
  --decay_steps DECAY_STEPS
                        Decay steps (default : 100000000)
  --decay_rate DECAY_RATE
                        Decay rate (default : 0.900000)
  --efficientnet        Use efficientnet backcone
  --vgg                 Use VGG16 backcone
  --concat_ima          Concatene image only in decoder (instead of images and DEM by default)
  --concat_dem          Concatene DEM only in decoder (instead of images and DEM by default)
  --nores               Use classic convolutions instead of residual ones
  --note NOTE           Personal note for the readme file
```


Command to test : 

```
 python test.py --h

options:
  -h, --help            show this help message and exit
  --data_ima DATA_IMA   path to image folder
  --data_dem DATA_DEM   path to dem folder
  --model MODEL         path to model
  --gpu_ids GPU_IDS [GPU_IDS ...]
                        priority for GPU (default : 0)
  --batch_size BATCH_SIZE
                        size of batch (larer = speeder but need ram, default : 15)
  --ncols NCOLS         numbers of columns (default : 256)
  --nrows NROWS         numbers of rows (default : 256)
  --ch_out CH_OUT       channels in output (default : 1)
  --ch_in CH_IN         channels in input (channels in output and structure. Default : 3)
  --output OUTPUT       path to output (if None : same folder as data
  --ground_truth GROUND_TRUTH
                        path to ground truth (if exists)
  --oldnorm             (for very first models compatibility, older normalization)
```

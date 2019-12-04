# MoodGANimation
## GANimation: Anatomically-aware Facial Animation from a Single Image
### [[Project]](http://www.albertpumarola.com/research/GANimation/index.html)[ [Paper]](https://arxiv.org/abs/1807.09251) 
Official implementation of [GANimation](http://www.albertpumarola.com/research/GANimation/index.html). In this work we introduce a novel GAN conditioning scheme based on Action Units (AU) annotations, which describe in a continuous manifold the anatomical facial movements defining a human expression. Our approach permits controlling the magnitude of activation of each AU and combine several of them. For more information please refer to the [paper](https://arxiv.org/abs/1807.09251).

This code was made public to share our research for the benefit of the scientific community. Do NOT use it for immoral purposes.

![GANimation](http://www.albertpumarola.com/images/2018/GANimation/teaser.png)

### Prerequisites
- Install PyTorch (version 0.3.1), Torch Vision and dependencies from http://pytorch.org
- Install requirements.txt (```pip install -r requirements.txt```)

### Data Preparation
The code requires a directory containing the following files:
- `imgs/`: folder with all image
- `aus_openface.pkl`: dictionary containing the images action units.
- `train_ids.csv`: file containing the images names to be used to train.
- `test_ids.csv`: file containing the images names to be used to test.

An example of this directory is shown in `sample_dataset/`.

To generate the `aus_openface.pkl` extract each image Action Units with [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Action-Units) and store each output in a csv file the same name as the image. Then run:
```
python data/prepare_au_annotations.py
```

### Run
To train:
```
bash launch/run_train.sh
```
To test:
```
python test --input_path path/to/img
```

### Citation
If you use this code or ideas from the paper for your research, please cite our paper:
```
@inproceedings{pumarola2018ganimation,
    title={GANimation: Anatomically-aware Facial Animation from a Single Image},
    author={A. Pumarola and A. Agudo and A.M. Martinez and A. Sanfeliu and F. Moreno-Noguer},
    booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
    year={2018}
}
```
## MoodGANimation
In recent years, there is a growing interest in affective computing and interpreting the emotional states of users in human-machine interaction. In HMI (Human-machine interaction), systems aim to interact with users in a natural way and in uncontrolled environments. Regarding this, facial expressions are responses to changes in someone’s emotional state or intentions which can be caused by social interactions as well. That’s why facial expressions has recently caught big interest in affective computing. Following this, in association with facial expressions, emotions have different levels of interpretation. Two commonly used ones are discrete categories and affectional dimensions.  

According to discrete categories theory proposed by Ekman, there are 6 universally accepted basic emotions which are happiness, surprise, fear, sadness, anger and disgust. However definition of emotion may be too limited with discrete categorization. Mehrabian and Russell defined emotion in 3 continuous dimensions called pleasure-arousal-dominance. Their argument is that emotions can be mapped to these 3D emotion space. Subtle differences from one expression to another in comparison to sharp categorical changes can be picked out along the affect dimensions. Within these 3 continuous scales, pleasure is ranging from unhappy/unpleasant to happy/pleasant, arousal is ranging from sleepiness/tiredness to frantic excitement and dominance is ranging from submissiveness to influence / autonomy. In most of the studies, they are also moods. In many studies, dominance is usually not considered as o effective dimension and regardless of dominance, 2D arousal and valence space is used. In particular, in this study, we study the utility of three independent affectional dimensions, namely pleasure, arousal and dominance, to describe the perception of human emotion in a continuous manner.

In GANimation, an attention based generator is used. Generator creates an attention mask and color maps for target condition. For the final output, in the original image, the regions encoded by attention mask is replaced by the features coming from color maps. Consequently, attention mask is pointing out the regions in original image and generated color maps respectively. The next step is to change the feature space of action units in __GANimation__ to moods.

EmotioNet has annotation of active AUs, unfortunately for training set, it does not provide categorical or continuous annotation. For that reason, instead of EmotioNet, AffectNet is preferred. In AffectNet, there are approximately 450.000 images which are manually annotated. It has 11 categories: _Neutral, Happy, Sadness, Surprise, Fear, Disgust, Anger, Contempt, None, No-Face, Uncertain_. Range of arousal and valence is in between [-1, +1]. For Uncertain and No-Face categories, arousal and valence values are taken as -2.

Simultaneous edition of multiple AUs in original GANimation model is tested along with single AU edition. AU combinations represent several emotion categories. So in this manner, it is aimed to assess the power of GANimation for generation of different categories of emotion. For each of action unit in combination, intensity is interpolated from 0 to 1 simultaneously. Results can be seen below:

<p align="center"> <img src="https://raw.githubusercontent.com/sevimcaliskann/MoodGANimation/master/emotions_with_aus.png" width="600" align="middle"> </p>

Results for the interpolation for 3 affectional dimensions can be seen below:

Positive Pleasure Scala: <br />
![](https://raw.githubusercontent.com/sevimcaliskann/MoodGANimation/master/valence.png)

Negative Pleasure Scala: <br />
![](https://raw.githubusercontent.com/sevimcaliskann/MoodGANimation/master/valence_negative.png)

Positive Arousal Scala: <br />
![](https://raw.githubusercontent.com/sevimcaliskann/MoodGANimation/master/arousal.png)

Negative Arousal Scala: <br />
![](https://raw.githubusercontent.com/sevimcaliskann/MoodGANimation/master/arousal_negative.png)

Positive Dominance Scala: <br />
![](https://raw.githubusercontent.com/sevimcaliskann/MoodGANimation/master/dominance_positive.png)

Negative Dominance Scala: <br />
![](https://raw.githubusercontent.com/sevimcaliskann/MoodGANimation/master/dominance.png)

In the end, we propose that affect space is more valueable and interpretable in terms of emotion representation and it gives competitive results in comparison to AU based representation. Discrete representation keeps manifold learning limited and AU representation is sparse. Continuous representation gives potential to exploit emotion manifold extensively.

### Generation of Videos of Facial Expressions
On top of that, for generating short video clips of expressions and learning spatiotemporal features, the model is trained with video clips from AffWild dataset. Short video clips are cropped from videos with their groundtruth labellings. In the beginning, given previous frame, next frame is generated and later on from a start frame, n number of frames are generated in recurrent way. In our case, n is equal to 16. Groundtruth frames are used to steer learning, L1 loss is used between groundtruth and generated frames. There are different methods tested with spatiotemporal expression learning:

- Convolutional GRU unit is included in discriminator before prediction and adversarial layers
- Convolutional GRU unit is included in generator instead of residual blocks of original model and U-Net connections between encoder and decoder of generator
- Using 2 discriminators, one is frame discriminator for fake/real discrimination of frames and affect prediction and video discriminator for collection of frames generated from a start point

Results from 3 different cases are presented below, same standard from original GANimation is followed: first column is for original image, second column is generated affect frame, third frame is attention mask and last column is color mask map associated with conditional affects. 

- Convolutional GRU on discriminator: <br />

![](https://raw.githubusercontent.com/sevimcaliskann/MoodGANimation/master/example%20outputs/gru_disc/140_epoch_1501_out.gif)

![](https://raw.githubusercontent.com/sevimcaliskann/MoodGANimation/master/example%20outputs/gru_disc/167_epoch_1501_out.gif)

![](https://raw.githubusercontent.com/sevimcaliskann/MoodGANimation/master/example%20outputs/gru_disc/178_epoch_1501_out.gif)

- Convolutional GRU on U-Net generator: <br />

![](https://raw.githubusercontent.com/sevimcaliskann/MoodGANimation/master/example%20outputs/gru_gen/140_epoch_3036_out.gif)

![](https://raw.githubusercontent.com/sevimcaliskann/MoodGANimation/master/example%20outputs/gru_gen/167_epoch_3036_out.gif)

![](https://raw.githubusercontent.com/sevimcaliskann/MoodGANimation/master/example%20outputs/gru_gen/178_epoch_3036_out.gif)

- Separated video and frame discriminators: <br />

![](https://raw.githubusercontent.com/sevimcaliskann/MoodGANimation/master/example%20outputs/two_disc/140_epoch_601_out.gif)

![](https://raw.githubusercontent.com/sevimcaliskann/MoodGANimation/master/example%20outputs/two_disc/167_epoch_701_out.gif)

![](https://raw.githubusercontent.com/sevimcaliskann/MoodGANimation/master/example%20outputs/two_disc/178_epoch_701_out.gif)

### How to reproduce results 

There are different branches for different application sides of project. In __master__ branch, there is original GANimation methods which you can run with action units. You can refer to upper side of blog for details about how to generate data for training and how to run training. After you get familiar with original GANimation model, different branches can be summarized as:

#### Branch *moods_static_images*

In this branch you can train GANimation model with affectional data which we also refer as *moods*. The only difference in this branch is how we read the data. Data reading is managed by *data/MoodDataset.py*. For dataset, AffectNet is chosen since it has both discrete emotion and affect labels (valence and arousal). There are two ways to read the data, one is reading from .csv file. It is customized with respect to original AffectNet annotation csv file. Therefore, if you are using other dataset, my suggestion would be to prepare your data in .pkl file and that's the second way to go. In .pkl file, data is stored as dictionary. For the name of the image, moods are put forward as corresponding values. 

As differences regarding to reading data, *dataset_mode* should be selected as 'mood' and also *train_info_file* stands for annotations file of training partition and *test_info_file* is for test partition similarly.

For training, an example command would look like this:

```
python GANimation/train.py \
--data_dir datasets/affectnet \
--train_images_folder cropped \
--test_images_folder cropped \
--train_ids_file datasets/affectnet/train.csv \
--test_ids_file datasets/affectnet/test.csv \
--train_info_file datasets/affectnet/train.pkl \
--test_info_file datasets/affectnet/test_norm_1000_resnet50.pkl \
--name experiment_1 \
--nepochs_no_decay 20 \
--nepochs_decay 10 \
--batch_size 25 \
--gpu_ids 0 \
--lambda_mask_smooth 1e-4 \
--checkpoints_dir checkpoints \
--load_epoch -1 \
--dataset_mode mood \
--lambda_D_cond 40000 \
--lambda_mask 0.10 \
--cond_nc 2 \
--lambda_D_prob 100 

```

For testing with a static image, an example command would look like this:

```
python test.py \
--input_path imgs/face.png \
--output_dir GANimation/test_outputs \
--checkpoints_dir checkpoints \
--name experiment_1 \
--cond_nc 2 \
--load_epoch -1

```

#### Branch *video_ganimation*

In this branch, the purpose is to generate small video clips of animated expressions. It is built upon the *moods_static_images* branch. GANimation model without reconstruction is used. In other words, the supervision for the frames are not coming from cycle loss in manner of CycleGAN, instead of that, supervision is coming from groundtruth frames. Previous conditional frame is either taken from generated or real frames, two of the options are possible. In the end, collection of generated frames are returned including attention mask and color maps. Implementation on training (models/ganimation.py) and testing scripts (test.py) is changed to accommodate this. 

For reading data, since we are using AffWild dataset, specifically for this dataset, a script is created which you can find under data/AffWildDataset.py. In here, as a quick tip, it is assumed that annotations for each frame is enumerated and concatenated to video ID, all of annotations are kept in OrderedDict data structure in a way consecutive frames will be following the order. 

One additional parameters in *'frames_cnt'* which keeps the number of frames to be generated in one iteration. 

For training, an example command would look like this:

```

python GANimation/train.py \
--data_dir dataset/affwild \
--train_images_folder cropped_faces \
--test_images_folder cropped_faces \
--train_ids_file datasets/affwild/train_ids.csv \
--test_ids_file datasets/affwild/test_ids.csv \
--train_info_file datasets/affwild/annotations/train/annotations.pkl \
--test_info_file datasets/affwild/annotations/test/annotations.pkl \
--name experiment_1 \
--nepochs_no_decay 200 \
--nepochs_decay 100 \
--batch_size 4 \
--gpu_ids 0 \
--lambda_mask_smooth 5e-5 \
--checkpoints_dir checkpoints \
--load_epoch -1 \
--dataset_mode affwild \
--lambda_D_cond 4000 \
--lambda_mask 0.1 \
--lambda_D_gp 10 \
--cond_nc 4 \
--lambda_D_prob 10 \
--lambda_cyc 1.5 \
--frames_cnt 6

```

For testing with a static image, an example command would look like this:

```
python test.py \
--input_path faces/imgs/face.jpg \
--output_dir GANimation/test_outputs \
--checkpoints_dir checkpoints \
--name experiment_1 \
--cond_nc 2 \
--load_epoch -1 \
--frames_cnt 16 \
--moods_pickle_file datasets/affwild/annotations/test/178.pkl \
--groundtruth_video datasets/affwild/videos/178.avi \

```

In the branches __*convgru_disc*__ and __*convgru_unet_gen*__, the differences are generator and discriminator architectures. In the discriminator, before affect prediction a GRU unit is added and in generator, U-Net connections between encoder and decoder, replacement of residual blocks with GRU unit is added. Training and testing implementations are kept same. For checking out differences and how ConvGRU is implemented, networks/discriminator_wasserstein_gan.py, networks/generator_wasserstein_gan.py and networks/convgru.py can be seen.

#### Branch *spatiotemporal_disc*

This branch is built upon __*convgru*...__ branches. ConvGRU units generator are kept, next to frame-wise discriminator, additional video discriminator is added. By adding video discriminator, temporal distribution of real data is aimed to be learnt. In video discriminator, likewise frame-based discriminator, fake/real discrimination and affect prediction is done and ConvGRU unit is included in video discriminator. Addition to frame-based discriminator paramaters, video discriminator parameters to be considered are: *lambda_D_temp* which is adversarial learning hyperparameter, lambda_D_temp_gp* which is gradient penalty hyperparameter. They are kept as one-tenth of frame-based discriminator hyperparameters:


```

python GANimation/train.py \
### ............. Default parameters etc. etc. ########
--lambda_D_prob 10 \
--lambda_D_gp 10 \
--lambda_D_temp 1 \
--lambda_D_temp_gp 1 \
### All the other hyperparameters###

```

If you have further questions and want to know more details, please open an issue or send me an e-mail: <sevimcaliskansc@gmail.com>.




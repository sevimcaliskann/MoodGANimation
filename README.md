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



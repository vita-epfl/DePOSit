<div align="center">
<h1> A generic diffusion-based approach for <br> 3D human pose prediction in the wild </h1>
<h3>Saeed Saadatnejad, Ali Rasekh, Mohammadreza Mofayezi, Yasamin Medghalchi, Sara Rajabzadeh, Taylor Mordan, Alexandre Alahi
</h3>
<h4> <i> International Conference on Robotics and Automation (ICRA), 2023 </i></h4>

 
[[arXiv](https://arxiv.org/abs/2210.05669)]

<image src="docs/diffusion.png" width="600">
</div>

<div align="center"> <h3> Abstract </h3>  </div>
<div align="justify">

3D human pose prediction, i.e., predicting a sequence of future human 3D poses given a sequence of past observed ones, is a challenging spatio-temporal task.
It can be more challenging in real-world applications where occlusions  will inevitably happen, and estimated 3D coordinates of joints would contain some noise.
We provide a unified formulation in which incomplete elements (whether in the prediction or observation) are treated as noise, and propose a conditional diffusion model that denoises them and forecasts plausible poses.
Instead of naively predicting all future frames at once, our model consists of two cascaded sub-models, each specialized for modeling short and long horizon distributions.
We also propose a generic framework to improve any 3D pose prediction model, by leveraging our diffusion model in two additional steps: a pre-processing step to repair the inputs, and a post-processing step to refine the outputs.
We investigate our findings on four standard datasets (Human3.6M, HumanEva-I, AMASS, and 3DPW), and obtain significant improvements over the state of the art.

</br>

# Getting started

## Requirements
The code requires Python 3.7 or later. The file [requirements.txt](requirements.txt) contains the full list of required Python modules.
```
pip install -r requirements.txt
```

## Data
[Human3.6M](http://vision.imar.ro/human3.6m/description.php) in exponential map can be downloaded from [here](http://www.cs.stanford.edu/people/ashesh/h3.6m.zip).

Directory structure: 
```
H3.6m
|-- S1
|-- S5
|-- S6
|-- ...
|-- S11
```
[AMASS](https://amass.is.tue.mpg.de/en) and [3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/) from their official websites.




Specify the data path with ```data_dir``` argument.

# Training and Testing

## Human3.6M
You need to train a short-term and long-term model using these commands:
```bash
python main_tcd_h36m.py --mode train --epochs 50 --data all --joints 22 --input_n 50 --output_n 5 --data_dir data_dir --output_dir model_s
```

```bash
python main_tcd_h36m.py --mode train --epochs 50 --data all --joints 22 --input_n 55 --output_n 20 --data_dir data_dir --output_dir model_l
```

For evaluating the TCD model you can run the following command. Specify the short-term and long-term model checkpoints directory with ```--model_s``` and ```--model_l``` arguments.

```bash
python main_tcd_h36m.py --mode test --data all --joints 22 --input_n 50 --output_n 25 --data_dir data_dir --model_s model_s --model_l model_l --output_dir model_l
```

The results will be saved in a csv file in the output directory.


## AMASS and 3DPW
You can train a model on AMASS dataset using the following command:
```bash
python main_amass.py --mode train --epochs 50 --dataset AMASS --data all --joints 18 --input_n 50 --output_n 25 --data_dir data_dir --output_dir model_amass
```

Then you can evaluate it on both AMASS and 3DPW datasets:

```bash
python main_amass.py --mode test --dataset AMASS --data all --joints 18 --input_n 50 --output_n 25 --data_dir data_dir --output_dir model_amass
```
```bash
python main_amass.py --mode test --dataset 3DPW --data all --joints 18 --input_n 50 --output_n 25 --data_dir data_dir --output_dir model_amass
```
The results will be saved in csv files in the output directory.


# Work in Progress
This repository is being updated so stay tuned!


# Acknowledgments

The overall code framework (dataloading, training, testing etc.) was adapted from [HRI](https://github.com/wei-mao-2019/HisRepItself).
The base of the diffusion was borrowed from [CSDI](https://github.com/ermongroup/CSDI).

## Citation

```
@INPROCEEDINGS{saadatnejad2023diffusion,
  author = {Saeed Saadatnejad and Ali Rasekh and Mohammadreza Mofayezi and Yasamin Medghalchi and Sara Rajabzadeh and Taylor Mordan and Alexandre Alahi},
  title = {A generic diffusion-based approach for 3D human pose prediction in the wild},
  booktitle={International Conference on Robotics and Automation (ICRA)}, 
  year  = {2023}
}
```
## License
AGPL-3.0 license
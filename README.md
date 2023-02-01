<div align="center">
<h1> A generic diffusion-based approach for <br> 3D human pose prediction in the wild </h1>
<h3>Saeed Saadatnejad, Ali Rasekh, Mohammadreza Mofayezi, Yasamin Medghalchi, Sara Rajabzadeh, Taylor Mordan, Alexandre Alahi
</h3>
<h4> <i> VITA lab, EPFL, Switzerland </i></h4>

 
[[arXiv](https://arxiv.org/abs/2210.05669)]

The first version of code

</div>

</br>

# Getting started

## Requirements
The code requires Python 3.7 or later. The file [requirements.txt](requirements.txt) contains the full list of required Python modules.
```
pip install -r requirements.txt
```

## Data
[Human3.6m](http://vision.imar.ro/human3.6m/description.php) in exponential map can be downloaded from [here](http://www.cs.stanford.edu/people/ashesh/h3.6m.zip).

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

# Training

## 2-levels model (TCD)
You need to train a short-term and long-term model using these commands:
```bash
python main_tcd_h36m.py --mode train --data all --joints 22 --input_n 50 --output_n 5 --data_dir data_dir --output_dir model_s
```

```bash
python main_tcd_h36m.py --mode train --data all --joints 22 --input_n 55 --output_n 20 --data_dir data_dir --output_dir model_l
```

# Testing

## 2-levels model (TCD)
For evaluating the TCD model you can run the following command. Specify the short-term and long-term model checkpoints directory with ```--model_s``` and ```--model_l``` arguments.

```bash
python main_tcd_h36m.py --mode test --data all --joints 22 --input_n 50 --output_n 25 --data_dir data_dir --model_s model_s --model_l model_l --output_dir results_dir
```

The results will be saved in a csv file in the output directory.


# Acknowledgments

The overall code framework (dataloading, training, testing etc.) is adapted from [HRI](https://github.com/wei-mao-2019/HisRepItself).
The base of the diffusion model is borrowed from [CSDI](https://github.com/ermongroup/CSDI).

## Citation

```
@INPROCEEDINGS{saadatnejad2023diffusion,
  author = {Saeed Saadatnejad and Ali Rasekh and Mohammadreza Mofayezi and Yasamin Medghalchi and Sara Rajabzadeh and Taylor Mordan and Alexandre Alahi},
  title = {A generic diffusion-based approach for 3D human pose prediction in the wild},
  year  = {2023},
  arxiv = {2210.05669},
}
```
## License
AGPL-3.0 license
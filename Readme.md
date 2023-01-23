This is Shiqi Liao's master thesis project:
## Counterfactual inference on retina fundus images using deep structural causal models

----------------------------------------

## Structure
This repository contains code and assets structured as follows:

- `assets/`
    - `data/`: outcome from custom DSCM on different preprocessed fundus images. 
    - `model/`: checkpoints of the trained models on color fundus images and normalized fundus images
    - `model_mask/`: checkpoints of the trained models on vessel mask
    - `model_sensitivity/`: checkpoints of the trained models on different causal assumptions
- `deepscm/`: contains the code used for running the experiments
    - `arch/`: model architectures used in experiments
    - `datasets/`: script for dataset generation and data loading used in experiments
    - `distributions/`: implementations of useful distributions or transformations
    - `experiments/`: implementation of experiments
- `Documents/`: contains the documented(term expplaination, progress report, final presentation, master thesis).
    - `final_document/Final presentation`: powerpoint and master thesis for final presentation.
- `thesis_plot_coding/`: contains the code for plotting in master thesis


## Requirements
We use Python 3.7.2 for all experiments and you will need to install the following packages:
```
pip install numpy pandas pyro-ppl pytorch-lightning scikit-image scikit-learn scipy seaborn tensorboard torch torchvision
```
or simply run `pip install -r requirements.txt`.
You will also need to sync the submodule: `git submodule update --recursive --init`.

## Usage

We assume that the code is executed from the root directory of this repository.

you can then train the models as:
```
python -m deepscm.experiments.medical.trainer -e SVIExperiment -m ConditionalVISEM --default_root_dir /path/to/checkpoints --downsample 3 --decoder_type fixed_var --train_batch_size 256 {--gpus 0}
```
The checkpoints are saved in `/path/to/checkpoints` or the provided checkpoints can be used for testing and plotting:
```
python -m deepscm.experiments.medical.tester -c /path/to/checkpoint/version_?
```
## Conclusion
We make counterfactual inference on three different preprocessed fundus images.

- On color fundus image, we find that aging causes fundus images to be slightly more yellow. This is the result of increasing pixel values on Red, Green channel and decreasing pixel values on Blue channel.

![My Image](../DSCM_fundus/assets/data/fundus/figures_original_fundus_image/counterfactual_300.png)

- On normalized fundus images and vessel mask of fundus images, the custom DSCM is not expressive enough to make valid counterfactual inference. I provide the counterfactual inferred fundus images for reference.
![My Image](../DSCM_fundus/assets/data/fundus/figures_contrast_normalized/contrast_cf_300.png)
![My Image](../DSCM_fundus/assets/data/fundus/figures_vessel_mask/vessel_counterfactual_300.png)



----------------------------------------
This Repository is following the DSCM strategy in paper _**Deep Structural Causal Models for Tractable Counterfactual Inference**_(https://arxiv.org/abs/2006.06485)

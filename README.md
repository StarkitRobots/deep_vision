# README #

## What is this repository for? ##

* Training lightweight neural networks for RoboCup

## How do I get set up? ##

* This repository uses catkin and a fixed version of tiny-dnn (github.com/rhobandeps/tiny-dnn)

## Training procedure ##

### Initial setup ###

* The initial setup only have to be done one time.
* Choose a folder in which the trainings will be made.
* Make the following symbolic links:
  * ln -sf path_to_workspace/src/rhoban/deep_vision/bin/dnn_trainer 
  * ln -sf path_to_workspace/src/rhoban/deep_vision/scripts
* The followings are optional:
  * ln -sf path_to_workspace/src/rhoban/deep_vision/README.md
  * ln -sf path_to_workspace/src/rhoban/deep_vision/configs

### Training ###

* Download the data from the website (balls or goals) to your chosen folder.
  * You need to be an administrator.
  * You can download the patches in 'Categories'.
    Note: In `Sessions` you can disable the sessions you don't want.
* Prepare data
  * Use `scripts/prepare_data.sh <Balls/Goals> <zip_file>
* Generate training and validation set
  * Run `python scripts/cifaren.py` (add ` -h` to in order to see the help message)
* Train the neural network
  * Run `./dnn_trainner` (without arguments to see help message for arguments)
    * The `test_data` and  `train_data` are the files created by `scripts/cifaren.py`.
    * The structure of the neural network is passed in a json file,
      they are in `rhoban/deep_vision/configs`.
      The parameters are the followings:
        * `networks`: contains the list of the networks you want to train.
           * The training of each network will take a core.
           * They are two kinds of networks, one layer or two layer.
             Note: Since one layer and two layers networks have different
             learning rates they shouldn't be trained with the same config file.
           * There are some examples at the end of the README.md.
        * `nb_minibatch`: for example 10
        * `nb_train_epochs`: for example 10 for fast learning 30 for more long learning
        * In order to find the "best" learning rate, a dichotomy is performed.
          `learning_rate_start` and `learning_rate_end` gives the initial boudaries
          of the dichotomy. `dichotomy_depth` specifies the depth at which we will stop searching
          When you find the best structure and the best learning rate, you should put:
              learning_rate_start=learning_rate_end
    * Acceptable rates around 95% of recognition rate is the minimal acceptable value.
    * For each network the results are in the folder `results/network_name`. It contains:
        * A file csv file`results_...csv`.
          * learning_rate: the learning rate used for the optimisation.
          * valisation_score: the validation score of the last epoch or overfit if the learning overfitted the data.
          * last_improvement: the improvement the validation score between the last and the second to last epochs.
          * last_epoch: the epoch at which the optimisation stopped.
          * learning_time: the time needed for the optimisation.
        * Two files corresponding to the neural network:
          * The architecture (.json)
          * The coefficients of the network (.bin)
      
  * When you find the best one, relunch the learning with only one network and the specific learning rate.

### Choice of detection threshold ###

In order to choose the appropriate value for the detection threshold, a program
named `analyze_acceptance_score` allows to compute the false positive and false
negative rates according to labeled images.

First, compute the scores:

```
bin/analyze_acceptance_score <architecture.json> <weights.bin> <positive_dir> <negative_dir> > scores.csv
```

Then, use the score to create a graph:

```
Rscript scripts/analyze_acceptance_score.r scores.csv
```

An image named `graph.png` will be created and will contain the false positive
and false negative rates as a function of the score threshold. This information
can be used to manipulate the trade-off between false positive and false
negative.



### Add the network to robots ###

* Put the files in the folder `env/common/dnn/posts` or `env/common/dnn/balls`
* From the folder make the symbolic links:
  * ln -sf you_architecture_file.json structure.json
  * ln -sf you_architecture_file.bin structure.json

### Examples of structures ###
 
* 1-layer (learning rate < 0.1)

 ROI   | Kernel | F1 | GX | GY | FC  
-------|--------|----|----|----|----
 32x32 | 5      | 16 | 2  | 2  | 8   
 32x32 | 5      | 16 | 2  | 2  | 8   
 32x32 | 5      | 16 | 4  | 4  | 16  
 32x32 | 5      | 32 | 4  | 4  | 16  
 32x32 | 5      | 16 | 8  | 8  | 16  
 32x32 | 5      | 32 | 8  | 8  | 16  

* 2-layer (learning rate < 0.01)

 ROI   | K1 | K2 | F1 | F2 | G1X | G1Y | G2X | G2Y | FC 
-------|----|----|----|----|-----|-----|-----|-----|----
 32x32 | 5  | 3  | 4  | 8  | 4   | 4   | 2   | 2   | 4  
 32x32 | 5  | 3  | 8  | 16 | 4   | 4   | 2   | 2   | 4  
 32x32 | 5  | 3  | 16 | 32 | 4   | 4   | 2   | 2   | 4  
 32x32 | 5  | 3  | 4  | 8  | 8   | 8   | 2   | 2   | 4  
 32x32 | 5  | 3  | 8  | 16 | 8   | 8   | 2   | 2   | 4  
 32x32 | 5  | 3  | 16 | 32 | 8   | 8   | 2   | 2   | 4  
 32x32 | 5  | 3  | 4  | 8  | 16  | 16  | 2   | 2   | 4  
 32x32 | 5  | 3  | 8  | 16 | 16  | 16  | 2   | 2   | 4  
 32x32 | 5  | 3  | 16 | 32 | 16  | 16  | 2   | 2   | 4  
 32x32 | 5  | 3  | 4  | 8  | 4   | 4   | 2   | 2   | 8  
 32x32 | 5  | 3  | 8  | 16 | 4   | 4   | 2   | 2   | 8  
 32x32 | 5  | 3  | 16 | 32 | 4   | 4   | 2   | 2   | 8  
 32x32 | 5  | 3  | 4  | 8  | 8   | 8   | 2   | 2   | 8  
 32x32 | 5  | 3  | 8  | 16 | 8   | 8   | 2   | 2   | 8  
 32x32 | 5  | 3  | 16 | 32 | 8   | 8   | 2   | 2   | 8  
 32x32 | 5  | 3  | 4  | 8  | 16  | 16  | 2   | 2   | 8  
 32x32 | 5  | 3  | 8  | 16 | 16  | 16  | 2   | 2   | 8  
 32x32 | 5  | 3  | 16 | 32 | 16  | 16  | 2   | 2   | 8  


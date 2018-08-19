# README #

## What is this repository for? ##

* Training lightweight neural networks for RoboCup

## How do I get set up? ##

* This repository uses catkin and a fixed version of tiny-dnn (github.com/rhobandeps/tiny-dnn)

## Training procedure ##

### Initial setup ###

* Choose a folder in which the trainings will be made.
* Make the following symbolic links:
  * ln -sf path_to_workspace/src/rhoban/deep_vision/bin/dnn_trainer 
  * ln -sf path_to_workspace/src/rhoban/deep_vision/scripts
* The followings are optional:
  * ln -sf path_to_workspace/src/rhoban/deep_vision/README.md

### Training ###

* Download the data from the website (balls or goals) to you chosen folder.
  * You need to be an administrator.
  * In 'Sessions' you can disable the sessions you don't want.
  * You can download the patches in 'Categories'.
* Prepare data
  * Use `scripts/prepare_data.sh <Balls/Goals> <zip_file>
  * Generate training and validation set
    * Run `python scripts/cifaren.py` (without arguments to see the help message for arguments)
* Train the neural network
  * Run `./dnn_trainner` (without arguments to see help message for arguments)
    * Structure of the neural network is passed in a json file,
      they are in `rhoban/deep_vision/configs`.
      The parameters are the followings:
        * `networks`: contains the list of the networks you want to train.
           * The training of each network will take a core.
           * They are two kinds of network, one layer or two layer.
           * There are some examples at the end of the README.md.
        * `nb_minibatch`: for example 10
        * `nb_train_epochs`: for example 10 for fast learning 30 for more long learning
        * In order to find the "best" learning rate, a dichotomy is performed.
          `learning_rate_start` and `learning_rate_end` gives the initial boudaries
          of the dichotomy. `dichotomy_depth` specifies the depth at which we will stop searching
          When you find the best structure and the best learning rate, you should put:
              learning_rate_start=learning_rate_end
    * Acceptable rates around 95% of recognition rate is the minimal acceptable value.
    * For each network the results are written in a file `results_...`.
  * When you find the best one, relunch the learning with only one network and the specific learning rate.

### Add the network to robots ###

* Put the files in the folder `env/common/dnn/posts` or `env/common/dnn/balls`
* From the folder make the symbolic links:
  * ln -sf you_architecture_file.json structure.json
  * ln -sf you_architecture_file.bin structure.json

### Examples of structures ###
 
* 1-layer (learning rate > 0.1)
| ROI   | Kernel | F1 | GX | GY | FC | 
|-------+--------+----+----+----+----+-
| 32x32 | 5      | 16 | 2  | 2  | 8  | 
| 32x32 | 5      | 16 | 2  | 2  | 8  | 
| 32x32 | 5      | 16 | 4  | 4  | 16 | 
| 32x32 | 5      | 32 | 4  | 4  | 16 | 
| 32x32 | 5      | 16 | 8  | 8  | 16 | 
| 32x32 | 5      | 32 | 8  | 8  | 16 | 

* 2-layer (learning rate > 0.01)
| ROI   | K1 | K2 | F1 | F2 | G1X | G1Y | G2X | G2Y | FC |
|-------+----+----+----+----+-----+-----+-----+-----+----+
| 32x32 | 5  | 3  | 4  | 8  | 4   | 4   | 2   | 2   | 4  |
| 32x32 | 5  | 3  | 8  | 16 | 4   | 4   | 2   | 2   | 4  |
| 32x32 | 5  | 3  | 16 | 32 | 4   | 4   | 2   | 2   | 4  |
| 32x32 | 5  | 3  | 4  | 8  | 8   | 8   | 2   | 2   | 4  |
| 32x32 | 5  | 3  | 8  | 16 | 8   | 8   | 2   | 2   | 4  |
| 32x32 | 5  | 3  | 16 | 32 | 8   | 8   | 2   | 2   | 4  |
| 32x32 | 5  | 3  | 4  | 8  | 16  | 16  | 2   | 2   | 4  |
| 32x32 | 5  | 3  | 8  | 16 | 16  | 16  | 2   | 2   | 4  |
| 32x32 | 5  | 3  | 16 | 32 | 16  | 16  | 2   | 2   | 4  |
| 32x32 | 5  | 3  | 4  | 8  | 4   | 4   | 2   | 2   | 8  |
| 32x32 | 5  | 3  | 8  | 16 | 4   | 4   | 2   | 2   | 8  |
| 32x32 | 5  | 3  | 16 | 32 | 4   | 4   | 2   | 2   | 8  |
| 32x32 | 5  | 3  | 4  | 8  | 8   | 8   | 2   | 2   | 8  |
| 32x32 | 5  | 3  | 8  | 16 | 8   | 8   | 2   | 2   | 8  |
| 32x32 | 5  | 3  | 16 | 32 | 8   | 8   | 2   | 2   | 8  |
| 32x32 | 5  | 3  | 4  | 8  | 16  | 16  | 2   | 2   | 8  |
| 32x32 | 5  | 3  | 8  | 16 | 16  | 16  | 2   | 2   | 8  |
| 32x32 | 5  | 3  | 16 | 32 | 16  | 16  | 2   | 2   | 8  |


# README #

### What is this repository for? ###

* Training lightweight neural networks for RoboCup

### How do I get set up? ###

* This repository uses catkin and a fixed version of tiny-dnn (github.com/rhobandeps/tiny-dnn)

### Training procedure ###

- Download and extract data from Tagger (rhoban
- Prepare data
  - Use `scripts/prepare_data.sh <Balls/Goals/Robots> <zip_file>
  - Filter negatives (optional)
  - Generate training and validation set
    - Run `python scripts/cifaren.py <positive_folder> <negative_folder> <output_prefix> <opt: nb_validation_images (1000)> <opt: size (16)> <opt: mode (BGR or Y)>`
      - nb_validation_image :: ~20% of the smaller category (usually the positive category)
- Train the neural network
  - Use `dnn_trainer` produced by compilation (see help message for arguments)
    - Structure of the neural network can be changed in code
    - Structure of the neural network is passed in a json file (look at configs/)
  - Learning rate need to be hand-tuned
    - How to tune
      - If the recognition rate falls suddenly:
        - System has overfitted
        - Interrupt directly
        - Use a lower learning rate
      - If in the end recognition rate is too low:
        - Try using a higher learning rate
  - Acceptable rates
    - Around 95% of recognition rate is the minimal acceptable value
- Tune acceptance_score
  - Use binary analyze_acceptance_score to produce a csv file
  - Use `Rscript  to plot the effect of acceptance_score on recognition rates

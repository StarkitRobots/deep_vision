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
    - Run `python scripts/get_n_random.py Balls/negative/generated <nb_images>`
      - Note: usually try to use a number equivalent to the number of positive generated images
      - Creates a repository `Balls/negative/generated/sample` with the appropriate files
  - Generate training and validation set
    - Run `python scripts/cifaren.py <positive_folder> <negative_folder> <file_name> <opt: nb_test_images (1000)> <opt: size (16)>`
- Train the neural network
  - Use `dnn_trainer` produced by compilation (see help message for arguments)
    - Structure of the neural network can be changed in code
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

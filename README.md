# README #

### What is this repository for? ###

* Experimenting vision with deep learning

### How do I get set up? ###

* Requires tiny-dnn https://github.com/tiny-dnn/tiny-dnn (submodule) and OpenCV
* git submodule init
* git submodule update
* mkdir build
* cd build
* cmake ..
* make

* WARNING! Make sure tiny-dnn is in a release version (currently tag v1.0.0a3)

### TODOs ###

* test avec rotation de l'img 90 deg à droite
* test goals (ball et goal dans des nn séparés?)

### Training procedure ###

- Download and extract data from Tagger
- Prepare data
  - Folder will contain all Images in the same folder
  - Sort is performed through python scripts
    - Create folder 'positive' in Balls (resp. Goals)
    - Run `python get_from_json.py <.../data.json>`
      - Positive files are moved to positive folder
  - Generated additional data
    - Run `python add_noise.py Balls/<positive_folder>`
      - Generate new data in Balls/positive/generated
    - Run `python add_noise.py Balls/`
      - Generate new data in Balls/generated 
  - Filter negatives
    - Run `python get_n_random.py Balls/generated <nb_images>`
      - Note: usually try to use a number equivalent to the number of positive generated images
      - create a repository generated/sample with the appropriate files
  - Generate training and validation set
    - Run `python cifaren.py <positive_folder> <negative_folder> <file_name> nb_test_images`
- Train the neural network
  - ball_train_exp <...> (see message)
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
  - Use Rscript to plot the effect of acceptance_score on recognition rates
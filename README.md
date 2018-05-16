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

* Améliorer le set de validation :
  - prendre des logs différents pour le jeu de validation
  - choisir le set de validation avant de générer
  - choisir aléatoirement le set de validation à chaque étape pour ne pas overfitter sur un set de validation
* Change imges EY (input_depth)
* Mettre dans un json les options des réseaux de neurones
* Tune hyperparameter automatically :
  - learning rate
  - 

### Training procedure ###

- Download and extract data from Tagger
- Prepare data :
  - run .../script/prepare_date.sh <Balls|Goals|Robots> <data.zip>
  - Filter negatives (optional)
    - Run `python get_n_random.py Balls/generated <nb_images>`
      - Note: usually try to use a number equivalent to the number of positive generated images
      - create a repository generated/sample with the appropriate files
  - Generate training and validation set
    - Run `python cifaren.py <positive_folder> <negative_folder> <file_name> <opt: nb_test_images (~10% of the images)> <opt: size (16|32)>`
      - Note: If the after generation there is significantly more negative images than positive images, do not use generated negative images.
- Train the neural network
  - ball_train_exp <...> (see message)
     - Note: change input_width and input_height (16|32)
  - Modify the used nn (optional)
     - ball_train_exp.cpp : contruct_* functions
  - Learning rate need to be hand-tuned (~0.01)
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

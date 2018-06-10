//*****************************************************************************
//
// File Name	: 'ball_train_exp.cpp'
// Author	: Steve NGUYEN
// Contact      : steve.nguyen.000@gmail.com
// Created	: lundi, march 20 2017
// Revised	:
// Version	:
// Target MCU	:
//
// This code is distributed under the GNU Public License
//		which can be found at http://www.gnu.org/licenses/gpl.txt
//
//
// Notes:	notes
//
//*****************************************************************************


#include <rhoban_utils/serialization/factory.h>
#include <rhoban_utils/util.h>

#include <iostream>
#include <vector>
#include "tiny_dnn/tiny_dnn.h"
#include <string>
#include <cmath>

using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace std;


//typedef convolutional_layer<activation::identity> conv;
typedef max_pooling_layer<relu> pool;


class InputConfig : public rhoban_utils::JsonSerializable {
public:

  int width;
  int height;
  int depth;

  InputConfig() : width(32), height(32), depth(3) {}
  InputConfig(const InputConfig & other)
    : width(other.width), height(other.height), depth(other.depth) {}
  ~InputConfig() {}

  Json::Value toJson() const override{
    throw std::logic_error(DEBUG_INFO + " not implemented");
  }

  void fromJson(const Json::Value & v, const std::string & dir_path) override{
    width  = rhoban_utils::read<int>(v, "width" );
    height = rhoban_utils::read<int>(v, "height");
    depth  = rhoban_utils::read<int>(v, "depth" );
  }

  std::string getClassName() const override{
    return "InputConfig";
  }
};

class NNBuilder : public rhoban_utils::JsonSerializable  {
public:
  /// Input of the neural network
  InputConfig input;

  /// Build a neural network according to inner configuration
  virtual network<sequential> buildNN() const = 0;

  virtual void fromJson(const Json::Value & v, const std::string & path) {
    input.read(v, "input", path);
  }
};

class OneLayerBuilder : public NNBuilder {
public:
/// The size of the kernel in conv layer
int kernel_size;

/// The number of features in conv layer
int nb_features;

/// The number of columns after pooling
int grid_x;
/// The number of lines after pooling
int grid_y;

/// The number of units in fully-connected layer
int fc_units;

OneLayerBuilder() : kernel_size(5), nb_features(16), grid_x(2), grid_y(2), fc_units(16) {}
virtual ~OneLayerBuilder() {}

network<sequential> buildNN() const override {
  network<sequential> nn;

  const serial_size_t pooling_x = input.width / grid_x;
  const serial_size_t pooling_y = input.height / grid_y;

  int convol_size = kernel_size * kernel_size * input.depth * nb_features;
  const serial_size_t fc_layer_input_dim = grid_x * grid_y * nb_features;
  int fc_size = fc_layer_input_dim * fc_units;

  std::cout << "fc_layer_input_dim: " << fc_layer_input_dim << std::endl;
  std::cout << "parameters in convolution layer    : " << convol_size << std::endl;
  std::cout << "parameters in fully-connected layer: " << fc_size << std::endl;

  nn << conv<identity>(input.width, input.height, kernel_size, input.depth, nb_features, padding::same)
     << pool(input.width, input.height, nb_features, pooling_x, pooling_y, pooling_x, pooling_y)
     << fully_connected_layer<activation::identity>(fc_layer_input_dim, fc_units)
     << fully_connected_layer<softmax>(fc_units, 2); //2 classes output
  return nn;
}

Json::Value toJson() const {
  throw std::logic_error(DEBUG_INFO + " not implemented");
}

void fromJson(const Json::Value & v, const std::string & dir_path) {
  NNBuilder::fromJson(v, dir_path);
  rhoban_utils::tryRead(v, "kernel_size", &kernel_size);
  rhoban_utils::tryRead(v, "nb_features", &nb_features);
  rhoban_utils::tryRead(v, "grid_x"     , &grid_x);
  rhoban_utils::tryRead(v, "grid_y"     , &grid_y);
  rhoban_utils::tryRead(v, "fc_units"   , &fc_units);
}

std::string getClassName() const {
  return "OneLayerBuilder";
}
};

class TwoLayersBuilder : public NNBuilder {
public:
/// The size of the kernel in the 1st conv layer
int kernel1_size;

/// The size of the kernel in the 2nd conv layer
int kernel2_size;

/// The number of features in 1st conv layer
int nb_features1;

/// The number of features in 2nd conv layer
int nb_features2;

/// The number of columns after 1st pooling
int grid1_x;
/// The number of lines after 1st pooling
int grid1_y;

/// The number of columns after 2nd pooling
int grid2_x;
/// The number of lines after 2nd pooling
int grid2_y;

/// The number of units in fully-connected layer
int fc_units;

TwoLayersBuilder()
  : kernel1_size(3), kernel2_size(3),
    nb_features1(8), nb_features2(8),
    grid1_x(2), grid1_y(2),
    grid2_x(2), grid2_y(2),
    fc_units(16)
  {}

virtual ~TwoLayersBuilder() {}

network<sequential> buildNN() const override {
  network<sequential> nn;

  const serial_size_t pooling1_x = input.width / grid1_x;
  const serial_size_t pooling1_y = input.height / grid1_y;
  const serial_size_t pooling2_x = grid1_x / grid2_x;
  const serial_size_t pooling2_y = grid1_y / grid2_y;

  const serial_size_t fc_layer_input_dim = grid2_x * grid2_y * nb_features2;

  int first_convol_size = kernel1_size * kernel1_size * input.depth * nb_features1;
  int second_convol_size = kernel2_size * kernel2_size * nb_features1 * nb_features2;
  int fc_size = fc_layer_input_dim * fc_units;

  std::cout << "fc_layer_input_dim: " << fc_layer_input_dim << std::endl;
  std::cout << "parameters in 1st convolution layer: " << first_convol_size << std::endl;
  std::cout << "parameters in 2nd convolution layer: " << second_convol_size << std::endl;
  std::cout << "parameters in fully-connected layer: " << fc_size << std::endl;

  nn << conv<identity>(input.width, input.height, kernel1_size, input.depth,
                       nb_features1, padding::same)
     << pool(input.width, input.height, nb_features1,
             pooling1_x, pooling1_y, pooling1_x, pooling1_y)
     << conv<identity>(grid1_x, grid1_y, kernel2_size, nb_features1, nb_features2, padding::same)
     << pool(grid1_x, grid1_y, nb_features2,
             pooling2_x, pooling2_y, pooling2_x, pooling2_y)
     << fully_connected_layer<activation::identity>(fc_layer_input_dim, fc_units)
     << fully_connected_layer<softmax>(fc_units, 2); //2 classes output
  return nn;
}


Json::Value toJson() const {
  throw std::logic_error(DEBUG_INFO + " not implemented");
}

void fromJson(const Json::Value & v, const std::string & dir_path) {
  NNBuilder::fromJson(v, dir_path);
  rhoban_utils::tryRead(v, "kernel1_size", &kernel1_size);
  rhoban_utils::tryRead(v, "kernel2_size", &kernel2_size);
  rhoban_utils::tryRead(v, "nb_features1", &nb_features1);
  rhoban_utils::tryRead(v, "nb_features2", &nb_features2);
  rhoban_utils::tryRead(v, "grid1_x"     , &grid1_x);
  rhoban_utils::tryRead(v, "grid1_y"     , &grid1_y);
  rhoban_utils::tryRead(v, "grid2_x"     , &grid2_x);
  rhoban_utils::tryRead(v, "grid2_y"     , &grid2_y);
  rhoban_utils::tryRead(v, "fc_units"    , &fc_units);
}

std::string getClassName() const {
  return "TwoLayersBuilder";
}
};

class NNBuilderFactory : public rhoban_utils::Factory<NNBuilder> {
public:
NNBuilderFactory() {
  registerBuilder("OneLayerBuilder",
                  [](){return std::unique_ptr<NNBuilder>(new OneLayerBuilder());});
  registerBuilder("TwoLayersBuilder",
                  [](){return std::unique_ptr<NNBuilder>(new TwoLayersBuilder());});
}
};

class Config : public rhoban_utils::JsonSerializable {
public:
std::unique_ptr<NNBuilder> nnbuilder;
int nb_minibatch;
int nb_train_epochs;
double learning_rate_start;
double learning_rate_end;
double dichotomy_depth;

Config() : nb_minibatch(10), nb_train_epochs(10), learning_rate_start(0.005),
  learning_rate_end(0.05), dichotomy_depth(5) {}
Config(const Config & other)
  : nb_minibatch(other.nb_minibatch), nb_train_epochs(other.nb_train_epochs),
  learning_rate_start(other.learning_rate_start), learning_rate_end(other.learning_rate_end),
  dichotomy_depth(other.dichotomy_depth) {}
~Config() {}

std::string getClassName() const override {
  return "DNNTrainerConfig";
}

Json::Value toJson() const override {
  throw std::logic_error(DEBUG_INFO + " not implemented");
}

virtual void fromJson(const Json::Value & v, const std::string & path) {
  nnbuilder =  NNBuilderFactory().read(v,"network", path);
    nb_minibatch = rhoban_utils::read<int>(v, "nb_minibatch" );
    nb_train_epochs = rhoban_utils::read<int>(v, "nb_train_epochs" );
    learning_rate_start = rhoban_utils::read<double>(v, "learning_rate_start" );
    learning_rate_end = rhoban_utils::read<double>(v, "learning_rate_end" );
    dichotomy_depth = rhoban_utils::read<int>(v, "dichotomy_depth" );
  }
};

static void parse_file(const std::string& filename,
                       std::vector<vec_t> *train_images,
                       std::vector<label_t> *train_labels,
                       float_t scale_min,
                       float_t scale_max,
                       int x_padding,
                       int y_padding,
                       int image_width,
                       int image_height,
                       int image_depth)
{
  // Computing image size
  int image_area = image_width * image_height;
  int image_size = image_area * image_depth;

  if (x_padding < 0 || y_padding < 0)
    throw nn_error("padding size must not be negative");
  if (scale_min >= scale_max)
    throw nn_error("scale_max must be greater than scale_min");

  std::ifstream ifs(filename.c_str(), std::ios::in | std::ios::binary);
  if (ifs.fail() || ifs.bad())
    throw nn_error("failed to open file:" + filename);

  uint8_t label;
  std::vector<unsigned char> buf(image_size);

  while (ifs.read((char*) &label, 1)) {
    vec_t img;

    if (!ifs.read((char*) &buf[0], image_size)) break;

    if (x_padding || y_padding)
    {
      int w = image_width + 2 * x_padding;
      int h = image_height + 2 * y_padding;

      img.resize(w * h * image_depth, scale_min);

      for (int c = 0; c < image_depth; c++) {
        for (int y = 0; y < image_height; y++) {
          for (int x = 0; x < image_width; x++) {
            img[c * w * h + (y + y_padding) * w + x + x_padding]
              = scale_min + (scale_max - scale_min) * buf[c * image_area + y * image_width + x] / 255;
          }
        }
      }
    }
    else
    {
      std::transform(buf.begin(), buf.end(), std::back_inserter(img),
                     [=](unsigned char c) { return scale_min + (scale_max - scale_min) * c / 255; });
    }

    train_images->push_back(img);
    train_labels->push_back(label);
  }
}

double train_cifar(string data_train, string data_test, string nn_config,
                 double learning_rate, ostream& log) {
    // specify loss-function and learning strategy
    Config config;
    config.loadFile(nn_config);
    adam optimizer;

    network<sequential> nn = config.nnbuilder->buildNN();
    InputConfig input = config.nnbuilder->input;

    log << "learning rate:" << learning_rate << endl;

    cout << "Input: " << input.width << "x" << input.height << "x" << input.depth << std::endl;

    cout << "load models..." << endl;

    // load cifar dataset
    vector<label_t> train_labels, test_labels;
    vector<vec_t> train_images, test_images;

    parse_file(data_train,
               &train_images, &train_labels, -1.0, 1.0, 0, 0,
               input.width, input.height, input.depth);

    parse_file(data_test,
               &test_images, &test_labels, -1.0, 1.0, 0, 0,
               input.width, input.height, input.depth);

    cout << "start learning" << endl;

    progress_display disp(train_images.size());
    timer t;
    const int nb_minibatch = config.nb_minibatch;     ///< minibatch size
    const int nb_train_epochs = config.nb_train_epochs;  ///< training duration

    optimizer.alpha *= static_cast<tiny_dnn::float_t>(sqrt(nb_minibatch) * learning_rate);

    bool overfitting_flag=false;
    // create callback
    auto on_enumerate_epoch = [&]() {
        cout << t.elapsed() << "s elapsed." << endl;
        timer t1;
        tiny_dnn::result res = nn.test(test_images, test_labels);
        cout << t1.elapsed() << "s elapsed (test)." << endl;
        float percentage = (float)(res.num_success)/(float)(res.num_total)*100.0;
        log << res.num_success << "/" << res.num_total << " "<< percentage <<"%"<<endl;

        if (percentage < 51.0){
            cout << "Overfitting." << endl;
            overfitting_flag=true;
            return false;
        }
        disp.restart(train_images.size());
        t.restart();
        return true;
    };

    auto on_enumerate_minibatch = [&]() {
        disp += nb_minibatch;
    };

    // training
    nn.train<cross_entropy>(optimizer, train_images, train_labels,
                            nb_minibatch, nb_train_epochs, on_enumerate_minibatch,
                            on_enumerate_epoch, true, 1);

    cout << "end training." << endl;

    if (overfitting_flag == true){
        return 0.0;
    }

    // test and show results
    tiny_dnn::result res = nn.test(test_images, test_labels);
    res.print_detail(cout);

    nn.save("dnn_architecture.json", content_type::model, file_format::json);
    nn.save("dnn_weights.bin", content_type::weights, file_format::binary);

    // save networks
    //ofstream ofs("ball_exp_weights");
    //ofs << nn;

    return (float)(res.num_success)/(float)(res.num_total)*100.0;
}

void dichotomic_train_cifar(string data_train, string data_test, string nn_config,
                 double learning_rate_start, double learning_rate_end, double dichotomy_depth, ofstream& results_file) {
  // search is finished
  if (dichotomy_depth < 0){
    cout << "Search finished" << endl;
    return;
  }
  cout << "Search depth: " << dichotomy_depth << endl;

  // if learning_rate_start = learning_rate_end we stop searching
  // independently of current dichotomy_depth
  if (learning_rate_start == learning_rate_end){
    dichotomy_depth=0;
  }

  // we try the middle learning rate
  double learning_rate = (learning_rate_end + learning_rate_start)/2.0;
  timer t;
  double validation_score = train_cifar(data_train, data_test, nn_config, learning_rate, cout);
  if (validation_score > 0){
    results_file << learning_rate << "," << setprecision (4) << validation_score << "," << t.elapsed() << std::endl;
    learning_rate_start = learning_rate;
  }
  else{
    results_file << learning_rate << "," << "overfit" << "," << t.elapsed() << std::endl;
    learning_rate_end = learning_rate;
  }
  dichotomic_train_cifar(data_train, data_test, nn_config, learning_rate_start, learning_rate_end, dichotomy_depth-1, results_file);
}

int main(int argc, char **argv) {
    if (argc != 4) {
      cerr << "Usage : " << argv[0]
           << " <train_file> <test_file> <nn_config_file>" << endl;
      exit(EXIT_FAILURE);
   }
  Config config;
  config.loadFile(argv[3]);

  double learning_rate_start = config.learning_rate_start;
  double learning_rate_end = config.learning_rate_end;
  double dichotomy_depth = config.dichotomy_depth;
  InputConfig input = config.nnbuilder->input;
  std::unique_ptr<NNBuilder> nnbuilder = config.nnbuilder;

  std::ofstream results_file("results.csv");
  results_file << "learning_rate,validationScore,learning_time_input"
               << input.width
               << "x"
               << input.height
               << "x"
               << input.depth
               << "_kernerl"
               << nnbuilder->kernel_size
               << "_features"
               << nnbuilder->nb_features
               << "_grid"
               << nnbuilder->grid_x
               << "x"
               << nnbuilder->grid_y
               << "_fcunits"
               << nnbuilder->fc_units
               << std::endl;
  dichotomic_train_cifar(argv[1], argv[2], argv[3], learning_rate_start, learning_rate_end, dichotomy_depth, results_file);
}

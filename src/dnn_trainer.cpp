//*****************************************************************************
//
// File Name	: 'dnn_trainer.cpp'
// Authors	: Steve NGUYEN
//            Patxi LABORDE-RUIBIETA
// Contact      : steve.nguyen.000@gmail.com
//                patxi.laborde.zubieta@gmail.com
// Created	: 6th of December 2018
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

#include "rhoban_utils/threading/multi_core.h"

#include <sys/stat.h>
#include <sys/types.h>

using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace std;

// typedef convolutional_layer<activation::identity> conv;
typedef max_pooling_layer<relu> pool;

class InputConfig : public rhoban_utils::JsonSerializable
{
public:
  int width;
  int height;
  int depth;

  InputConfig() : width(32), height(32), depth(3)
  {
  }
  InputConfig(const InputConfig& other) : width(other.width), height(other.height), depth(other.depth)
  {
  }
  ~InputConfig()
  {
  }

  Json::Value toJson() const override
  {
    throw std::logic_error(DEBUG_INFO + " not implemented");
  }

  void fromJson(const Json::Value& v, const std::string& dir_path) override
  {
    width = rhoban_utils::read<int>(v, "width");
    height = rhoban_utils::read<int>(v, "height");
    depth = rhoban_utils::read<int>(v, "depth");
  }

  std::string getClassName() const override
  {
    return "InputConfig";
  }
};

class NNBuilder : public rhoban_utils::JsonSerializable
{
public:
  /// Input of the neural network
  InputConfig input;

  /// Build a neural network according to inner configuration
  virtual network<sequential> buildNN() const = 0;

  virtual void fromJson(const Json::Value& v, const std::string& path)
  {
    input.read(v, "input", path);
  }

  virtual std::string toString() const = 0;
};

class OneLayerBuilder : public NNBuilder
{
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

  OneLayerBuilder() : kernel_size(5), nb_features(16), grid_x(2), grid_y(2), fc_units(16)
  {
  }
  virtual ~OneLayerBuilder()
  {
  }

  network<sequential> buildNN() const override
  {
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
       << fully_connected_layer<softmax>(fc_units, 2);  // 2 classes output
    return nn;
  }

  Json::Value toJson() const
  {
    throw std::logic_error(DEBUG_INFO + " not implemented");
  }

  void fromJson(const Json::Value& v, const std::string& dir_path)
  {
    NNBuilder::fromJson(v, dir_path);
    rhoban_utils::tryRead(v, "kernel_size", &kernel_size);
    rhoban_utils::tryRead(v, "nb_features", &nb_features);
    rhoban_utils::tryRead(v, "grid_x", &grid_x);
    rhoban_utils::tryRead(v, "grid_y", &grid_y);
    rhoban_utils::tryRead(v, "fc_units", &fc_units);
  }

  std::string getClassName() const
  {
    return "OneLayerBuilder";
  }

  std::string toString() const
  {
    return "_kernerl" + to_string(kernel_size) + "_features" + to_string(nb_features) + "_grid" + to_string(grid_x) +
           "x" + to_string(grid_y) + "_fcunits" + to_string(fc_units);
  }
};

class TwoLayersBuilder : public NNBuilder
{
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
    : kernel1_size(3)
    , kernel2_size(3)
    , nb_features1(8)
    , nb_features2(8)
    , grid1_x(2)
    , grid1_y(2)
    , grid2_x(2)
    , grid2_y(2)
    , fc_units(16)
  {
  }

  virtual ~TwoLayersBuilder()
  {
  }

  network<sequential> buildNN() const override
  {
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

    nn << conv<identity>(input.width, input.height, kernel1_size, input.depth, nb_features1, padding::same)
       << pool(input.width, input.height, nb_features1, pooling1_x, pooling1_y, pooling1_x, pooling1_y)
       << conv<identity>(grid1_x, grid1_y, kernel2_size, nb_features1, nb_features2, padding::same)
       << pool(grid1_x, grid1_y, nb_features2, pooling2_x, pooling2_y, pooling2_x, pooling2_y)
       << fully_connected_layer<activation::identity>(fc_layer_input_dim, fc_units)
       << fully_connected_layer<softmax>(fc_units, 2);  // 2 classes output
    return nn;
  }

  Json::Value toJson() const
  {
    throw std::logic_error(DEBUG_INFO + " not implemented");
  }

  void fromJson(const Json::Value& v, const std::string& dir_path)
  {
    NNBuilder::fromJson(v, dir_path);
    rhoban_utils::tryRead(v, "kernel1_size", &kernel1_size);
    rhoban_utils::tryRead(v, "kernel2_size", &kernel2_size);
    rhoban_utils::tryRead(v, "nb_features1", &nb_features1);
    rhoban_utils::tryRead(v, "nb_features2", &nb_features2);
    rhoban_utils::tryRead(v, "grid1_x", &grid1_x);
    rhoban_utils::tryRead(v, "grid1_y", &grid1_y);
    rhoban_utils::tryRead(v, "grid2_x", &grid2_x);
    rhoban_utils::tryRead(v, "grid2_y", &grid2_y);
    rhoban_utils::tryRead(v, "fc_units", &fc_units);
  }

  std::string getClassName() const
  {
    return "TwoLayersBuilder";
  }
  std::string toString() const
  {
    return "_kernerl1-" + to_string(kernel1_size) + "_kernerl2-" + to_string(kernel2_size) + "_features1" +
           to_string(nb_features1) + "_features2" + to_string(nb_features2) + "_grid1" + to_string(grid1_x) + "x" +
           to_string(grid1_y) + "_grid2" + to_string(grid2_x) + "x" + to_string(grid2_y) + "_fcunits" +
           to_string(fc_units);
  }
};

class cifar10Builder : public NNBuilder
{
public:
  int n_fmaps1;  ///< number of feature maps for upper layer
  int n_fmaps2;  ///< number of feature maps for middle layer
  int n_fmaps3;  ///< number of feature maps for lower layer
  int n_fc;      ///< number of hidden units in fully-connected layer

  cifar10Builder() : n_fmaps1(32), n_fmaps2(32), n_fmaps3(64), n_fc(64)
  {
  }

  virtual ~cifar10Builder()
  {
  }

  network<sequential> buildNN() const override
  {
    // specify loss-function and learning strategy
    network<sequential> nn;

    typedef convolutional_layer<activation::identity> conv;
    typedef max_pooling_layer<relu> pool;

    nn << conv(32, 32, 5, 3, n_fmaps1, padding::same) << pool(32, 32, n_fmaps1, 2)
       << conv(16, 16, 5, n_fmaps1, n_fmaps2, padding::same) << pool(16, 16, n_fmaps2, 2)
       << conv(8, 8, 5, n_fmaps2, n_fmaps3, padding::same) << pool(8, 8, n_fmaps3, 2)
       << fully_connected_layer<activation::identity>(4 * 4 * n_fmaps3, n_fc)
       << fully_connected_layer<softmax>(n_fc, 10);

    return nn;
  }

  Json::Value toJson() const
  {
    throw std::logic_error(DEBUG_INFO + " not implemented");
  }

  void fromJson(const Json::Value& v, const std::string& dir_path)
  {
    NNBuilder::fromJson(v, dir_path);
    rhoban_utils::tryRead(v, "n_fmaps1", &n_fmaps1);
    rhoban_utils::tryRead(v, "n_fmaps2", &n_fmaps2);
    rhoban_utils::tryRead(v, "n_fmaps3", &n_fmaps3);
    rhoban_utils::tryRead(v, "n_fc", &n_fc);
  }

  std::string getClassName() const
  {
    return "cifar10Builder";
  }
  std::string toString() const
  {
    return "cifar10_n_fmaps1-" + to_string(n_fmaps1) + "n_fmaps2-" + to_string(n_fmaps2) + "n_fmaps3-" +
           to_string(n_fmaps3) + "n_fc" + to_string(n_fc);
  }
};

class NNBuilderFactory : public rhoban_utils::Factory<NNBuilder>
{
public:
  NNBuilderFactory()
  {
    registerBuilder("OneLayerBuilder", []() { return std::unique_ptr<NNBuilder>(new OneLayerBuilder()); });
    registerBuilder("TwoLayersBuilder", []() { return std::unique_ptr<NNBuilder>(new TwoLayersBuilder()); });
    registerBuilder("cifar10Builder", []() { return std::unique_ptr<NNBuilder>(new cifar10Builder()); });
  }
};

class Config : public rhoban_utils::JsonSerializable
{
public:
  std::map<std::string, std::unique_ptr<NNBuilder>> nnbuilders;
  std::unique_ptr<NNBuilder> nnbuilder;
  int nb_minibatch;
  int nb_train_epochs;
  double learning_rate_start;
  double learning_rate_end;
  int dichotomy_depth;

  Config()
    : nb_minibatch(10), nb_train_epochs(10), learning_rate_start(0.005), learning_rate_end(0.05), dichotomy_depth(5)
  {
  }
  Config(const Config& other)
    : nb_minibatch(other.nb_minibatch)
    , nb_train_epochs(other.nb_train_epochs)
    , learning_rate_start(other.learning_rate_start)
    , learning_rate_end(other.learning_rate_end)
    , dichotomy_depth(other.dichotomy_depth)
  {
  }
  ~Config()
  {
  }

  std::string getClassName() const override
  {
    return "DNNTrainerConfig";
  }

  Json::Value toJson() const override
  {
    throw std::logic_error(DEBUG_INFO + " not implemented");
  }

  virtual void fromJson(const Json::Value& v, const std::string& path)
  {
    nnbuilders = NNBuilderFactory().readMap(v, "networks", path);
    nb_minibatch = rhoban_utils::read<int>(v, "nb_minibatch");
    nb_train_epochs = rhoban_utils::read<int>(v, "nb_train_epochs");
    learning_rate_start = rhoban_utils::read<double>(v, "learning_rate_start");
    learning_rate_end = rhoban_utils::read<double>(v, "learning_rate_end");
    dichotomy_depth = rhoban_utils::read<int>(v, "dichotomy_depth");
  }
};

static void parse_file(const std::string& filename, std::vector<vec_t>* train_images,
                       std::vector<label_t>* train_labels, float_t scale_min, float_t scale_max, int x_padding,
                       int y_padding, int image_width, int image_height, int image_depth)
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

  while (ifs.read((char*)&label, 1))
  {
    vec_t img;

    if (!ifs.read((char*)&buf[0], image_size))
      break;

    if (x_padding || y_padding)
    {
      int w = image_width + 2 * x_padding;
      int h = image_height + 2 * y_padding;

      img.resize(w * h * image_depth, scale_min);

      for (int c = 0; c < image_depth; c++)
      {
        for (int y = 0; y < image_height; y++)
        {
          for (int x = 0; x < image_width; x++)
          {
            img[c * w * h + (y + y_padding) * w + x + x_padding] =
                scale_min + (scale_max - scale_min) * buf[c * image_area + y * image_width + x] / 255;
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

vector<double> train_cifar(string data_train, string data_test, string nn_config, double learning_rate, ostream& log,
                           Config& config, string nnbuilder_name, network<sequential>* nn)
{
  // specify loss-function and learning strategy
  adam optimizer;

  *nn = config.nnbuilders[nnbuilder_name]->buildNN();
  InputConfig input = config.nnbuilders[nnbuilder_name]->input;

  log << "learning rate:" << learning_rate << endl;

  cout << "Input: " << input.width << "x" << input.height << "x" << input.depth << std::endl;

  cout << "load models..." << endl;

  // load cifar dataset
  vector<label_t> train_labels, test_labels;
  vector<vec_t> train_images, test_images;

  parse_file(data_train, &train_images, &train_labels, -1.0, 1.0, 0, 0, input.width, input.height, input.depth);

  parse_file(data_test, &test_images, &test_labels, -1.0, 1.0, 0, 0, input.width, input.height, input.depth);

  if (train_images.size() < test_images.size())
  {
    throw std::runtime_error("train_cifar: less training images than test images (" +
                             std::to_string(train_images.size()) + " < " + std::to_string(test_images.size()) +
                             "), did you invert the arguments?");
  }

  cout << "start learning" << endl;

  progress_display disp(train_images.size());
  timer t;
  const int nb_minibatch = config.nb_minibatch;        ///< minibatch size
  const int nb_train_epochs = config.nb_train_epochs;  ///< training duration

  optimizer.alpha *= static_cast<tiny_dnn::float_t>(sqrt(nb_minibatch) * learning_rate);

  // create callback

  int i = 0;
  bool overfitting_flag = false;
  double previous_percentage, percentage = 0;

  auto on_enumerate_epoch = [&]() {
    previous_percentage = percentage;

    cout << t.elapsed() << "s elapsed." << endl;

    i = i + 1;
    cout << "Epoch number " << to_string(i) << endl;

    timer t1;
    tiny_dnn::result res = nn->test(test_images, test_labels);
    cout << t1.elapsed() << "s elapsed (test)." << endl;

    percentage = (double)(res.num_success) / (double)(res.num_total) * 100.0;
    log << res.num_success << "/" << res.num_total << " " << percentage << "%" << endl;

    // In a more general situation the idea would be the following for overfitting detection :
    // abs(average_class_sizes/nb_class - succes) < epsilon => overfitting
    if (percentage < 60.0 && i > nb_train_epochs / 2)
    {
      cout << "Probably overfitting." << endl;
      overfitting_flag = true;
      return false;
    }

    cout << "improvement : " << percentage - previous_percentage;
    disp.restart(train_images.size());
    t.restart();
    return true;
  };

  auto on_enumerate_minibatch = [&]() { disp += nb_minibatch; };

  // training
  nn->train<cross_entropy>(optimizer, train_images, train_labels, nb_minibatch, nb_train_epochs, on_enumerate_minibatch,
                           on_enumerate_epoch, true, 1);

  cout << "end training." << endl;

  if (overfitting_flag == true)
  {
    return { 0.0, 0.0, (double)i };
  }

  // test and show results
  tiny_dnn::result res = nn->test(test_images, test_labels);
  res.print_detail(cout);

  // save networks
  // ofstream ofs("ball_exp_weights");
  // ofs << nn;

  percentage = (double)(res.num_success) / (double)(res.num_total) * 100.0;
  return { percentage, percentage - previous_percentage, i };
}

void dichotomic_train_cifar(string data_train, string data_test, string nn_config, double learning_rate_start,
                            double learning_rate_end, double dichotomy_depth, ofstream& results_file, Config& config,
                            string nnbuilder_name, double current_best_validation_score)
{
  // search is finished
  if (dichotomy_depth < 0)
  {
    cout << "Search finished" << endl;
    return;
  }
  cout << "Search depth: " << dichotomy_depth << endl;

  // if learning_rate_start = learning_rate_end we stop searching
  // independently of current dichotomy_depth
  if (learning_rate_start == learning_rate_end)
  {
    dichotomy_depth = 0;
  }

  // we try the middle learning rate
  double learning_rate = (learning_rate_end + learning_rate_start) / 2.0;

  timer t;
  network<sequential> nn;
  vector<double> results =
      train_cifar(data_train, data_test, nn_config, learning_rate, cout, config, nnbuilder_name, &nn);

  // writting result in csv file
  double validation_score = results[0];
  if (validation_score > 0)
  {
    results_file << learning_rate << "," << setprecision(4) << validation_score << "," << to_string(results[1]) << ","
                 << to_string((int)results[2]) << "," << t.elapsed() << std::endl;
    learning_rate_start = learning_rate;
  }
  else
  {
    results_file << learning_rate << ","
                 << "overfit"
                 << "," << to_string(results[1]) << "," << to_string((int)results[2]) << "," << t.elapsed()
                 << std::endl;
    learning_rate_end = learning_rate;
  }

  // saving the neural network if it is better than the current best one
  if (validation_score > current_best_validation_score)
  {
    cout << "better" << endl;
    current_best_validation_score = validation_score;
    string folder = "results/" + nnbuilder_name + "/";
    nn.save(folder + "dnn_architecture.json", content_type::model, file_format::json);
    nn.save(folder + "dnn_weights.bin", content_type::weights, file_format::binary);
  }

  dichotomic_train_cifar(data_train, data_test, nn_config, learning_rate_start, learning_rate_end, dichotomy_depth - 1,
                         results_file, config, nnbuilder_name, current_best_validation_score);
}

int main(int argc, char** argv)
{
  if (argc < 4)
  {
    cerr << "Usage : " << argv[0]
         << " <learning_data> <test_data> <neural_network_config_file.json> [thread_number (default 4)]" << endl;
    exit(EXIT_FAILURE);
  }
  Config config;
  config.loadFile(argv[3]);

  double learning_rate_start = config.learning_rate_start;
  double learning_rate_end = config.learning_rate_end;
  double dichotomy_depth = config.dichotomy_depth;

  vector<std::string> keys;
  for (const auto& entry : config.nnbuilders)
  {
    keys.push_back(entry.first);
  }

  system("exec rm -r results");
  mkdir("results", 0777);
  auto learning_task = [&](int start_idx, int end_idx) {
    for (int idx = start_idx; idx < end_idx; idx++)
    {
      std::string nnbuilder_name = keys[idx];
      const NNBuilder& nnbuilder = *(config.nnbuilders[nnbuilder_name]);
      InputConfig input = nnbuilder.input;

      string result_folder = "results/" + nnbuilder_name + "/";
      mkdir(result_folder.c_str(), 0777);

      std::string file = "results_" + to_string(input.width) + "x" + to_string(input.height) + "x" +
                         to_string(input.depth) + nnbuilder.toString() + ".csv";
      std::ofstream results_file(result_folder + file);
      results_file << "learning_rate,validation_score,last_improvement,last_epoch,learning_time" << std::endl;

      dichotomic_train_cifar(argv[1], argv[2], argv[3], learning_rate_start, learning_rate_end, dichotomy_depth,
                             results_file, config, nnbuilder_name, 0.0);
      std::cout << "Starting training the configuration " + nnbuilder_name << std::endl;
    }
  };
  int nb_threads = 4;
  if (argc > 4)
  {
    nb_threads = std::stoi(argv[4]);
  }
  std::cout << "The number of thread that will be launched is " + std::to_string(nb_threads) << std::endl;
  rhoban_utils::MultiCore::runParallelTask(learning_task, static_cast<int>(config.nnbuilders.size()), nb_threads);
}

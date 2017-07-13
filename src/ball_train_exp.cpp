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




#include <iostream>
#include <vector>
#include "tiny_dnn/tiny_dnn.h"
#include <string>
#include <cmath>

// #define CNN_TASK_SIZE 8


using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace std;

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
r}


template <typename N>
void construct_basic_net(N& nn,
                         int input_width,
                         int input_height,
                         int input_depth) {
    typedef convolutional_layer<activation::identity> conv;
    typedef max_pooling_layer<relu> pool;

    const serial_size_t kernel_size = 5;
    const serial_size_t fc_grid_size = 4; /// number of columns in last layer
    const serial_size_t pooling_size = input_width / fc_grid_size;
    const serial_size_t n_fmaps = 16;   ///< number of feature maps for upper layer
    const serial_size_t n_fc = 16;      ///< number of hidden units in fully-connected layer

    // Some of the values can be determined automatically
    const serial_size_t fc_layer_input_dim =
      (input_width / pooling_size) * (input_height / pooling_size) * n_fmaps;


    int convol_size = kernel_size * kernel_size * input_depth * n_fmaps;
    int fc_size = fc_layer_input_dim * n_fc;

    std::cout << "fc_layer_input_dim: " << fc_layer_input_dim << std::endl;
    std::cout << "parameters in convolution layer    : " << convol_size << std::endl;
    std::cout << "parameters in fully-connected layer: " << fc_size << std::endl;

    nn << conv(input_width, input_height, kernel_size, input_depth, n_fmaps, padding::same)
       << pool(input_width, input_height, n_fmaps, pooling_size)
       << fully_connected_layer<activation::identity>(fc_layer_input_dim, n_fc)
       << fully_connected_layer<softmax>(n_fc, 2); //2 classes output
}

template <typename N>
void construct_2layers_net(N& nn,
                           int input_width,
                           int input_height,
                           int input_depth) {
    typedef convolutional_layer<activation::identity> conv;
    typedef max_pooling_layer<relu> pool;

    const serial_size_t kernel1_size = 5; /// kernel1
    const serial_size_t kernel2_size = 5; /// kernel2
    const serial_size_t pooling1 = 2;     /// pooling ratio from upper to lower layer
    const serial_size_t pooling2 = 2;     /// pooling ratio from lower layer to fully-connected
    const serial_size_t n_fmaps1 = 16;     /// number of feature maps for upper layer
    const serial_size_t n_fmaps2 = 4;     /// number of feature maps for lower layer
    const serial_size_t n_fc = 8;         /// number of hidden units in fully-connected layer

    // Some of the values can be determined automatically
    const serial_size_t layer2_width  = input_width / pooling1;
    const serial_size_t layer2_height = input_height / pooling1;
    const serial_size_t fc_layer_input_dim =
      (layer2_width / pooling2) * (layer2_height / pooling2) * n_fmaps2;

    int first_convol_size = kernel1_size * kernel1_size * input_depth * n_fmaps1;
    int second_convol_size = kernel2_size * kernel2_size * n_fmaps1 * n_fmaps2;
    int fc_size = fc_layer_input_dim * n_fc;

    std::cout << "fc_layer_input_dim: " << fc_layer_input_dim << std::endl;
    std::cout << "parameters in 1st convolution layer: " << first_convol_size << std::endl;
    std::cout << "parameters in 2nd convolution layer: " << second_convol_size << std::endl;
    std::cout << "parameters in fully-connected layer: " << fc_size << std::endl;

    nn << conv(input_width, input_height, kernel1_size, input_depth, n_fmaps1, padding::same)
       << pool(input_width, input_height, n_fmaps1, pooling1)
       << conv(layer2_width, layer2_height, kernel2_size, n_fmaps1, n_fmaps2, padding::same)
       << pool(layer2_width, layer2_height, n_fmaps2, pooling2)
       << fully_connected_layer<activation::identity>(fc_layer_input_dim, n_fc)
       << fully_connected_layer<softmax>(n_fc, 2); //2 classes output
}


template <typename N>
void construct_net(N& nn) {
    typedef convolutional_layer<activation::identity> conv;
    typedef max_pooling_layer<relu> pool;

    //good but too slow
    // const serial_size_t n_fmaps = 64;   ///< number of feature maps for upper layer
    // const serial_size_t n_fmaps2 = 64;  ///< number of feature maps for lower layer
    // const serial_size_t n_fc = 64;      ///< number of hidden units in fully-connected layer

    // nn << conv(32, 32, 5, 3, n_fmaps, padding::same)
    //    << pool(32, 32, n_fmaps, 2)
    //    << conv(16, 16, 5, n_fmaps, n_fmaps, padding::same)
    //    << pool(16, 16, n_fmaps, 2)
    //    << conv(8, 8, 5, n_fmaps, n_fmaps2, padding::same)
    //    << pool(8, 8, n_fmaps2, 2)
    //    << fully_connected_layer<activation::identity>(4 * 4 * n_fmaps2, n_fc)
    //    << fully_connected_layer<softmax>(n_fc, 2); //2 classes output

    const serial_size_t n_fmaps = 32;   ///< number of feature maps for upper layer
    const serial_size_t n_fmaps2 = 32;  ///< number of feature maps for lower layer
    const serial_size_t n_fc = 32;      ///< number of hidden units in fully-connected layer

    // const serial_size_t n_fmaps = 16;   ///< number of feature maps for upper layer
    // const serial_size_t n_fmaps2 = 16;  ///< number of feature maps for lower layer
    // const serial_size_t n_fc = 16;      ///< number of hidden units in fully-connected layer

    nn << conv(32, 32, 5, 3, n_fmaps, padding::same)
       << pool(32, 32, n_fmaps2, 8)
            // << conv(8, 8, 5, n_fmaps, n_fmaps2, padding::same)
            // << pool(8, 8, n_fmaps2, 2)
       << fully_connected_layer<activation::identity>(4 * 4 * n_fmaps2, n_fc)
       << fully_connected_layer<softmax>(n_fc, 2); //2 classes output
}

void train_cifar(string data_train, string data_test,double learning_rate, ostream& log) {
    // image size
    int input_width = 16;
    int input_height = 16;
    int input_depth = 3;

    // specify loss-function and learning strategy
    network<sequential> nn;
    adam optimizer;

    construct_basic_net(nn, input_width, input_height, input_depth);

    log << "learning rate:" << learning_rate << endl;

    cout << "load models..." << endl;

    // load cifar dataset
    vector<label_t> train_labels, test_labels;
    vector<vec_t> train_images, test_images;


    parse_file(data_train,
               &train_images, &train_labels, -1.0, 1.0, 0, 0,
               input_width, input_height, input_depth);

    parse_file(data_test,
               &test_images, &test_labels, -1.0, 1.0, 0, 0,
               input_width, input_height, input_depth);

    cout << "start learning" << endl;

    progress_display disp(train_images.size());
    timer t;
    const int n_minibatch = 10;     ///< minibatch size
    const int n_train_epochs = 10;  ///< training duration

    optimizer.alpha *= static_cast<tiny_dnn::float_t>(sqrt(n_minibatch) * learning_rate);

    // create callback
    auto on_enumerate_epoch = [&]() {
        cout << t.elapsed() << "s elapsed." << endl;
        timer t1;
        tiny_dnn::result res = nn.test(test_images, test_labels);
        cout << t1.elapsed() << "s elapsed (test)." << endl;
        log << res.num_success << "/" << res.num_total << " "<<(float)(res.num_success)/(float)(res.num_total)*100.0<<"%"<<endl;

        disp.restart(train_images.size());
        t.restart();
    };

    auto on_enumerate_minibatch = [&]() {
        disp += n_minibatch;
    };

    // training
    nn.train<cross_entropy>(optimizer, train_images, train_labels,
                            n_minibatch, n_train_epochs, on_enumerate_minibatch,
                            on_enumerate_epoch);

    cout << "end training." << endl;

    // test and show results
    nn.test(test_images, test_labels).print_detail(cout);

    nn.save("test_exp.json", content_type::model, file_format::json);
    nn.save("test_exp_weights.bin", content_type::weights, file_format::binary);

    // save networks
    ofstream ofs("ball_exp_weights");
    ofs << nn;
}

int main(int argc, char **argv) {
    if (argc != 4) {
      cerr << "Usage : " << argv[0] << endl
           << " arg[0]: train_file " << endl
           << " arg[1]: test_file " << endl
           << " arg[2]: learning rate (example:0.01)" << endl;
        return -1;
    }
    train_cifar(argv[1], argv[2],stod(argv[3]), cout);
}

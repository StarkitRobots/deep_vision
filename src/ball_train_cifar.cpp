//*****************************************************************************
//
// File Name	: 'ball_train_cifar.cpp'
// Author	: Steve NGUYEN
// Contact      : steve.nguyen.000@gmail.com
// Created	: lundi, novembre 21 2016
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

using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace std;

template <typename N>
void construct_net(N& nn) {
    typedef convolutional_layer<activation::identity> conv;
    typedef max_pooling_layer<relu> pool;


    const serial_size_t n_fmaps = 32;   ///< number of feature maps for upper layer
    const serial_size_t n_fmaps2 = 64;  ///< number of feature maps for lower layer
    const serial_size_t n_fc = 64;      ///< number of hidden units in fully-connected layer

    nn << conv(32, 32, 5, 3, n_fmaps, padding::same)
       << pool(32, 32, n_fmaps, 2)
       << conv(16, 16, 5, n_fmaps, n_fmaps, padding::same)
       << pool(16, 16, n_fmaps, 2)
       << conv(8, 8, 5, n_fmaps, n_fmaps2, padding::same)
       << pool(8, 8, n_fmaps2, 2)
       << fully_connected_layer<activation::identity>(4 * 4 * n_fmaps2, n_fc)
       << fully_connected_layer<softmax>(n_fc, 2); //2 classes output
}

void train_cifar(string data_train, string data_test,double learning_rate, ostream& log) {
    // specify loss-function and learning strategy
    network<sequential> nn;
    adam optimizer;

    construct_net(nn);

    log << "learning rate:" << learning_rate << endl;

    cout << "load models..." << endl;

    // load cifar dataset
    vector<label_t> train_labels, test_labels;
    vector<vec_t> train_images, test_images;


    parse_cifar10(data_train,
                  &train_images, &train_labels, -1.0, 1.0, 0, 0);

    parse_cifar10(data_test,
                  &test_images, &test_labels, -1.0, 1.0, 0, 0);

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
        log << res.num_success << "/" << res.num_total << endl;

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

    // save networks
    ofstream ofs("ball_cifar_weights");
    ofs << nn;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        cerr << "Usage : " << argv[0]
             << "arg[0]: train_file"
             << "arg[1]: test_file"
             << "arg[2]: learning rate (example:0.01)" << endl;
        return -1;
    }
    train_cifar(argv[1], argv[2],stod(argv[3]), cout);
}

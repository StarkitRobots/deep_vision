//*****************************************************************************
//
// File Name	: 'ball_test_cifar.cpp'
// Author	: Steve NGUYEN
// Contact      : steve.nguyen.000@gmail.com
// Created	: mardi, novembre 22 2016
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
#include <opencv2/opencv.hpp>

#include "tiny_dnn/tiny_dnn.h"

using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace std;

// rescale output to 0-100
template <typename Activation>
double rescale(double x) {
    Activation a;
    return 100.0 * (x - a.scale().first) / (a.scale().second - a.scale().first);
}

void convert_image(const std::string& imagefilename,
                   double minv,
                   double maxv,
                   int w,
                   int h,
                   vec_t& data) {

    cv::Mat img = cv::imread(imagefilename);
    if (img.data == nullptr) return; // cannot open, or it's not an image
    cv::Mat resized,res;
    cv::resize(img, resized, cv::Size(w, h), .0, .0,cv::INTER_AREA);
    data.resize(w*h*resized.channels(), minv);

    // std::vector<unsigned char> buf(w*h*resized.channels());
    // std::transform(buf.begin(), buf.end(), std::back_inserter(resized),
    //                [=](unsigned char c) { return minv + (maxv - minv) * c / 255; });

    // for(int c = 0; c < resized.channels(); ++c){
    //     for(int y = 0; y < resized.rows; ++y){
    //         for(int x = 0; x < resized.cols; ++x){
    //             data[c * w * h + y*w + x] = minv + (maxv - minv) * resized.data[c*resized.step + y*resized.step + x] / 255;
    //         }
    //     }
    // }

    // data=buf.data;
    // cv::normalize(res,resized,-1,1);
    cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );// Create a window for display.
    cv::imshow( "Display window", img );                   // Show our image inside it.

    std::cout<<"DEBUG"<<std::endl;
    std::cout<<resized.rows<<" "<<resized.cols<<std::endl;
    // for(int i=0; i<3*resized.rows*resized.cols;++i)
    //     std::cout<<resized.data[i]<<" ";

    resized.copyTo(res);
    cv::imshow( "resize", resized );                   // Show our image inside it.
    // for(int i=0; i<resized.channels()*resized.rows*resized.cols;++i)
    //     res.data[i]=minv + (maxv - minv) * (float)resized.data[i] / 255.0;


    cv::Mat ch1, ch2, ch3;
    vector<cv::Mat> channels(3);
    cv::split(res, channels);
    ch1 = channels[0];
    ch2 = channels[1];
    ch3 = channels[2];

    int j=0;

    cv::Size sz = ch1.size();
    int size=sz.height*sz.width;

    for(int i=0; i<size;i++)
        data[j*size+i]=minv+(maxv-minv)*ch1.data[i]/255.0;
    j++;
    for(int i=0; i<size;i++)
        data[j*size+i]=minv+(maxv-minv)*ch2.data[i]/255.0;
    j++;
    for(int i=0; i<size;i++)
        data[j*size+i]=minv+(maxv-minv)*ch3.data[i]/255.0;

    // cv::imshow( "scale", res );                   // Show our image inside it.

    cv::waitKey(0);                                          // Wait for a keystroke in the window

    // for(int i=0; i<res.channels()*res.rows*res.cols;++i)
    //     data[i]=res.data[i];
    // for(int i=0; i<resized.channels()*resized.rows*resized.cols;++i)
    //     std::cout<<data[i]<<" ";

}

template <typename N>
void construct_net(N& nn) {
    typedef convolutional_layer<activation::identity> conv;
    typedef max_pooling_layer<relu> pool;

    const int n_fmaps = 32; ///< number of feature maps for upper layer
    const int n_fmaps2 = 64; ///< number of feature maps for lower layer
    const int n_fc = 64; ///< number of hidden units in fully-connected layer

    nn << conv(32, 32, 5, 3, n_fmaps, padding::same)
       << pool(32, 32, n_fmaps, 2)
       << conv(16, 16, 5, n_fmaps, n_fmaps, padding::same)
       << pool(16, 16, n_fmaps, 2)
       << conv(8, 8, 5, n_fmaps, n_fmaps2, padding::same)
       << pool(8, 8, n_fmaps2, 2)
       << fully_connected_layer<activation::identity>(4 * 4 * n_fmaps2, n_fc)
            << fully_connected_layer<softmax>(n_fc, 2);
        }

    void recognize(const std::string& dictionary, const std::string& filename) {
        network<sequential> nn;

        construct_net(nn);

        // load nets
        ifstream ifs(dictionary.c_str());
        ifs >> nn;

        // convert imagefile to vec_t
    vec_t data;
    convert_image(filename, -1.0, 1.0, 32, 32, data);

    // recognize
    auto res = nn.predict(data);

    std::cout<<"Score"<<std::endl;
    for (size_t i = 0; i < res.size(); i++)
    {
        std::cout<<i<<": "<<rescale<tan_h>(res[i])<<"% "<<res[i]<<std::endl;
    }


    // save outputs of each layer
    for (size_t i = 0; i < nn.depth(); i++) {
        auto out_img = nn[i]->output_to_image();
        auto filename = "cifar_layer_" + std::to_string(i) + ".png";
        out_img.save(filename);
    }
    // // save filter shape of first convolutional layer
    // {
    //     auto weight = nn.at<convolutional_layer<tan_h>>(0).weight_to_image();
    //     auto filename = "cifar_weights.png";
    //     weight.save(filename);
    // }


    }

int main(int argc, char** argv) {
    if (argc != 2) {
        cout << "please specify image file";
        return 0;
    }
    recognize("ball_cifar_weights", argv[1]);
}

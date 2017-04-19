//*****************************************************************************
//
// File Name	: 'predict.cpp'
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
#include <opencv2/opencv.hpp>

#include "tiny_dnn/tiny_dnn.h"
#include <string>
#include <vector>

using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace std;


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

    // std::cout<<"DEBUG"<<std::endl;
    // std::cout<<resized.rows<<" "<<resized.cols<<std::endl;
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

    // cv::waitKey(0);                                          // Wait for a keystroke in the window

    // for(int i=0; i<res.channels()*res.rows*res.cols;++i)
    //     data[i]=res.data[i];
    // for(int i=0; i<resized.channels()*resized.rows*resized.cols;++i)
    //     std::cout<<data[i]<<" ";

}

template <typename N>
void construct_net(N& nn, const std::string& arch, const std::string& weights) {


    // load the architecture of the model in json format
    nn.load(arch, content_type::model, file_format::json);

    // load the weights of the model in binary format
    nn.load(weights, content_type::weights, file_format::binary);



}

int recognize(const std::string& arch, const std::string& weights,const std::string& filename) {
    network<sequential> nn;

    construct_net(nn, arch, weights);

    // // load nets
    // ifstream ifs(dictionary.c_str());
    // ifs >> nn;

    // convert imagefile to vec_t
    vec_t data;
    convert_image(filename, -1.0, 1.0, 32, 32, data);

    timer t;
    t.restart();
    // recognize
    auto res = nn.predict(data);
    std::cout<<"(time: "<<t.elapsed()<<" s)"<<std::endl;


    // save outputs of each layer
    for (size_t i = 0; i < nn.depth(); i++) {
        /*
          auto out_img = nn[i]->output_to_image();
          auto filename = "exp_layer_" + std::to_string(i) + ".png";
          out_img.save(filename);

          // save filter shape of first convolutional layer


          if(i==0||i==2||i==4)
          {
          auto weight = nn.at<conv<activation::identity>>(i).weight_to_image();
          auto filename1 = "exp_weight_" + std::to_string(i) + ".png";
          weight.save(filename1) ;
          }
        */

        //DEBUG
        cout << "#layer:" << i << "\n";
        cout << "layer type:" << nn[i]->layer_type() << "\n";
        cout << "input:" << nn[i]->in_data_size() << "(" << nn[i]->in_data_shape() << ")\n";
        cout << "output:" << nn[i]->out_data_size() << "(" << nn[i]->out_data_shape() << ")\n";

    }


    std::cout<<"Score"<<std::endl;

    int maxclass=0;
    double prevmax=0;
    for (size_t i = 0; i < res.size(); i++)
    {

        if(res[i]>prevmax)
        {
            prevmax=res[i];
            maxclass=i;
        }
        // std::cout<<i<<": "<<rescale<tan_h>(res[i])<<"% "<<res[i]<<std::endl;
        std::cout<<i<<": "<<res[i]<<std::endl;

    }
    return maxclass;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        cout << "USAGE: ./predict ARCH.json WEIGHTS.bin image"<<endl;
        return 0;
    }
    return recognize(argv[1],argv[2],argv[3]);
}

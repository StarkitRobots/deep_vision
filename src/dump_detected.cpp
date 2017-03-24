//*****************************************************************************
//
// File Name	: 'dump_detected.cpp'
// Author	: Steve NGUYEN
// Contact      : steve.nguyen.000@gmail.com
// Created	: vendredi, march 24 2017
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
#include <unistd.h>
#include "tiny_dnn/tiny_dnn.h"


#include <chrono>  // chrono::system_clock
#include <ctime>   // localtime
#include <sstream> // stringstream
#include <iomanip> // put_time
#include <string>  // string
#include <sys/time.h>

using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace std;

std::string return_current_time_and_date()
{


    timeval curTime;
    gettimeofday(&curTime, NULL);
    int milli = curTime.tv_usec / 1000;

    char buffer [80];
    strftime(buffer, 80, "%Y%m%d_%H%M%S", localtime(&curTime.tv_sec));

    char currentTime[84] = "";
    sprintf(currentTime, "%s%03d", buffer, milli);

    return string(currentTime);

}


void convert_mat(cv::Mat& image,
                 double minv,
                 double maxv,
                 int w,
                 int h,
                 vec_t& data) {

    data.resize(w*h*image.channels(), minv);


    cv::Mat ch1, ch2, ch3;
    vector<cv::Mat> channels(3);
    cv::split(image, channels);
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

}

void slide(cv::Mat img, int w, int h, int step,  network<sequential> nn, char *dest)
{
    cv::Mat DrawResultGrid= img.clone();
    int nb=0;
    // Cycle row step
    for (int row = 0; row <= img.rows - w; row += step)
    {
        // Cycle col step
        for (int col = 0; col <= img.cols - h; col += step)
        {
            cv::Rect windows(col, row, w, h);

            cv::Mat DrawResultHere = img.clone();
            // Select windows roi
            cv::Mat Roi = img(windows);

            vec_t data;
            convert_mat(Roi,-1,1,32,32,data);

            auto res = nn.predict(data);

            // std::cout<<"Score"<<std::endl;
            // for (size_t i = 0; i < res.size(); i++)
            // {
            //     std::cout<<i<<": "<<rescale<tan_h>(res[i])<<"% "<<res[i]<<std::endl;
            // }


            // Draw only rectangle
            cv::rectangle(DrawResultHere, windows, cv::Scalar(255), 1, 8, 0);
            // Draw grid
            if(res[1]>0.50)
            {
                cv::rectangle(DrawResultGrid, windows, cv::Scalar(0,0,255), 1, 8, 0);
                cv::imwrite(string(dest)+string("/")+return_current_time_and_date()+string(".png"),Roi);
                nb++;
            }
            // else
            //     cv::rectangle(DrawResultGrid, windows, cv::Scalar(255), 1, 8, 0);

            // Show  rectangle
            // cv::namedWindow("Step 2 draw Rectangle", cv::WINDOW_AUTOSIZE);
            // cv::imshow("Step 2 draw Rectangle", DrawResultHere);
            // cv::waitKey(100);


            // cv::namedWindow("Step 3 Show Grid", cv::WINDOW_AUTOSIZE);
            // cv::imshow("Step 3 Show Grid", DrawResultGrid);


            // cv::waitKey(100);


            //Show ROI
            // cv::namedWindow("Step 4 Draw selected Roi", cv::WINDOW_AUTOSIZE);
            // cv::imshow("Step 4 Draw selected Roi", Roi);
            // cv::waitKey(100);
        }
    }
    std::cout<<nb<<" detected"<<std::endl;
}
/*
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
*/
template <typename N>
void construct_net(N& nn, const std::string& arch, const std::string& weights) {


    // load the architecture of the model in json format
    nn.load(arch, content_type::model, file_format::json);

    // load the weights of the model in binary format
    nn.load(weights, content_type::weights, file_format::binary);

}



int main(int argc, char** argv) {
    // if (argc != 2) {
    //     cout << "please specify image file";
    //     return 0;
    // }

    if (argc != 5) {
        cout << "USAGE: ./dump_detected ARCH.json WEIGHTS.bin image destination_folder"<<endl;
        return 0;
    }
    // recognize("ball_cifar_weights", argv[1]);


    cv::Mat img = cv::imread(argv[3]);
    cv::Mat resized;

    cv::resize(img, resized, cv::Size(160, 120), .0, .0,cv::INTER_AREA);

    // cv::resize(img, resized, cv::Size(320, 240), .0, .0,cv::INTER_AREA);
    cv::cvtColor(resized, resized, CV_YCrCb2BGR);


    network<sequential> nn;
    // construct_net(nn);
    // load nets
    // ifstream ifs("ball_cifar_weights");
    // ifs >> nn;


    construct_net(nn, argv[1],argv[2]);



    slide(resized,32,32,8,nn,argv[4]);
    // slide(resized,64,64,8,nn);
    // cv::waitKey();


    /*
      cv::VideoCapture cap(0); // open the default camera
      if(!cap.isOpened())  // check if we succeeded
      return -1;
      cv::Mat resized;
      for(;;)
      {
      cv::Mat frame;
      cap >> frame; // get a new frame from camera
      cv::resize(frame, resized, cv::Size(160, 120), .0, .0,cv::INTER_AREA);
      slide(resized,32,32,16,nn);
      if(cv::waitKey(10) >= 0) break;
      }
    */

}

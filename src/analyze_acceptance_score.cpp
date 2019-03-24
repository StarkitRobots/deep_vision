//*****************************************************************************
//
// File Name	: 'tune_acceptation_rate.cpp'
// Author	: Ludovic Hofer
// Contact      : medrimonia@gmail.com
// Created	: friday, april 28th 2017
// Revised	:
// Version	:
// Target MCU	:
//
// This code is distributed under the GNU Public License
//		which can be found at http://www.gnu.org/licenses/gpl.txt
//
//
// Notes:	notes
// Highly inspired from test_prediction by steve Nguyen
//
//*****************************************************************************

#include <iostream>
#include <opencv2/opencv.hpp>

#include "tiny_dnn/tiny_dnn.h"
#include <string>
#include <vector>

#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <map>

using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace std;

int getDir(string dir, vector<string>& files)
{
  DIR* dp;
  struct dirent* dirp;
  if ((dp = opendir(dir.c_str())) == NULL)
  {
    cout << "Error(" << errno << ") opening " << dir << endl;
    return errno;
  }

  while ((dirp = readdir(dp)) != NULL)
  {
    if (strstr(dirp->d_name, ".png"))
      files.push_back(dir + '/' + string(dirp->d_name));
  }
  closedir(dp);
  return 0;
}

void convert_image(const std::string& imagefilename, double minv, double maxv, int w, int h, vec_t& data)
{
  cv::Mat img = cv::imread(imagefilename);
  if (img.data == nullptr)
    return;  // cannot open, or it's not an image
  cv::Mat resized, res;
  cv::resize(img, resized, cv::Size(w, h), .0, .0, cv::INTER_AREA);
  data.resize(w * h * resized.channels(), minv);

  resized.copyTo(res);

  cv::Mat ch1, ch2, ch3;
  vector<cv::Mat> channels(3);
  cv::split(res, channels);
  ch1 = channels[0];
  ch2 = channels[1];
  ch3 = channels[2];

  int j = 0;

  cv::Size sz = ch1.size();
  int size = sz.height * sz.width;

  for (int i = 0; i < size; i++)
    data[j * size + i] = minv + (maxv - minv) * ch1.data[i] / 255.0;
  j++;
  for (int i = 0; i < size; i++)
    data[j * size + i] = minv + (maxv - minv) * ch2.data[i] / 255.0;
  j++;
  for (int i = 0; i < size; i++)
    data[j * size + i] = minv + (maxv - minv) * ch3.data[i] / 255.0;
}

template <typename N>
void construct_net(N& nn, const std::string& arch, const std::string& weights)
{
  // load the architecture of the model in json format
  nn.load(arch, content_type::model, file_format::json);

  // load the weights of the model in binary format
  nn.load(weights, content_type::weights, file_format::binary);
}

double getScore(network<sequential>& nn, const std::string& filename)
{
  int width = nn[0]->in_data_shape()[0].width_;
  int height = nn[0]->in_data_shape()[0].height_;

  // convert imagefile to vec_t
  vec_t data;
  convert_image(filename, -1.0, 1.0, width, height, data);

  // recognize
  auto res = nn.predict(data);

  return res[1];
}

/// Return the scores for all of the mentioned files given the current network
std::vector<double> getScores(network<sequential>& nn, const vector<string>& file_paths)
{
  std::vector<double> scores;
  for (string file_path : file_paths)
  {
    scores.push_back(getScore(nn, file_path));
  }
  return scores;
}

/// Return a map with:
/// - acceptanceScore as key
/// (false_pos_rate, false_neg_rate) as value
///
/// With:
/// false_pos_rate : p(reco=Pos|reality=Neg) in [0,1]
/// false_neg_rate : p(reco=Neg|reality=Pos) in [0,1]
std::map<double, std::pair<double, double>> getErrorRates(network<sequential>& nn, const vector<string>& pos_paths,
                                                          const vector<string>& neg_paths, double min = 0.01,
                                                          double max = 0.99, double step = 0.05)
{
  std::vector<double> pos_scores = getScores(nn, pos_paths);
  std::vector<double> neg_scores = getScores(nn, neg_paths);

  std::map<double, std::pair<double, double>> result;

  for (double p = min; p <= max; p += step)
  {
    int fp(0), fn(0);
    for (double score : pos_scores)
    {
      if (score < p)
        fn++;
    }
    for (double score : pos_scores)
    {
      if (score >= p)
        fp++;
    }
    result[p] = { fp / (double)neg_paths.size(), fn / (double)pos_paths.size() };
  }
  return result;
}

int main(int argc, char** argv)
{
  if (argc != 5 && argc != 6)
  {
    cout << "USAGE: ./analyze_acceptance_score ARCH.json WEIGHTS.bin <pos_dir> <neg_dir> <opt: step_size>" << endl;
    return 0;
  }

  string architecture_path(argv[1]);
  string weights_path(argv[2]);
  string positive_dir(argv[3]);
  string negative_dir(argv[4]);

  double step = 0.05;
  if (argc > 5)
  {
    step = std::stod(argv[5]);
  }

  vector<string> testfiles;

  network<sequential> nn;

  construct_net(nn, architecture_path, weights_path);

  vector<string> positive_paths, negative_paths;
  getDir(positive_dir, positive_paths);
  getDir(negative_dir, negative_paths);

  std::map<double, std::pair<double, double>> error_rates_by_acceptance_rate;
  error_rates_by_acceptance_rate = getErrorRates(nn, positive_paths, negative_paths, step, 1.0 - step, step);

  std::cout << "acceptance_score,false_pos_rate,false_neg_rate" << std::endl;
  for (const auto& entry : error_rates_by_acceptance_rate)
  {
    std::cout << entry.first << "," << entry.second.first << "," << entry.second.second << std::endl;
  }
}

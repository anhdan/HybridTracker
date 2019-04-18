#include <iostream>
#include "opencv2/opencv.hpp"
#include <time.h>
//#include <dirent.h> 
#include <fstream>

//using namespace cv;
using namespace std;
void calculateHistogram(cv::Mat im, cv::Mat &averageX, cv::Mat &averageY, cv::Mat &averageX2, cv::Mat &averageY2, vector<float> &LL, vector<float> &AA, vector<float> &BB, cv::Mat &histogram, cv::Mat &histogramIndex);
int precomputeParameters(cv::Mat histogram, vector<float> LL, vector<float> AA, vector<float> BB, int numberOfPixels, vector<int> &reverseMap, cv::Mat &map, cv::Mat &colorDistance, cv::Mat &exponentialColorDistance);
void bilateralFiltering(cv::Mat colorDistance, cv::Mat exponentialColorDistance, vector<int> reverseMap, int* histogramPtr,
	float* averageXPtr, float* averageYPtr, float* averageX2Ptr, float* averageY2Ptr, cv::Mat &mx, cv::Mat &my, cv::Mat &Vx, cv::Mat &Vy, cv::Mat &contrast);
void calculateProbability(cv::Mat mx, cv::Mat my, cv::Mat Vx, cv::Mat Vy, cv::Mat modelMean, cv::Mat modelInverseCovariance,
	int width, int height, cv::Mat &Xsize, cv::Mat &Ysize, cv::Mat &Xcenter, cv::Mat &Ycenter, cv::Mat &shapeProbability);
void computeSaliencyMap(cv::Mat shapeProbability, cv::Mat contrast, cv::Mat exponentialColorDistance, cv::Mat histogramIndex, int* mapPtr,
	cv::Mat& SM, cv::Mat& saliency);
cv::Rect saliency(cv::Mat&  im);


#include "saliency.h"
float sigmac = 16;   
int histogramSize1D = 8;    // Number of histogram bins per channel

int histogramSize2D = histogramSize1D * histogramSize1D;
int histogramSize3D = histogramSize2D * histogramSize1D;
int logSize = (int)log2(histogramSize1D);
int logSize2 = 2 * logSize;

cv::Mat squares = cv::Mat::zeros(1, 10000, CV_32FC1);

float* squaresPtr = squares.ptr<float>(0);

vector<cv::Mat> LAB;
vector<float> L, A, B;
float meanVectorFloat[4] = { 0.5555, 0.6449, 0.0002, 0.0063 };
float inverseCovarianceFloat[4][4] = { { 43.3777, 1.7633, -0.4059, 1.0997 },
{ 1.7633, 40.7221, -0.0165, 0.0447 },
{ -0.4059, -0.0165, 87.0455, -3.2744 },
{ 1.0997, 0.0447, -3.2744, 125.1503 } };

cv::Mat modelMean = cv::Mat(4, 1, CV_32FC1, meanVectorFloat);
cv::Mat modelInverseCovariance = cv::Mat(4, 4, CV_32FC1, inverseCovarianceFloat);
void calculateHistogram(cv::Mat im, cv::Mat &averageX, cv::Mat &averageY, cv::Mat &averageX2, cv::Mat &averageY2, vector<float> &LL, vector<float> &AA, vector<float> &BB, cv::Mat &histogram, cv::Mat &histogramIndex)
	{

	cv::Mat lab, Lshift, Ashift, Bshift;

	double minL, maxL, minA, maxA, minB, maxB;

	averageX = cv::Mat::zeros(1, histogramSize3D, CV_32FC1);
	averageY = cv::Mat::zeros(1, histogramSize3D, CV_32FC1);
	averageX2 = cv::Mat::zeros(1, histogramSize3D, CV_32FC1);
	averageY2 = cv::Mat::zeros(1, histogramSize3D, CV_32FC1);

	//  LAB channels
	cv::cvtColor(im, lab, CV_BGR2Lab);

	split(lab, LAB);

	minMaxLoc(LAB[0], &minL, &maxL);
	minMaxLoc(LAB[1], &minA, &maxA);
	minMaxLoc(LAB[2], &minB, &maxB);

	float tempL = (255 - maxL + minL) / (maxL - minL + 1e-3);
	float tempA = (255 - maxA + minA) / (maxA - minA + 1e-3);
	float tempB = (255 - maxB + minB) / (maxB - minB + 1e-3);

	Lshift = cv::Mat::zeros(1, 256, CV_32SC1);
	Ashift = cv::Mat::zeros(1, 256, CV_32SC1);
	Bshift = cv::Mat::zeros(1, 256, CV_32SC1);

	for (int i = 0; i < 256; i++) {

		Lshift.at<int>(0, i) = tempL * (i - minL) - minL;
		Ashift.at<int>(0, i) = tempA * (i - minA) - minA;
		Bshift.at<int>(0, i) = tempB * (i - minB) - minB;

	}

	// Calculate quantized LAB values

	minL = minL / 2.56;
	maxL = maxL / 2.56;

	minA = minA - 128;
	maxA = maxA - 128;

	minB = minB - 128;
	maxB = maxB - 128;

	tempL = float(maxL - minL) / histogramSize1D;
	tempA = float(maxA - minA) / histogramSize1D;
	tempB = float(maxB - minB) / histogramSize1D;

	float sL = float(maxL - minL) / histogramSize1D / 2 + minL;
	float sA = float(maxA - minA) / histogramSize1D / 2 + minA;
	float sB = float(maxB - minB) / histogramSize1D / 2 + minB;

	for (int i = 0; i < histogramSize3D; i++) {

		int lpos = i % histogramSize1D;
		int apos = i % histogramSize2D / histogramSize1D;
		int bpos = i / histogramSize2D;

		LL.push_back(lpos * tempL + sL);
		AA.push_back(apos * tempA + sA);
		BB.push_back(bpos * tempB + sB);

	}

	// Calculate LAB histogram

	histogramIndex = cv::Mat::zeros(im.rows, im.cols, CV_32SC1);
	histogram = cv::Mat::zeros(1, histogramSize3D, CV_32SC1);

	int*    histogramPtr = histogram.ptr<int>(0);

	float* averageXPtr = averageX.ptr<float>(0);
	float* averageYPtr = averageY.ptr<float>(0);
	float* averageX2Ptr = averageX2.ptr<float>(0);
	float* averageY2Ptr = averageY2.ptr<float>(0);

	int*    LshiftPtr = Lshift.ptr<int>(0);
	int*    AshiftPtr = Ashift.ptr<int>(0);
	int*    BshiftPtr = Bshift.ptr<int>(0);

	int histShift = 8 - logSize;

	for (int y = 0; y < im.rows; y++) {

		int*    histogramIndexPtr = histogramIndex.ptr<int>(y);

		uchar*    LPtr = LAB[0].ptr<uchar>(y);
		uchar*    APtr = LAB[1].ptr<uchar>(y);
		uchar*    BPtr = LAB[2].ptr<uchar>(y);

		for (int x = 0; x < im.cols; x++) {

			// Instead of division, we use bit-shift operations for efficieny. This is valid if number of bins is a power of two (4, 8, 16 ...)

			int lpos = (LPtr[x] + LshiftPtr[LPtr[x]]) >> histShift;
			int apos = (APtr[x] + AshiftPtr[APtr[x]]) >> histShift;
			int bpos = (BPtr[x] + BshiftPtr[BPtr[x]]) >> histShift;

			int index = lpos + (apos << logSize) + (bpos << logSize2);

			histogramIndexPtr[x] = index;

			histogramPtr[index]++;

			// These values are collected here for efficiency. They will later be used in computing the spatial center and variances of the colors

			averageXPtr[index] += x;
			averageYPtr[index] += y;
			averageX2Ptr[index] += squaresPtr[x];
			averageY2Ptr[index] += squaresPtr[y];

		}
	}

}

int precomputeParameters(cv::Mat histogram, vector<float> LL, vector<float> AA, vector<float> BB, int numberOfPixels, vector<int> &reverseMap, cv::Mat &map, cv::Mat &colorDistance, cv::Mat &exponentialColorDistance)
	{

	int*    histogramPtr = histogram.ptr<int>(0);

	cv::Mat problematic = cv::Mat::zeros(histogram.cols, 1, CV_32SC1);
	cv::Mat closestElement = cv::Mat::zeros(histogram.cols, 1, CV_32SC1);

	cv::Mat sortedHistogramIdx;


	sortIdx(histogram, sortedHistogramIdx, CV_SORT_EVERY_ROW + CV_SORT_DESCENDING);

	int* sortedHistogramIdxPtr = sortedHistogramIdx.ptr<int>(0);

	float energy = 0;

	int binCountThreshold = 0;

	float energyThreshold = 0.95 * numberOfPixels;

	for (int i = 0; i < histogram.cols; i++) {

		energy += (float)histogramPtr[sortedHistogramIdxPtr[i]];

		if (energy > energyThreshold){

			binCountThreshold = histogramPtr[sortedHistogramIdx.at<int>(0, i)];

			break;

		}
	}

	// Calculate problematic histogram bins 
	for (int i = 0; i < histogram.cols; i++)
	if (histogramPtr[i] < binCountThreshold)
		problematic.at<int>(i, 0) = 1;

	map = cv::Mat::zeros(1, histogram.cols, CV_32SC1);

	int* mapPtr = map.ptr<int>(0);

	int count = 0;

	for (int i = 0; i < histogram.cols; i++) {

		if (histogramPtr[i] >= binCountThreshold) {

			// Save valid colors for later use.

			L.push_back(LL[i]);
			A.push_back(AA[i]);
			B.push_back(BB[i]);

			mapPtr[i] = count;

			reverseMap.push_back(i);

			count++;
		}
		else if (histogramPtr[i] < binCountThreshold && histogramPtr[i] > 0){

			float mini = 1e6;

			int closest = 0;

			// Calculate the perceptually closest color of bins with a few pixels.

			for (int k = 0; k < histogram.cols; k++) {

				// Don't forget to check this, we don't want to assign them to empty histogram bins.

				if (!problematic.at<int>(k, 0)){

					float dd = pow((LL[i] - LL[k]), 2) + pow((AA[i] - AA[k]), 2) + pow((BB[i] - BB[k]), 2);

					if (dd < mini) {
						mini = dd;
						closest = k;
					}
				}

			}

			closestElement.at<int>(i, 0) = closest;

		}

	}

	for (int i = 0; i < histogram.cols; i++)
	if (problematic.at<int>(i, 0))
		mapPtr[i] = mapPtr[closestElement.at<int>(i, 0)];

	int numberOfColors = (int)L.size();

	// Precompute the color weights here

	exponentialColorDistance = cv::Mat::zeros(numberOfColors, numberOfColors, CV_32FC1);

	colorDistance = cv::Mat::zeros(numberOfColors, numberOfColors, CV_32FC1);

	for (int i = 0; i < numberOfColors; i++) {

		colorDistance.at<float>(i, i) = 0;

		exponentialColorDistance.at<float>(i, i) = 1.0;

		for (int k = i + 1; k < numberOfColors; k++) {

			float colorDifference = pow(L[i] - L[k], 2) + pow(A[i] - A[k], 2) + pow(B[i] - B[k], 2);

			colorDistance.at<float>(i, k) = sqrt(colorDifference);

			colorDistance.at<float>(k, i) = sqrt(colorDifference);

			exponentialColorDistance.at<float>(i, k) = exp(-colorDifference / (2 * sigmac * sigmac));

			exponentialColorDistance.at<float>(k, i) = exponentialColorDistance.at<float>(i, k);

		}
	}

	return numberOfColors;

}

void bilateralFiltering(cv::Mat colorDistance, cv::Mat exponentialColorDistance, vector<int> reverseMap, int* histogramPtr,
	float* averageXPtr, float* averageYPtr, float* averageX2Ptr, float* averageY2Ptr, cv::Mat &mx, cv::Mat &my, cv::Mat &Vx, cv::Mat &Vy, cv::Mat &contrast)
{
	int numberOfColors = colorDistance.cols;

	cv::Mat X = cv::Mat::zeros(1, numberOfColors, CV_32FC1);
	cv::Mat Y = cv::Mat::zeros(1, numberOfColors, CV_32FC1);
	cv::Mat X2 = cv::Mat::zeros(1, numberOfColors, CV_32FC1);
	cv::Mat Y2 = cv::Mat::zeros(1, numberOfColors, CV_32FC1);
	cv::Mat NF = cv::Mat::zeros(1, numberOfColors, CV_32FC1);

	float* XPtr = X.ptr<float>(0);
	float* YPtr = Y.ptr<float>(0);
	float* X2Ptr = X2.ptr<float>(0);
	float* Y2Ptr = Y2.ptr<float>(0);
	float* NFPtr = NF.ptr<float>(0);


	contrast = cv::Mat::zeros(1, numberOfColors, CV_32FC1);

	float* contrastPtr = contrast.ptr<float>(0);

	for (int i = 0; i < numberOfColors; i++) {

		float* colorDistancePtr = colorDistance.ptr<float>(i);
		float* exponentialColorDistancePtr = exponentialColorDistance.ptr<float>(i);

		for (int k = 0; k < numberOfColors; k++) {

			contrastPtr[i] += colorDistancePtr[k] * histogramPtr[reverseMap[k]];

			XPtr[i] += exponentialColorDistancePtr[k] * averageXPtr[reverseMap[k]];
			YPtr[i] += exponentialColorDistancePtr[k] * averageYPtr[reverseMap[k]];
			X2Ptr[i] += exponentialColorDistancePtr[k] * averageX2Ptr[reverseMap[k]];
			Y2Ptr[i] += exponentialColorDistancePtr[k] * averageY2Ptr[reverseMap[k]];
			NFPtr[i] += exponentialColorDistancePtr[k] * histogramPtr[reverseMap[k]];

		}
	}

	divide(X, NF, X);
	divide(Y, NF, Y);
	divide(X2, NF, X2);
	divide(Y2, NF, Y2);

	// The mx, my, Vx, and Vy represent the same symbols in the paper. They are the spatial center and variances of the colors, respectively.

	X.assignTo(mx);
	Y.assignTo(my);

	Vx = X2 - mx.mul(mx);
	Vy = Y2 - my.mul(my);

}

void calculateProbability(cv::Mat mx, cv::Mat my, cv::Mat Vx, cv::Mat Vy, cv::Mat modelMean, cv::Mat modelInverseCovariance,
	int width, int height, cv::Mat &Xsize, cv::Mat &Ysize, cv::Mat &Xcenter, cv::Mat &Ycenter, cv::Mat &shapeProbability){

	// Convert the spatial center and variances to vector "g" in the paper, so we can compute the probability of saliency.

	sqrt(12 * Vx, Xsize);
	Xsize = Xsize / (float)width;

	sqrt(12 * Vy, Ysize);
	Ysize = Ysize / (float)height;

	Xcenter = (mx - width / 2) / (float)width;
	Ycenter = (my - height / 2) / (float)height;

	cv::Mat     g;

	vconcat(Xsize, Ysize, g);
	vconcat(g, Xcenter, g);
	vconcat(g, Ycenter, g);

	cv::Mat repeatedMeanVector;

	repeat(modelMean, 1, Xcenter.cols, repeatedMeanVector);

	g = g - repeatedMeanVector;

	g = g / 2;

	shapeProbability = cv::Mat::zeros(1, Xcenter.cols, CV_32FC1);

	float* shapeProbabilityPtr = shapeProbability.ptr<float>(0);

	// Comptuing the probability of saliency. As we will perform a normalization later, there is no need to multiply it with a constant term of the Gaussian function.

	for (int i = 0; i < Xcenter.cols; i++) {

		cv::Mat result, transposed;

		transpose(g.col(i), transposed);

		gemm(transposed, modelInverseCovariance, 1.0, 0.0, 0.0, result);

		gemm(result, g.col(i), 1.0, 0.0, 0.0, result);

		shapeProbabilityPtr[i] = exp(-result.at<float>(0, 0) / 2);

	}

}

void computeSaliencyMap(cv::Mat shapeProbability, cv::Mat contrast, cv::Mat exponentialColorDistance, cv::Mat histogramIndex, int* mapPtr,
	cv::Mat& SM, cv::Mat& saliency){

	double minVal, maxVal;

	int numberOfColors = shapeProbability.cols;

	saliency = shapeProbability.mul(contrast);

	float* saliencyPtr = saliency.ptr<float>(0);

	for (int i = 0; i < numberOfColors; i++) {

		float a1 = 0;
		float a2 = 0;

		for (int k = 0; k < numberOfColors; k++) {

			if (exponentialColorDistance.at<float>(i, k) > 0.0){

				a1 += saliencyPtr[k] * exponentialColorDistance.at<float>(i, k);
				a2 += exponentialColorDistance.at<float>(i, k);

			}

		}

		saliencyPtr[i] = a1 / a2;
	}

	minMaxLoc(saliency, &minVal, &maxVal);

	saliency = saliency - minVal;
	saliency = 255 * saliency / (maxVal - minVal) + 1e-3;

	minMaxLoc(saliency, &minVal, &maxVal);

	for (int y = 0; y < SM.rows; y++){

		uchar* SMPtr = SM.ptr<uchar>(y);

		int* histogramIndexPtr = histogramIndex.ptr<int>(y);

		for (int x = 0; x < SM.cols; x++){

			float sal = saliencyPtr[mapPtr[histogramIndexPtr[x]]];

			SMPtr[x] = (uchar)(sal);

		}
	}


}


cv::Rect saliency(cv::Mat&  im){
	//int totalImages = 0;
	/*cv::imshow("_im ", im);
	cv::waitKey(0); */
	cv::Rect _rect;
	float totalColor = 0;

	float totalPixels = 0;
	for (int i = 0; i < squares.cols; i++)
		squaresPtr[i] = pow(i, 2);

	if (im.data){

		cv::Mat lab;

		totalPixels += im.cols*im.rows;
		LAB.clear();
		L.clear();
		A.clear();
		B.clear();

		cv::Mat averageX, averageY, averageX2, averageY2, histogram, histogramIndex;

		vector<float> LL, AA, BB;

		calculateHistogram(im, averageX, averageY, averageX2, averageY2, LL, AA, BB, histogram, histogramIndex);
		float* averageXPtr = averageX.ptr<float>(0);
		float* averageYPtr = averageY.ptr<float>(0);
		float* averageX2Ptr = averageX2.ptr<float>(0);
		float* averageY2Ptr = averageY2.ptr<float>(0);

		int*    histogramPtr = histogram.ptr<int>(0);

		cv::Mat map, colorDistance, exponentialColorDistance;

		vector<int> reverseMap;

		int numberOfColors = precomputeParameters(histogram, LL, AA, BB, im.cols * im.rows, reverseMap, map, colorDistance, exponentialColorDistance); 
		totalColor += numberOfColors;

		int* mapPtr = map.ptr<int>(0);

		cv::Mat mx, my, Vx, Vy, contrast;

		bilateralFiltering(colorDistance, exponentialColorDistance, reverseMap, histogramPtr, averageXPtr, averageYPtr, averageX2Ptr, averageY2Ptr, mx, my, Vx, Vy, contrast); 

		cv::Mat Xsize, Ysize, Xcenter, Ycenter, shapeProbability;

		calculateProbability(mx, my, Vx, Vy, modelMean, modelInverseCovariance, im.cols, im.rows, Xsize, Ysize, Xcenter, Ycenter, shapeProbability);

		cv::Mat SM = cv::Mat::zeros(im.rows, im.cols, CV_8UC1);

		cv::Mat saliency;

		computeSaliencyMap(shapeProbability, contrast, exponentialColorDistance, histogramIndex, mapPtr, SM, saliency);
		//totalTime += double(et - st) / CLOCKS_PER_SEC;
		//	printf(" time to detect saliency map %d \r\n", totalTime); 
		//float* saliencyPtr = saliency.ptr<float>(0);
		float* saliencyPtr = saliency.ptr<float>(0);

		cv::Mat ellipseDetection, rectangleDetection;

		im.copyTo(ellipseDetection);
		im.copyTo(rectangleDetection);

		ofstream objectRectangles;

		//objectRectangles.open(savePath + "rectangleBoundingBoxes/" + imageNames[totalImages].substr(0, imageNames[totalImages].length() - 4) + "txt");

		for (int i = 0; i < numberOfColors; i++) {
			float rx = Xsize.at<float>(0, i)*im.cols;
			float ry = Ysize.at<float>(0, i)*im.rows;

			float xx, yy, ww, hh;

			xx = mx.at<float>(0, i) - rx / 2 >= 0 ? mx.at<float>(0, i) - rx / 2 : 0;
			yy = my.at<float>(0, i) - ry / 2 >= 0 ? my.at<float>(0, i) - ry / 2 : 0;
			ww = xx + rx < im.cols ? rx : im.cols - xx;
			hh = yy + ry < im.rows ? ry : im.rows - yy;

			objectRectangles << xx << "," << yy << "," << ww << "," << hh << "," << saliencyPtr[i] << "\n";

			if (saliencyPtr[i] > 254){
				ellipse(ellipseDetection, cv::Point(mx.at<float>(0, i), my.at<float>(0, i)), cv::Size(rx / 2, ry / 2), 0, 0, 360, cv::Scalar(0, 0, saliencyPtr[i]), 3, CV_AA);
				rectangle(rectangleDetection, cv::Point(xx, yy), cv::Point(xx + ww, yy + hh), cv::Scalar(0, 0, 255), 3, CV_AA);
				_rect = cv::Rect(cv::Point(xx, yy), cv::Point(xx + ww, yy + hh));
				//cout << "_ rect  " << _rect << endl;

			}

		}

		objectRectangles.close();
        cv::imshow(" rectangle detection", rectangleDetection);
        cv::waitKey(10);
		double minVal, maxVal;

		minMaxLoc(shapeProbability, &minVal, &maxVal);

		shapeProbability = shapeProbability - minVal;
		shapeProbability = shapeProbability / (maxVal - minVal + 1e-3);

		minMaxLoc(contrast, &minVal, &maxVal);

		contrast = contrast - minVal;
		contrast = contrast / (maxVal - minVal + 1e-3);

		float* shapeProbabilityPtr = shapeProbability.ptr<float>(0);
		float* contrastPtr = contrast.ptr<float>(0);

		cv::Mat saliencyProbabilityImage = cv::Mat::zeros(im.rows, im.cols, CV_32FC1);
		cv::Mat globalContrastImage = cv::Mat::zeros(im.rows, im.cols, CV_32FC1);

		for (int y = 0; y < im.rows; y++){

			float* saliencyProbabilityImagePtr = saliencyProbabilityImage.ptr<float>(y);
			float* globalContrastImagePtr = globalContrastImage.ptr<float>(y);

			int* histogramIndexPtr = histogramIndex.ptr<int>(y);

			for (int x = 0; x < im.cols; x++){

				saliencyProbabilityImagePtr[x] = shapeProbabilityPtr[mapPtr[histogramIndexPtr[x]]];
				globalContrastImagePtr[x] = contrastPtr[mapPtr[histogramIndexPtr[x]]];

			}
		}

		saliencyProbabilityImage = 255 * saliencyProbabilityImage;
		saliencyProbabilityImage.convertTo(saliencyProbabilityImage, CV_8UC1);
		/*cv::imshow("saliencyProbabilityImage ", saliencyProbabilityImage);
		cv::waitKey(0);*/
		globalContrastImage = 255 * globalContrastImage;
		globalContrastImage.convertTo(globalContrastImage, CV_8UC1);
		//cv::imshow("Before globalContrastImage", globalContrastImage);
		//cv::waitKey(0);
		cv::threshold(globalContrastImage, globalContrastImage, 80, 255, cv::ADAPTIVE_THRESH_MEAN_C);

		//cv::imshow("After globalContrastImage", globalContrastImage);
		//cv::waitKey(0);
		//cout << "_ rect 1 " << _rect << endl;
	}

	return _rect;
}

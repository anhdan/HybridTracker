#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <fstream>
#include <ctime>
#include <chrono>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>

#include <opencv2/opencv.hpp>


/***************************************************
 *              Macros
 **************************************************/
#define     __DEBUG__

#ifdef      __DEBUG__
#define     LOG_MSG( ... )      printf( __VA_ARGS__ )
#else
#define     LOG_MSG( ... )
#endif

/***************************************************
 *              Types
 **************************************************/
enum htError_enum
{
    htSuccess = 0,                      /**< Success */
    htErrorInvalidParameters = -1,      /**< Invalid input parameters */
    htErrorNotAllocated = -2,           /**< A pointer was not allocated */
    htErrorNotCompatible = -3,          /**< Two objects are not compatible to be processed */
    htErrorProcessFailure               /**< An unspecified failure in processing a function */
};

typedef int htError_t;

enum class GMEType : int
{
    HOMOGRAPHY = 0,                     /**< Homography transform estimator */
    RIGID_TRANSFORM                     /**< Rigid transform estimator */
};

enum class TrackStatus : int
{
    IN_VISION = 0,                      /**< Object is clear and being tracked */
    OCCLUDED,                           /**< Object is occluded by background */
    TRACK_LOST                          /**< Object is lost */
};

/***************************************************
 *              Parameters
 **************************************************/
//======== Global motion estimation parameters
#define     GME_MAX_FEATURES        500     /**< Maximum number of feature points to detect in an image */
#define     GME_MIN_FEATURES        20      /**< Minimum number of feature points to detect in an image */
#define     GME_FEATURE_QUALITY     0.005f  /**< Quality threshold for good feature to track */
#define     GME_FEATURE_MIN_DIS     20.0f   /**< Minimum distance between any two good features */
#define     GME_FEATURE_BLKSIZE     3       /**< Block size for detecting good feature */
#define     GME_RANSAC_INLIER_THRESH 2.0f   /**< RANSAC threshold for estimating transformation matrix */
#define     GME_MIN_INLIER          15      /**< Mimimum number of inliers for each transformation estimation */
#define     GME_MIN_INLIER_RATIO    0.1f    /**< Minimum ratio of inliers over the total key point pairs */
#define     GME_MIN_EIGEN_VALUE     1e-4f   /**< Minimum valid eigen value of a transformation matrix */

//======== KCF tracking parameters
#define     HOG_CELL_SIZE           4       /**< Cell size parameter for HOG feature extraction */
#define     KCF_TEMPLATE_SIZE       96      /**< KCF tracking template size */
#define     KCF_PADD_RATIO          2.0     /**< Padding ratio of image region surrounding object when extracting image patch */
#define     KCF_GAUSS_CORR_SIGMA    0.6     /**< Bandwidth of Gaussian kernel for computing correlation */
#define     KCF_GAUSS_RES_SIGMA     0.1     /**< Bandwidth of Gaussian distribution of desired target response */
#define     KCF_LAMDA               1e-4    /**< Additive to avoid division by zero */
#define     KCF_LEARNING_RATE       0.015   /**< Learning rate */
#define     KCF_MAX_OCCLUDED_FRAMES 30      /**< Number of frames in which object is occluded before it is considered to be lost*/
#define     KCF_HISTOGRAM_BINS      16      /**< Number of histogram bins in each marginal color histogram vector of a color image */
#define     KCF_HIST_OCC_THRESH     30      /**< Histogram disimilarity threshold to detect object occlusion using color information */
#define     KCF_PSR_OCC_THRESH      7       /**< Peak-To-Sidelope threshold to detect object occlusion based on correlation response map */

/***************************************************
 *              Macros
 **************************************************/


/***************************************************
 *              Functions
 **************************************************/

namespace kcf {

    /**
     * @brief Compute FFT/IFFT of a matrix
     * @param _src      Source matrix
     * @param _dst      Destination matrix
     * @param _isInvert Flag for chosing invert FFT
     * @return          Completion status of the function
     */
    htError_t fftd( const cv::Mat &_src, cv::Mat &_dst, bool _isInvert );

    /**
     * @brief Obtains real component of a complex matrix
     * @param _cmpl     Complex matrix
     * @return          Its real component
     */
    cv::Mat real( const cv::Mat &_cmpl );

    /**
     * @brief Obtains imaginary component of a complex matrix
     * @param _cmpl     Complex matrix
     * @return          Its imaginary component
     */
    cv::Mat imag( const cv::Mat &_cmpl );

    /**
     * @brief complexDivision
     * @param a
     * @param b
     * @return
     */
    cv::Mat complexDivision( const cv::Mat &a, const cv::Mat &b);

    /**
     * @brief createHanningWindow
     * @param _hann
     * @param _dimX
     * @param _dimY
     * @param _dimZ
     * @return
     */
    htError_t createHanningWindow( cv::Mat &_hann, const int _dimX, const int _dimY, const int _dimZ );

    /**
     * @brief createGaussianWindow
     * @param _gauss
     * @param _dimX
     * @param _dimY
     * @return
     */
    htError_t createGaussianWindow( cv::Mat &_gauss, const int _dimX, const int _dimY, const float _sigma );


    /**
     * @brief createTukeyWindow
     * @param _tukey
     * @param _dimX
     * @param _dimY
     * @param _alphaX
     * @param _alphaY
     * @return
     */
    htError_t createTukeyWindow( cv::Mat &_tukey, const int _dimX, const int _dimY, const float _alphaX, const float _alphaY );


    /**
     * @brief gaussianCorrelation
     * @param _x
     * @param _z
     * @param _corr
     * @param _mapSize
     * @param _sigma
     * @return
     */
    htError_t gaussianCorrelation( const cv::Mat &_x, const cv::Mat &_z,  cv::Mat &_corr,
                                   const cv::Point3i _mapSize, const float &_sigma );

    /**
     * @brief detectPeakNPSR
     * @param _response
     * @param _peakLoc_i
     * @param _peakVal
     * @param _psr
     * @return
     */
    htError_t detectPeakNPSR( const cv::Mat &_response, cv::Point &_peakLoc_i, float *_peakVal, float *_psr );

    /**
     * @brief fixPeak
     * @param left
     * @param center
     * @param right
     * @return
     */
    float fixPeak( const float left , const float center, const float right );

    /**
     * @brief decomposeTransform
     * @param _trans
     * @param _dx
     * @param _dy
     * @param _rot
     * @param _scale
     * @return
     */
    htError_t decomposeTransform( const cv::Mat &_trans, float *_dx, float *_dy, float *_rot, float *_scale );


    /**
     * @brief computeHistogram
     * @param _img
     * @param _mask
     * @param _binsPerChannel
     * @param _hist
     * @return
     */
    htError_t computeHistogram( const cv::Mat &_img, const cv::Mat &_mask, const int _binsPerChannel, float *_hist );


    /**
     * @brief computeBhattacharyaCoeff
     * @param _vec1
     * @param _vec2
     * @param _len
     * @return
     */
    float computeBhattacharyaCoeff( const float *_vec1, const float *_vec2, const int _len );
}

namespace graphix {
    /**
     * @brief Draws a rotated rectangle on an image
     * @param _img      Image to draw on
     * @param _rotRect  Rotated rectangle
     */
    void rotRectangle( cv::Mat &_img, const cv::RotatedRect _rotRect );

    /**
     * @brief writeMatToText
     * @param _filename
     * @param hMat
     */
    void writeMatToText( const char *_filename , const cv::Mat &hMat );

    /**
     * @brief imshowFloat
     * @param _window
     * @param _img
     */
    void imshowFloat( const std::string _window, const cv::Mat &_img );
}


#endif // UTILS_H

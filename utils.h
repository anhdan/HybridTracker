#ifndef UTILS_H
#define UTILS_H


#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <math.h>
#include <float.h>
#include <fstream>
#include <chrono>

#include <opencv2/opencv.hpp>


/*****************************************************
 *
 *                  Macros
 *
 ****************************************************/
#define DEBUG

#ifdef DEBUG
#define     LOG_MSG( ... )    printf( __VA_ARGS__ )
#else
#define     LOG_MSG( ... )
#endif

/*****************************************************
 *
 *              Type Definitions
 *
 ****************************************************/
/**
 * @brief The htError_t enum
 */
enum htError_t : int
{
    htSuccess               = 0,
    htErrorProcessFailure   = -1,
    htErrorNotAllocated     = -2
};


/**
 * @brief The TrackStatus enum
 */
enum class TrackStatus : int
{
    IN_VISION   = 0,
    OCCLUDED,
    TRACK_LOST
};


/*****************************************************
 *
 *          Function for KCF tracker
 *
 ****************************************************/
/**
 * @brief calcPSR
 * @param _response
 * @return
 */
extern float calcPSR( const cv::Mat &_response );

/**
 * @brief fixPeak
 * @param _left
 * @param _center
 * @param right
 * @return
 */
extern float fixPeak( const float _left, const float _center, const float right );


#endif // UTILS_H

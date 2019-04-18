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
#define     LOG_MSG( ... )      prinf( __VA_ARGS__ )
#else
#define     LOG_MSG( ... )
#endif

/***************************************************
 *              Types
 **************************************************/
enum htError_t : int
{
    htSuccess = 0,
    htErrorInvalidParameters = -1,
    htErrorNotAllocated = -2,
    htErrorProcessFailure = -3
};

enum class TrackStatus : int
{
    IN_VISION = 0,
    OCCLUDED,
    TRACK_LOST
};

/***************************************************
 *              Parameters
 **************************************************/
#define     KCF_PADD_RATIO      2.0

/***************************************************
 *              Macros
 **************************************************/

#endif // UTILS_H

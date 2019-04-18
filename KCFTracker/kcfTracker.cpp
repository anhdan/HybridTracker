#include "kcfTracker.h"

/**
 * @brief KCFTracker::KCFTracker
 */
KCFTracker::KCFTracker()
{
    LOG_MSG( "[DEB] A new KCF tracker has been created\n" );
    m_occFramesCnt  = 0;
    m_trackStatus   = TrackStatus::TRACK_LOST;
    m_tmplSize      = cv::Size(-1, -1);
    m_feaMapSize    = cv::Point3i(-1, -1, -1);
    m_scale         = 1.0;
}


/**
 * @brief KCFTracker::~KCFTracker
 */
KCFTracker::~KCFTracker()
{
    LOG_MSG( "[DEB] A new KCF tracker has been destroyed\n" );
}


/**
 * @brief KCFTracker::getPatch
 */
htError_t KCFTracker::getPatch( const cv::Mat &_rgbImg, cv::Mat &_grayPatch, cv::Mat &_rgbPatch,
                                const cv::RotatedRect _targetBound, const float _targetScale )
{
    //===== 1. Extract data int the rectangular bounding box of the rotated rectangle
    cv::RotatedRect expandedRotRect = _targetBound;
    expandedRotRect.size.width  *= _targetScale * KCF_PADD_RATIO;
    expandedRotRect.size.height *= _targetScale * KCF_PADD_RATIO;

    cv::Rect rect = expandedRotRect.boundingRect();

    // Rectange that is inside the image boundary
    int top     = rect.y,
        left    = rect.x,
        bot     = rect.y + rect.height,
        right   = rect.x + rect.width;
    if( top < 0 ) top = 0;
    if( left < 0 ) left = 0;
    if( bot >= _rgbImg.rows ) bot = _rgbImg.rows - 1;
    if( right >= _rgbImg.cols ) right = _rgbImg.cols - 1;

    if( (top >= bot) || (left >= right) )
    {
        LOG_MSG("[ERR] %s:%d: stt = %d: Invalid target ROI\n",
                __FUNCTION__, __LINE__, htErrorInvalidParameters );
        return htErrorInvalidParameters;
    }

    cv::Rect validRect(left, top, right - left, bot - top);

    int deltaTop   = top - rect.y,
        deltaLeft  = left - rect.x,
        deltaBot   = rect.y + rect.height - bot,
        deltaRight = rect.x + rect.width - right;

    // Extract valid image patch
    cv::Mat rectPatch = cv::Mat::zeros( rect.height, rect.width, CV_8UC3 );
    cv::copyMakeBorder( _rgbImg(validRect), rectPatch, deltaTop, deltaBot, deltaLeft, deltaRight,
                        cv::BORDER_CONSTANT, cv::Scalar::all(0.0) );

    //===== 2. Extract rotated patch from its rectangular bounding box patch
    // Compute rotation matrix
    float angle = 90.0 - expandedRotRect.angle;
    cv::Point2f center = cv::Point2f((float)rectPatch.cols / 2.0, (float)rectPatch.rows / 2.0);
    cv::Mat R = cv::getRotationMatrix2D( center, angle, 1.0 );

    // Perform warp affine the bounding box patch so that the extracted patch is vertical
    cv::Mat rotated;
    cv::warpAffine( rectPatch, rotated, R, rectPatch.size(), cv::INTER_CUBIC );

    // Crop the resulting image to obtain RGB rotated image patch
    cv::getRectSubPix( rotated, cv::Size((int)expandedRotRect.size.width, (int)expandedRotRect.size.height),
                       center, _rgbPatch );

    // Convert RGB to gray format to obtain gray image patch
    cv::cvtColor( _rgbPatch, _grayPatch, cv::COLOR_RGB2GRAY );

    return htSuccess;
}

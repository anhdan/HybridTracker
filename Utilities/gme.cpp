#include "gme.h"

/**
 * @brief GME::ageDelay
 */
void GME::ageDelay()
{
    m_currGray.copyTo( m_prevGray );
}


/**
 * @brief GME::process
 */
htError_t GME::process( const cv::Mat &_img )
{
    if( _img.empty() )
    {
        LOG_MSG( "[ERR] %s:%d: status = %d: Empty input image\n",
                 __FUNCTION__, __LINE__, htErrorNotAllocated );
        return htErrorNotAllocated;
    }

    ageDelay();
    if( _img.channels() > 1 )
    {
        cv::cvtColor( _img, m_currGray, cv::COLOR_RGB2GRAY );
    }
    else
    {
        _img.copyTo( m_currGray );
    }

    if( m_prevGray.empty() )
    {
        m_trans = cv::Mat::eye(3, 3, CV_32FC1);
        return htSuccess;
    }

    //===== 1. Key points detection
    std::vector<cv::Point2f> prevPts, tmp;
    cv::goodFeaturesToTrack( m_prevGray, prevPts, GME_MAX_FEATURES, GME_FEATURE_QUALITY, GME_FEATURE_MIN_DIS );
    tmp = prevPts;
    try
    {
        cv::cornerSubPix( m_prevGray, tmp, cv::Size(5, 5), cv::Size(-1, -1),
                          cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.03) );
    }
    catch( std::exception &e )
    {
        // Do nothing
    }

    if( tmp.size() > 0 )
    {
        prevPts = tmp;
    }

    //===== 2. Optical flow
    if( prevPts.size() <= GME_MIN_FEATURES )
    {
        m_trans = cv::Mat::eye( 3, 3, CV_32FC1 );
        return htSuccess;
    }

    std::vector<cv::Point2f> currPts, prevPtsLK, currPtsLK;
    std::vector<uchar> status;
    std::vector<float> error;
    try
    {
        cv::calcOpticalFlowPyrLK( m_prevGray, m_currGray,
                                  prevPts, currPts, status, error,
                                  cv::Size(21, 21), 5,
                                  cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.03),
                                  0, 0.001 );
    }
    catch( std::exception &e )
    {
        LOG_MSG( "[ERR] %s:%d: Exception when computing optical flow: %s\n",
                 __FUNCTION__, __LINE__, e.what() );
        return htErrorProcessFailure;
    }

    for( int idx = 0; idx < prevPts.size(); idx++ )
    {
        if( status[idx] )
        {
            prevPtsLK.push_back( prevPts[idx] );
            currPtsLK.push_back( currPts[idx] );
        }
    }

    if( currPtsLK.size() < 10 )
    {
        LOG_MSG( "[WAR] %s:%d: Too few matching key points\n", __FUNCTION__, __LINE__ );
        m_trans = cv::Mat::eye( 3, 3, CV_32FC1 );
        return htSuccess;
    }

    //===== 3. Estimate transform
    cv::Mat tmpTrans;
    if( m_type == GMEType::HOMOGRAPHY )
    {
        tmpTrans = cv::findHomography( prevPtsLK, currPtsLK, cv::RANSAC );
    }
    else if( m_type == GMEType::RIGID_TRANSFORM )
    {
        cv::videostab::RansacParams ransacParams = cv::videostab::RansacParams::default2dMotion( cv::videostab::MM_SIMILARITY );
        ransacParams.thresh = 2.0f;
        ransacParams.eps = 0.5f;
        ransacParams.prob = 0.99f;
        ransacParams.size = 4;

        int inliers = 0;
        tmpTrans = cv::videostab::estimateGlobalMotionRansac( prevPtsLK, currPtsLK,
                                                              cv::videostab::MM_SIMILARITY, ransacParams,
                                                              nullptr, &inliers );
        if( (float)inliers / (float)prevPtsLK.size() < 0.2 )
        {
            tmpTrans = cv::Mat::eye(3, 3, CV_32FC1);
        }
    }
    else
    {
        LOG_MSG( "[ERR] %s:%d: status = %d: Invalid type of transformation matrix to estimate\n",
                 __FUNCTION__, __LINE__, htErrorInvalidParameters );
        return htErrorInvalidParameters;
    }

    if( tmpTrans.type() != CV_32FC1 )
    {
        tmpTrans.convertTo( m_trans, CV_32FC1 );
    }
    else
    {
        tmpTrans.copyTo( m_trans );
    }

    return htSuccess;
}


/**
 * @brief GME::getTrans
 */
cv::Mat GME::getTrans()
{
    return m_trans;
}

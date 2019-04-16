#ifndef KCFTRACKER_H
#define KCFTRACKER_H

#include "utils.h"


class KCFTracker
{
public:
    KCFTracker();
    ~KCFTracker();

private:
    htError_t getPatch( const cv::Mat &_rgbImg, cv::Mat &_grayPatch, cv::Mat &_rgbPatch, cv::Size _targetSize );
    htError_t extractHOG( const cv::Mat &_grayImg, cv::Mat &_feaMap );
    htError_t shiftDetection( const cv::Mat &_tmplMap, const cv::Mat &_targetMap,
                              float *_peakVal, float *_psrVal, cv::Point2f &_shift );
    htError_t train( const cv::Mat &_targetMap, const float _learningRate );

public:
    htError_t initTrack( const cv::Mat &_rgb, const cv::Point &_initCenter, const cv::Size &_initSize );
    htError_t performTrack( const cv::Mat &_rgb, const cv::Point &_prevCenter );
    void getLocation( cv::Point &_center, cv::Point &_size );
    TrackStatus getTrackStatus( );

private:
    // KCF Model data
    cv::Mat     m_Alphaf;
    cv::Mat     m_K;
    cv::Mat     m_Y;
    cv::Mat     m_Hann;

    cv::Mat     m_featureMap;
    cv::Mat     m_tmplMap;
    cv::Mat     m_targetMap;

    // Scale & Rotation model data
    float       m_scale;
    float       m_rotation;

    // Occlusion model data
    cv::Mat     m_Tukey;
    cv::Mat     m_tmplRGB;
    cv::Mat     m_targetRGB;

    // Tracking object parameters
    cv::Size    m_tmplSize;
    cv::Size    m_feaMapSize;
    cv::Size    m_objSize;
};

#endif // KCFTRACKER_H

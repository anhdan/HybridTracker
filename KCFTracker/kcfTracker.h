#ifndef KCFTRACKER_H
#define KCFTRACKER_H

#include "Utilities/utils.h"


class KCFTracker
{
public:
    KCFTracker();
    ~KCFTracker();

private:
    /**
     * @brief getPatch
     * @param _rgbImg
     * @param _grayPatch
     * @param _rgbPatch
     * @param _targetBound
     * @param _targetScale
     * @return
     */
    htError_t getPatch( const cv::Mat &_rgbImg, cv::Mat &_grayPatch, cv::Mat &_rgbPatch,
                        const cv::RotatedRect _targetBound, const float _targetScale );

    /**
     * @brief extractHOG
     * @param _grayImg
     * @param _feaMap
     * @return
     */
    htError_t extractHOG( const cv::Mat &_grayImg, cv::Mat &_feaMap );

    /**
     * @brief shiftDetection
     * @param _tmplMap
     * @param _targetMap
     * @param _peakVal
     * @param _psrVal
     * @param _shift
     * @return
     */
    htError_t shiftDetection( const cv::Mat &_tmplMap, const cv::Mat &_targetMap,
                              float *_peakVal, float *_psrVal, cv::Point2f &_shift );

    /**
     * @brief train
     * @param _targetMap
     * @param _learningRate
     * @return
     */
    htError_t train( const cv::Mat &_targetMap, const float _learningRate );

public:
    /**
     * @brief initTrack
     * @param _rgb
     * @param _initCenter
     * @param _initSize
     * @return
     */
    htError_t initTrack( const cv::Mat &_rgb, const cv::Point &_initCenter, const cv::Size &_initSize );

    /**
     * @brief performTrack
     * @param _rgb
     * @param _prevCenter
     * @return
     */
    htError_t performTrack( const cv::Mat &_rgb, const cv::Point &_prevCenter );

    /**
     * @brief getLocation
     * @param _center
     * @param _size
     */
    void getLocation( cv::Point &_center, cv::Point &_size );

    /**
     * @brief getTrackStatus
     * @return
     */
    TrackStatus getTrackStatus( );

private:
    // KCF Model data
    cv::Mat         m_Alphaf;
    cv::Mat         m_K;
    cv::Mat         m_Y;
    cv::Mat         m_Hann;

    cv::Mat         m_featureMap;
    cv::Mat         m_tmplMap;
    cv::Mat         m_targetMap;

    // Scale & Rotation model data
    float           m_scale;
    cv::RotatedRect m_objBound;

    // Occlusion model data
    cv::Mat         m_Tukey;
    cv::Mat         m_tmplRGB;
    cv::Mat         m_targetRGB;

    // Tracking object parameters
    cv::Size        m_tmplSize;
    cv::Point3i     m_feaMapSize;
    TrackStatus     m_trackStatus;
    int             m_occFramesCnt;
};

#endif // KCFTRACKER_H

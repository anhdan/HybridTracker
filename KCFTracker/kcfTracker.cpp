#include "kcfTracker.h"

/**
 * @brief KCFTracker::KCFTracker
 */
KCFTracker::KCFTracker() : m_gme(GMEType::RIGID_TRANSFORM)
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
    free( m_tmplHist );
    free( m_targetHist );

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
    cv::Point2f center = cv::Point2f((float)rectPatch.cols / 2.0, (float)rectPatch.rows / 2.0);
    cv::Mat R = cv::getRotationMatrix2D( center, expandedRotRect.angle, 1.0 );

    // Perform warp affine the bounding box patch so that the extracted patch is vertical
    cv::Mat rotated;
    cv::warpAffine( rectPatch, rotated, R, rectPatch.size(), cv::INTER_CUBIC );

    // Crop the resulting image to obtain RGB rotated image patch
    cv::Mat tmpRGB;
    cv::getRectSubPix( rotated, cv::Size((int)expandedRotRect.size.width, (int)expandedRotRect.size.height),
                       center, tmpRGB );

    if( (tmpRGB.cols != m_tmplSize.width) || (tmpRGB.rows != m_tmplSize.height) )
    {
        cv::resize( tmpRGB, _rgbPatch, m_tmplSize );
    }
    else
    {
        tmpRGB.copyTo( _rgbPatch );
    }

    // Convert RGB to gray format to obtain gray image patch
    cv::cvtColor( _rgbPatch, _grayPatch, cv::COLOR_RGB2GRAY );

    return htSuccess;
}


/**
 * @brief KCFTracker::extractHOG
 */
htError_t KCFTracker::extractHOG( const cv::Mat &_grayImg, cv::Mat &_feaMap )
{
    IplImage ipl = _grayImg;
    CvLSVMFeatureMapCaskade *map;
    getFeatureMaps( &ipl, HOG_CELL_SIZE, &map );
    normalizeAndTruncate( map, 0.2 );
    PCAFeatureMaps( map );

    _feaMap.release();
    _feaMap = cv::Mat( cv::Size(map->numFeatures, map->sizeX*map->sizeY), CV_32F, map->map );
    _feaMap = _feaMap.t();

    // Multiply with Hanning window to reduce edge effect
    if( m_Hann.empty() )
    {
        LOG_MSG( "[ERR] %s:%d: status = %d: Hanning window has not been initialized\n",
                 __FUNCTION__, __LINE__, htErrorNotAllocated );
        return htErrorNotAllocated;
    }
    _feaMap = m_Hann.mul( _feaMap );

    return htSuccess;
}


/**
 * @brief KCFTracker::shiftDetection
 */
htError_t KCFTracker::shiftDetection( const cv::Mat &_tmplMap, const cv::Mat &_targetMap, float *_peakVal, float *_psrVal, cv::Point2f &_shift )
{
    htError_t err = htSuccess;

    // Compute kernelized correlation
    cv::Mat k;
    err |= kcf::gaussianCorrelation( _tmplMap, _targetMap, k, m_feaMapSize, KCF_GAUSS_CORR_SIGMA );

    if( err != htSuccess )
    {
        LOG_MSG( "[ERR] %s:%d: status = %d: Fail to compute kernelized correlation between template and target map\n",
                 __FUNCTION__, __LINE__, err );
        return err;
    }

    // Apply correlation filter
    cv::Mat K, Y;
    err |= kcf::fftd( k, K, false );
    cv::mulSpectrums( m_Alphaf, K, Y, 0, false );

    err |= kcf::fftd( Y, Y, true );
    cv::Mat y = kcf::real( Y );

    if( err != htSuccess )
    {
        LOG_MSG( "[ERR] %s:%d: status = %d: Failed to apply correlation filter\n",
                 __FUNCTION__, __LINE__, err );
        return err;
    }

    // Find response peak and PSR value
    cv::Point2i peakLoc_i;
    err |= kcf::detectPeakNPSR( y, peakLoc_i, _peakVal, _psrVal );
    LOG_MSG( "\n[DEB] %s:%d: peakLoc_i = (%d, %d)\n", __FUNCTION__, __LINE__, peakLoc_i.x, peakLoc_i.y );
    if( err != htSuccess )
    {
        LOG_MSG( "[ERR] %s:%d: status = %d: Failed to detect peak and PSR value\n",
                 __FUNCTION__, __LINE__, err );
        return err;
    }

    cv::Point2f peakLoc_f( (float)peakLoc_i.x, (float)peakLoc_i.y );
    if( (peakLoc_f.x > 0) && (peakLoc_f.x < (y.cols - 1)) )
    {
        peakLoc_f.x += kcf::fixPeak( y.at<float>(peakLoc_i.y, peakLoc_i.x-1),
                                     *_peakVal,
                                     y.at<float>(peakLoc_i.y, peakLoc_i.x+1) );
    }

    if( (peakLoc_f.y > 0) && (peakLoc_f.y < (y.rows - 1)) )
    {
        peakLoc_f.y += kcf::fixPeak( y.at<float>(peakLoc_i.y-1, peakLoc_i.x),
                                     *_peakVal,
                                     y.at<float>(peakLoc_i.y+1, peakLoc_i.x) );
    }

    _shift.x = -peakLoc_f.x + (float)y.cols / 2.0;
    _shift.y = -peakLoc_f.y + (float)y.rows / 2.0;

    return htSuccess;
}


/**
 * @brief KCFTracker::train
 */
htError_t KCFTracker::train( const cv::Mat &_targetMap, const cv::Mat &_rgbPatch, const float _learningRate )
{
    htError_t err = htSuccess;
    // Compute A & B in MOOSE filter update formular
    cv::Mat k, K;
    err |= kcf::gaussianCorrelation( _targetMap, _targetMap, k, m_feaMapSize, KCF_GAUSS_CORR_SIGMA );
    err |= kcf::fftd( k, K, false );
    if( err != htSuccess )
    {
        LOG_MSG( "[ERR] %s:%d: status = %d: Fail to compute kernelized correlation\n",
                 __FUNCTION__, __LINE__, err );
        return err;
    }

    m_K = _learningRate * K + (1 - _learningRate) * m_K;

    // Update filter
    m_Alphaf = kcf::complexDivision( m_Y, m_K + KCF_LAMDA );

    // Update feature template
    m_tmplMap = _learningRate * _targetMap + (1 - _learningRate) * m_tmplMap;

    // Update color patch template
//    m_tmplRGB = _learningRate * _rgbPatch + (1 - _learningRate) * m_tmplRGB;
    cv::addWeighted( _rgbPatch, _learningRate, m_tmplRGB, 1 - _learningRate, 0.0, m_tmplRGB );

    return htSuccess;
}


/**
 * @brief KCFTracker::init
 */
htError_t KCFTracker::init( const cv::Mat &_rgb, const cv::Point &_initCenter, const cv::Size &_initSize )
{
    if( _rgb.empty() || (_rgb.channels() != 3) )
    {
        LOG_MSG( "[ERR] %s:%d: status = %d: Empty input image\n",
                 __FUNCTION__, __LINE__, htErrorNotAllocated );
        return htErrorNotAllocated;
    }

    m_config.width  = _rgb.cols;
    m_config.height = _rgb.rows;

    //===== 1. Compute template and feature map sizes
    int paddedW = _initSize.width * KCF_PADD_RATIO,
        paddedH = _initSize.height * KCF_PADD_RATIO;
    m_scale = (paddedW > paddedH) ? ((float)paddedW / KCF_TEMPLATE_SIZE) : ((float)paddedH / KCF_TEMPLATE_SIZE);
    m_objBound.center = _initCenter;
    m_objBound.size   = cv::Size2f(_initSize.width, _initSize.height);
    m_objBound.angle  = 0.0;

    m_tmplSize.width  = (int)(paddedW / m_scale);
    m_tmplSize.height = (int)(paddedH / m_scale);
    m_tmplSize.width  = (m_tmplSize.width / (2 * HOG_CELL_SIZE) + 1) * (2 * HOG_CELL_SIZE);
    m_tmplSize.height = (m_tmplSize.height / (2 * HOG_CELL_SIZE) + 1) * (2 * HOG_CELL_SIZE);

    m_feaMapSize.x = m_tmplSize.width / HOG_CELL_SIZE - 2;
    m_feaMapSize.y = m_tmplSize.height / HOG_CELL_SIZE - 2;
    m_feaMapSize.z = 31;

    if( (m_feaMapSize.x <= 0) || (m_feaMapSize.y <= 0) )
    {
        LOG_MSG( "[ERR] %s:%d: status = %d: Too small initial tracking size\n",
                 __FUNCTION__, __LINE__, htErrorInvalidParameters );
        return htErrorInvalidParameters;
    }

    //===== 2. Initialize tracking model
    htError_t err = htSuccess;
    // Create Hanning window
    err |= kcf::createHanningWindow( m_Hann, m_feaMapSize.x, m_feaMapSize.y, m_feaMapSize.z );    

    // Create Gaussian response
    float sigma = sqrt( (float)(m_feaMapSize.x * m_feaMapSize.y) ) / KCF_PADD_RATIO * KCF_GAUSS_RES_SIGMA;
    cv::Mat y;
    err |= kcf::createGaussianWindow( y, m_feaMapSize.x, m_feaMapSize.y, sigma );
    err |= kcf::fftd( y, m_Y, false );

    // Create Tukey window for histogram computation
    err |= kcf::createTukeyWindow( m_Tukey, m_tmplSize.width, m_tmplSize.height, 0.5, 0.5 );

    if( err != htSuccess )
    {
        LOG_MSG( "[ERR] %s:%d: status = %d: Failed to create windows\n",
                 __FUNCTION__, __LINE__, err );
        return err;
    }

    // Allocate memory for histogram vector
    m_histLen    = KCF_HISTOGRAM_BINS * 3;
    m_tmplHist   = (float*)malloc( m_histLen * sizeof( float ) );
    m_targetHist = (float*)malloc( m_histLen * sizeof( float ) );
    if( m_tmplHist == NULL || m_targetHist == NULL )
    {
        free( m_tmplHist );
        free( m_targetHist );
        LOG_MSG( "[ERR] %s:%d: status = %d: Failed to allocate memory\n",
                 __FUNCTION__, __LINE__, htErrorNotAllocated );
        htErrorNotAllocated;
    }

    // Initialize template feature and correlation filter
    cv::Mat patchGray, feaMap;
    err |= getPatch( _rgb, patchGray, m_tmplRGB, m_objBound, 1.0 );

    err |= extractHOG( patchGray, feaMap );

    err |= train( feaMap, m_tmplRGB, 1.0 );

    if( err != htSuccess )
    {
        free( m_tmplHist );
        free( m_targetHist );
        LOG_MSG( "[ERR] %s:%d: status = %d: Failed to initialize template feature and correlation filter\n",
                 __FUNCTION__, __LINE__, err );
        return err;
    }

    //===== 3. Initialize GME and Kalman filter for position prediction
    m_gme.process( _rgb );
    m_kalman.setDefaultModel();
    m_kalman.setState( 0.0, 0.0, 0.0, 0.0 );

    m_trackStatus  = TrackStatus::IN_VISION;
    m_occFramesCnt = 0;

    return htSuccess;
}


/**
 * @brief KCFTracker::process
 */
htError_t KCFTracker::process( const cv::Mat &_rgb )
{
    if( (_rgb.cols != m_config.width) || (_rgb.rows != m_config.height) || (_rgb.channels() != 3) )
    {
        LOG_MSG( "[ERR] %s:%d: status = %d: Invalid input image\n",
                 __FUNCTION__, __LINE__, htErrorNotAllocated );
        return htErrorNotAllocated;
    }

    if( m_trackStatus == TrackStatus::TRACK_LOST )
    {
        LOG_MSG( "[DEB] %s:%d: =============> Track Lost\n", __FUNCTION__, __LINE__ );
        return htSuccess;
    }

    htError_t err = htSuccess;

    //===== 1. Predict new position using GME and Kalman filter
    err |= m_gme.process( _rgb );
    cv::Mat trans = m_gme.getTrans();
    float gmeX = (trans.at<float>(0, 0) * m_objBound.center.x + trans.at<float>(0, 1) * m_objBound.center.y + trans.at<float>(0, 2)),
          gmeY = (trans.at<float>(1, 0) * m_objBound.center.x + trans.at<float>(1, 1) * m_objBound.center.y + trans.at<float>(1, 2));

    float gmeRot, gmeScale;
    err |= kcf::decomposeTransform( trans, nullptr, nullptr, &gmeRot, &gmeScale );
    m_scale *= gmeScale;
    m_objBound.size.width *= gmeScale;
    m_objBound.size.height *= gmeScale;
    m_objBound.angle += gmeRot;

    float dxKalman, dyKalman;
    m_kalman.predictMotion( &dxKalman, &dyKalman );
    m_objBound.center.x = gmeX;// + dxKalman,
    m_objBound.center.y = gmeY;// + dyKalman;

    if( err != htSuccess )
    {
        LOG_MSG( "[ERR] %s:%d: status = %d: Failed to predict new position of object\n",
                 __FUNCTION__, __LINE__, err );
        return err;
    }

    //===== 2. Apply KCF tracking algorithm
    //! FIXIT: Add Log-polar scale estimation
    float relativeScale = 1.0;
    cv::Mat rgbPatch, grayPatch;
    err |= getPatch( _rgb, grayPatch, rgbPatch, m_objBound, relativeScale );

    cv::Mat targetMap;
    err |= extractHOG( grayPatch, targetMap );

    float peak, psr;
    cv::Point2f shift;
    err |= shiftDetection( m_tmplMap, targetMap, &peak, &psr, shift );

    LOG_MSG( "\n[DEB] %s:%d: peak = %f  -  psr = %f  -  shift = (%f, %f)\n",
             __FUNCTION__, __LINE__, peak, psr, shift.x, shift.y );

    if( err != htSuccess )
    {
        LOG_MSG( "[ERR] %s:%d: status = %d: Failed to detect object shift\n",
                 __FUNCTION__, __LINE__, err );
        return err;
    }

    float dxKCF = shift.x * m_scale * HOG_CELL_SIZE,
          dyKCF = shift.y * m_scale * HOG_CELL_SIZE;

    float kcfX  = cosf( m_objBound.angle ) * dxKCF + sinf( m_objBound.angle ) * dyKCF + m_objBound.center.x,
          kcfY  = -sinf( m_objBound.angle ) * dxKCF + cosf( m_objBound.angle ) * dyKCF + m_objBound.center.y;

    float kcfW  = m_objBound.size.width * relativeScale,
          kcfH  = m_objBound.size.height * relativeScale;

    //===== 3. Detect occlusion
    std::cout << m_tmplRGB.type() << "\t" << CV_8UC3 << std::endl;
    err |= kcf::computeHistogram( m_tmplRGB, m_Tukey, KCF_HISTOGRAM_BINS, m_tmplHist );
    err |= kcf::computeHistogram( rgbPatch, m_Tukey, KCF_HISTOGRAM_BINS, m_targetHist );
    if( err != htSuccess )
    {
        LOG_MSG( "[ERR] %s:%d: status = %d: Failed to compute histogram\n",
                 __FUNCTION__, __LINE__, err );
        return err;
    }

    float bhattCoeff = kcf::computeBhattacharyaCoeff( m_tmplHist, m_targetHist, m_histLen );
    LOG_MSG( "\n[DEB]: %s:%d: Bhattacharrya coeff = %f\n", __FUNCTION__, __LINE__, bhattCoeff );

    if( bhattCoeff > KCF_HIST_OCC_THRESH )
    {
        m_trackStatus = TrackStatus::OCCLUDED;
    }
    else if( psr < KCF_PSR_OCC_THRESH )
    {
        m_trackStatus = TrackStatus::OCCLUDED;
    }

    //===== 4. Update model
    if( m_trackStatus == TrackStatus::IN_VISION )
    {
        m_objBound.center = cv::Point2f(kcfX, kcfY);
        m_objBound.size   = cv::Size2f(kcfW, kcfH);
        m_occFramesCnt    = (m_occFramesCnt >= 2) ? (m_occFramesCnt - 2) : 0;

        // Update KCF model
        err |= getPatch( _rgb, grayPatch, rgbPatch, m_objBound, relativeScale );

        err |= extractHOG( grayPatch, targetMap );

        err |= train( targetMap, rgbPatch, KCF_LEARNING_RATE );

        if( err != htSuccess )
        {
            LOG_MSG( "[ERR] %s:%d: status = %d: Failed to update tracking model\n",
                     __FUNCTION__, __LINE__, err );
            return err;
        }

        // UPdate Kalman filter
        m_kalman.correctModel( kcfX - gmeX, kcfY - gmeY );
    }
    else
    {
        m_occFramesCnt++;
        if( m_occFramesCnt > KCF_MAX_OCCLUDED_FRAMES )
        {
            m_trackStatus = TrackStatus::TRACK_LOST;
        }
        m_kalman.correctModel( dxKalman, dyKalman );
    }

    if( (m_objBound.center.x <= 0) || (m_objBound.center.x >= m_config.width)
     || (m_objBound.center.y <= 0) || (m_objBound.center.y >= m_config.height) )
    {
        m_trackStatus = TrackStatus::TRACK_LOST;
    }

    return err;
}


/**
 * @brief KCFTracker::getLocation
 */
void KCFTracker::getLocation( cv::Point &_center, cv::Point &_size )
{
    _center = cv::Point((int)(m_objBound.center.x + 0.5), (int)(m_objBound.center.y + 0.5));
    _size   = cv::Point((int)(m_objBound.size.width + 0.5), (int)(m_objBound.size.height + 0.5));
}


/**
 * @brief KCFTracker::getLocation
 * @param _bound
 */
void KCFTracker::getLocation( cv::Rect &_bound )
{
    int top   = (int)(m_objBound.center.y - m_objBound.size.height / 2.0 + 0.5),
        left  = (int)(m_objBound.center.x - m_objBound.size.width / 2.0 + 0.5);
    int bot   = top + (int)(m_objBound.size.height + 0.5),
        right = left + (int)(m_objBound.size.width + 0.5);

    top   = (top >= 0) ? top : 0;
    left  = (left >= 0) ? left : 0;
    bot   = (bot < m_config.height) ? bot : (m_config.height - 1);
    right = (right < m_config.width) ? right : (m_config.width - 1);

    _bound = cv::Rect(left, top, right - left, bot - top);
}


/**
 * @brief KCFTracker::getTrackStatus
 */
TrackStatus KCFTracker::getTrackStatus()
{
    return m_trackStatus;
}

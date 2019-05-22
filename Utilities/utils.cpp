#include "utils.h"


namespace kcf {

    /**
     * @brief fftd
     */
    htError_t fftd( const cv::Mat &_src, cv::Mat &_dst, bool _isInvert )
    {        
        if( !_isInvert ) // Perform forward FFT
        {
            if( (_src.type() != CV_32FC1) && (_src.type() != CV_32FC2) )
            {
                LOG_MSG( "[ERR] %s:%d: status = %d: Invalid input matrix\n",
                         __FUNCTION__, __LINE__, htErrorInvalidParameters );
                return htErrorInvalidParameters;
            }

            cv::dft( _src, _dst, cv::DFT_COMPLEX_OUTPUT );
        }
        else    // Perform invert FFT
        {
            if( _src.type() != CV_32FC2 )
            {
                LOG_MSG( "[ERR] %s:%d: status = %d: Invalid input matrix\n",
                         __FUNCTION__, __LINE__, htErrorInvalidParameters );
                return htErrorInvalidParameters;
            }

            cv::dft( _src, _dst, cv::DFT_INVERSE | cv::DFT_SCALE );
        }

        return htSuccess;
    }


    /**
     * @brief real
     */
    cv::Mat real( const cv::Mat &_cmpl )
    {
        cv::Mat ret;
        if( _cmpl.channels() <= 1 )
        {
            ret = _cmpl.clone();
            return ret;
        }
        else if( _cmpl.channels() == 2 )
        {
            cv::Mat planes[2];
            cv::split( _cmpl, planes );
            planes[0].copyTo( ret );
            return ret;
        }
        else
        {
            LOG_MSG( "[WAR] %s:%d: The input matrix has more than 2 channels\n",
                     __FUNCTION__, __LINE__ );
            return ret;
        }
    }


    /**
     * @brief real
     */
    cv::Mat imag( const cv::Mat &_cmpl )
    {
        cv::Mat ret;
        if( _cmpl.channels() <= 1 )
        {
            ret = _cmpl.clone();
            return ret;
        }
        else if( _cmpl.channels() == 2 )
        {
            cv::Mat planes[2];
            cv::split( _cmpl, planes );
            planes[1].copyTo( ret );
            return ret;
        }
        else
        {
            LOG_MSG( "[WAR] %s:%d: The input matrix has more than 2 channels\n",
                     __FUNCTION__, __LINE__ );
            return ret;
        }
    }

    /**
     * @brief complexDivision
     */
    cv::Mat complexDivision( const cv::Mat &a, const cv::Mat &b)
    {

        std::vector<cv::Mat> pa;
        std::vector<cv::Mat> pb;
        cv::split(a, pa);
        cv::split(b, pb);

        cv::Mat divisor = 1. / (pb[0].mul(pb[0]) + pb[1].mul(pb[1]));

        std::vector<cv::Mat> pres;

        pres.push_back((pa[0].mul(pb[0]) + pa[1].mul(pb[1])).mul(divisor));
        pres.push_back((pa[1].mul(pb[0]) - pa[0].mul(pb[1])).mul(divisor));

        cv::Mat res;
        cv::merge(pres, res);
        return res;
    }

    /**
     * @brief createHanningWindow
     */
    htError_t createHanningWindow( cv::Mat &_hann, const int _dimX, const int _dimY, const int _dimZ )
    {
        if( (_dimX <= 0) || (_dimY <= 0) || (_dimZ <= 0) )
        {
            LOG_MSG( "[ERR] %s:%d: status = %d: Invalid input dimensions\n",
                     __FUNCTION__, __LINE__, htErrorInvalidParameters );
            return htErrorInvalidParameters;
        }

        cv::Mat hannX = cv::Mat::zeros( 1, _dimX, CV_32FC1 );
        cv::Mat hannY = cv::Mat::zeros( _dimY, 1, CV_32FC1 );
        for( int i = 0; i < hannX.cols; i++ )
        {
            hannX.at<float>(i) = 0.5 * (1 - cos(2 * CV_PI * (float)i / (float)(hannX.cols - 1)));
        }

        for( int i = 0; i < hannY.rows; i++ )
        {
            hannY.at<float>(i) = 0.5 * (1 - cos(2 * CV_PI * (float)i / (float)(hannY.rows - 1)));
        }

        cv::Mat hann2d = hannY * hannX;
        if( _dimZ == 1 )
        {
            hann2d.copyTo( _hann );
        }
        else
        {
            _hann = cv::Mat::zeros( _dimZ, _dimX * _dimY, CV_32FC1 );
            hann2d = hann2d.reshape( 1, 1 );
            for( int i = 0; i < _dimZ; i++ )
            {
                hann2d.copyTo( _hann.row(i) );
            }
        }

        return htSuccess;
    }


    /**
     * @brief createGaussianWindow
     */
    htError_t createGaussianWindow( cv::Mat &_gauss, const int _dimX, const int _dimY, const float _sigma )
    {
        if( (_dimX <= 0) || (_dimY <= 0) )
        {
            LOG_MSG( "[ERR] %s:%d: status = %d: Invalid input dimensions\n",
                     __FUNCTION__, __LINE__, htErrorInvalidParameters );
            return htErrorInvalidParameters;
        }

        _gauss = cv::Mat::zeros( _dimY, _dimX, CV_32FC1 );
        float cx = (float)_dimX / 2.0,
              cy = (float)_dimY / 2.0;
        float power = - 0.5 / (_sigma * _sigma);
        for( int i = 0; i < _dimY; i++ )
        {
            for( int j = 0; j < _dimX; j++ )
            {
                float dx = (float)(j - cx),
                      dy = (float)(i - cy);
                _gauss.at<float>(i, j) = (float)std::exp( power * (dx*dx + dy*dy) );
            }
        }

        return htSuccess;
    }


    htError_t createTukeyWindow( cv::Mat &_tukey, const int _dimX, const int _dimY, const float _alphaX, const float _alphaY )
    {
        if( (_dimX <= 0) || (_dimY <= 0) )
        {
            LOG_MSG( "[ERR] %s:%d: status = %d: Invalid input dimensions\n",
                     __FUNCTION__, __LINE__, htErrorInvalidParameters );
            return htErrorInvalidParameters;
        }

        if( (_alphaX <= 0) || (_alphaY <= 0) || (_alphaX > 1) || (_alphaY  > 1) )
        {
            LOG_MSG( "[ERR] %s:%d: status = %d: Invalid Tukey factors\n",
                     __FUNCTION__, __LINE__, htErrorInvalidParameters );
            return htErrorInvalidParameters;
        }

        // Create 1-D Tukey windows
        cv::Mat tukeyX = cv::Mat::zeros( 1, _dimX, CV_32FC1 );
        cv::Mat tukeyY = cv::Mat::zeros( _dimY, 1, CV_32FC1 );

        float haX = _alphaX / 2.0,
              haY = _alphaY / 2.0;

        float NX = (float)_dimX - 1,
              NY = (float)_dimY - 1;

        int lowerBound = (int)(NX * haX),
            upperBound = (int)(NX * (1 - haX));
        for( int i = 0; i < _dimX; i++ )
        {
            if( i <= lowerBound )
            {
                tukeyX.at<float>(i) = 0.5 * (1 + cosf( CV_PI * ((float)i / (haX * NX) - 1) ));
            }
            else if( i <= upperBound )
            {
                tukeyX.at<float>(i) = 1.0;
            }
            else
            {
                tukeyX.at<float>(i) = 0.5 * (1 + cosf( CV_PI * ((float)i / (haX * NX) - 1.0 / haX + 1) ));
            }
        }

        lowerBound = (int)(NY * haY),
        upperBound = (int)(NY * (1 - haY));
        for( int i = 0; i < _dimY; i++ )
        {
            if( i <= lowerBound )
            {
                tukeyY.at<float>(i) = 0.5 * (1 + cosf( CV_PI * ((float)i / (haY * (float)NY) - 1) ));
            }
            else if( i <= upperBound )
            {
                tukeyY.at<float>(i) = 1.0;
            }
            else
            {
                tukeyY.at<float>(i) = 0.5 * (1 + cosf( CV_PI * ((float)i / (haY * (float)NY) - 1.0 / haY + 1) ));
            }
        }

        // 2-D Tukey window is obtained by multiplying two 1-D windows
        _tukey = tukeyY * tukeyX;

        return htSuccess;
    }

    /**
     * @brief gaussianCorrelation
     */
    htError_t gaussianCorrelation( const cv::Mat &_x, const cv::Mat &_z, cv::Mat &_corr, const cv::Point3i _mapSize, const float &_sigma )
    {
        htError_t err = htSuccess;
        cv::Mat c = cv::Mat( _mapSize.y, _mapSize.x, CV_32FC1, cv::Scalar::all(0.0) );

        // Compute correlation between corresponding rows of _x & _z
        // then combine them to derive total correlation map
        cv::Mat caux, xaux, zaux;
        cv::Mat CAux, XAux, ZAux;
        for( int idx = 0; idx < _mapSize.z; idx++ )
        {
            _x.row( idx ).copyTo( xaux );
            _z.row( idx ).copyTo( zaux );
            xaux = xaux.reshape( 1, _mapSize.y );
            zaux = zaux.reshape( 1, _mapSize.y );

            // Compute cross correlation using FFT trick
            err |= fftd( xaux, XAux, false );
            err |= fftd( zaux, ZAux, false );

            cv::mulSpectrums( XAux, ZAux, CAux, 0, true );
            err |= fftd( CAux, caux, true );

            // Combine feature channel
            c = c + real( caux );
        }

        if( err != htSuccess )
        {
            LOG_MSG( "[ERR] %s:%d: status = %d: Failed to compute correlation\n",
                     __FUNCTION__, __LINE__, err );
            return err;
        }

        // Apply kernel trick to correlation map
        cv::Mat xx = _x.mul( _x ),
                zz = _z.mul( _z );
        float totalEnergy = (float)(cv::sum( xx )[0] + cv::sum( zz )[0]),
              f = (2.0 * _sigma * _sigma) * (float)(_mapSize.x * _mapSize.y * _mapSize.z);
        cv::Mat d;
        cv::max( totalEnergy - 2.0 * c, 0.0, d );
        cv::exp( -d / f, _corr );

        return htSuccess;
    }


    /**
     * @brief detectPeakNPSR
     */
    htError_t detectPeakNPSR( const cv::Mat &_response, cv::Point &_peakLoc_i, float *_peakVal, float *_psr )
    {
        if( _response.empty() )
        {
            LOG_MSG( "[ERR] %s:%d: status = %d: Empty input matrix\n",
                     __FUNCTION__, __LINE__, htErrorNotAllocated );
            return htErrorNotAllocated;
        }

        float psr ;
        double max_val = 0 ;
        cv::Point max_loc ;
        int psr_mask_sz = 11;
        cv::Mat psr_mask = cv::Mat::ones(_response.rows, _response.cols, CV_8U);
        cv::Scalar mn_;
        cv::Scalar std_;

        // Detect peak
        cv::minMaxLoc(_response, NULL, &max_val, NULL, &max_loc);
        _peakLoc_i = max_loc;
        if( _peakVal )
        {
            *_peakVal  = (float)max_val;
        }

        // Compute PSR value
        int win_sz = floor(psr_mask_sz/2);
        cv::Rect side_lobe = cv::Rect(std::max(max_loc.x - win_sz, 0), std::max(max_loc.y - win_sz,0),11, 11);

        if ((side_lobe.x + side_lobe.width) > psr_mask.cols)
        {
            side_lobe.width = psr_mask.cols - side_lobe.x;
        }
        if ((side_lobe.y + side_lobe.height) > psr_mask.rows)
        {
            side_lobe.height = psr_mask.rows - side_lobe.y;
        }

        cv::Mat tmp = psr_mask(side_lobe);
        tmp *= 0;
        cv::meanStdDev(_response, mn_, std_, psr_mask);
        psr = (max_val - mn_[0])/ (std_[0] + std::numeric_limits<float>::epsilon());
        if( _psr )
        {
            *_psr = (float)psr;
        }

        return  htSuccess;
    }

    /**
     * @brief fixPeak
     */
    float fixPeak( const float left , const float center, const float right )
    {
        float divisor = 2 * center - right - left;
        if (divisor == 0)
            return 0;
        float rate = 0.5 * (right - left) / divisor;

        return rate;
    }

    /**
     * @brief decomposeTransform
     */
    htError_t decomposeTransform( const cv::Mat &_trans, float *_dx, float *_dy, float *_rot, float *_scale )
    {
        if( _trans.empty() || (_trans.type() != CV_32FC1) )
            return htErrorInvalidParameters;

        if( (_trans.at<float>(2, 0) != 0) || (_trans.at<float>(2, 1) != 0) )
            return htErrorInvalidParameters;

        float *data = (float*)_trans.data;

        if( _dx != nullptr )
            *_dx = data[2];

        if( _dy != nullptr )
            *_dy = data[5];

        if( _scale != nullptr )
            *_scale = sqrtf( data[0] * data[0] + data[1] * data[1] );

        if( _rot != nullptr )
        {
            if( data[0] == 0 )
            {
                if( data[1] > 0 )
                    *_rot = CV_PI / 2.0;
                else if( data[1] < 0 )
                    *_rot = CV_PI / 2.0;
                else
                    *_rot = 0.0;

            }
            else
                *_rot = atanf( data[1] / data[0] );
        }

        return htSuccess;
    }


    /**
     * @brief computeHistogram
     */
    htError_t computeHistogram( const cv::Mat &_img, const cv::Mat &_mask, const int _binsPerChannel, float *_hist )
    {
        if( _img.empty() || _hist == NULL )
        {
            LOG_MSG( "[ERR] %s:%d: status = %d: Empty input matrix or null pointer\n",
                     __FUNCTION__, __LINE__, htErrorNotAllocated );
            return htErrorNotAllocated;
        }

        if( !_mask.empty() && ((_mask.cols != _img.cols) || (_mask.rows != _img.rows)) )
        {
            LOG_MSG( "[ERR] %s:%d: status = %d: Two matrice must be equal in size\n",
                     __FUNCTION__, __LINE__, htErrorNotCompatible );
            return htErrorNotCompatible;
        }

        // Compute marginal histogram
        int binWidth = 255 / _binsPerChannel;
        int channels = _img.channels();
        unsigned char *imgData = (unsigned char*)_img.data;
        memset( _hist, 0, channels * _binsPerChannel * sizeof( float ) );
        float sum = 0.0;
        for( int r = 0; r < _img.rows; r++ )
        {
            for( int c = 0; c < _img.cols; c++ )
            {
                int id = (r * _img.cols + c) * channels;
                float w = 1.0;
                if( !_mask.empty() )
                {
                    w = _mask.at<float>(r, c);
                }

                for( int ch = 0; ch < channels; ch++ )
                {
                    int histId = imgData[id+ch] / binWidth;
                    histId = (histId < _binsPerChannel) ? histId : _binsPerChannel-1;
                    _hist[histId + ch*_binsPerChannel] += w;
                }
                sum += channels * w;
            }
        }

        // Normalize
        int len = channels * _binsPerChannel;
        if( sum > 0 )
        {
            for( int i = 0; i < len; i++ )
            {
                _hist[i] /= sum;
            }
        }

        return htSuccess;
    }

    
    /**
     * @brief computeBhattacharyaCoeff
     * @param _vec1
     * @param _vec2
     * @param _len
     * @return 
     */
    float computeBhattacharyaCoeff( const float *_vec1, const float *_vec2, const int _len )
    {
        if( _vec1 == NULL || _vec2 == NULL )
        {
            return 0.0;
        }

        double sum  = 0.0,
               sum1 = 0.0,
               sum2 = 0.0;
        for( int i = 0; i < _len; i++ )
        {

            if( _vec1[i] < 0 || _vec2[i] < 0 )
            {
                return 0.0;
            }
            double prod = _vec1[i] * _vec2[i];
            sum  += sqrt( prod );
            sum1 += (double)_vec1[i];
            sum2 += (double)_vec2[i];
        }

        if( sum1 == 0 || sum2 == 0)
        {
            return 0.0;
        }

        double bhat = sum / sqrt(sum1) / sqrt(sum2);
        return (float)acos( bhat ) * 180.0 / CV_PI;
    }
}

namespace graphix {

    /**
     * @brief rotRectangle
     */
    void rotRectangle( cv::Mat &_img, const cv::RotatedRect _rotRect)
    {
        cv::Point2f corners[4];
        _rotRect.points( corners );
        for( int i = 0; i < 4; i++ )
        {
            cv::line( _img, corners[i], corners[(i+1)%4], cv::Scalar(125, 255, 180), 1, cv::LINE_AA );
        }
    }


    /**
     * @brief writeMatToText
     */
    void writeMatToText( const char *_filename , const cv::Mat &hMat )
    {
        std::ofstream fp( _filename );
        if( !fp )
        {
            LOG_MSG( "! ERROR: %s:%d: Failed to open file\n" );
            return;
        }

        for( int i = 0; i < hMat.rows; i++ )
        {
            for( int j = 0; j < hMat.cols; j++ )
            {
                fp << hMat.at<float>(i, j) << ' ';
            }
            fp << std::endl;
        }
        fp.close();
    }

    /**
     * @brief imshowFloat
     * @param _img
     */
    void imshowFloat( const std::string _window, const cv::Mat &_img )
    {
        double minVal, maxVal;
        cv::minMaxLoc( _img, &minVal, &maxVal );
        cv::Mat dispIm;
        if( minVal != maxVal )
        {
            dispIm = (_img - minVal) / (maxVal - minVal);
        }
        else
        {
            dispIm = _img * 0.0;
        }
        cv::imshow( _window, dispIm );
    }
}

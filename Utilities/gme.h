#ifndef GME_H
#define GME_H

#include <opencv2/opencv.hpp>
#include <opencv2/videostab.hpp>

#include "utils.h"

class GME
{
public:
    /**
     * @brief Constructor
     */
    GME( GMEType _type = GMEType::RIGID_TRANSFORM )
    {
        m_type = _type;
        LOG_MSG( "[DEB] %s:%d: A GME object has been created\n", __FUNCTION__, __LINE__ );
    }

    /**
     * @brief Destructor
     */
    ~GME( )
    {
        LOG_MSG( "[DEB] %s:%d: A GME object has been destroyed\n", __FUNCTION__, __LINE__ );
    }

private:
    /**
     * @brief ageDelay
     */
    void ageDelay( );

public:
    htError_t process( const cv::Mat &_img );
    cv::Mat getTrans( );

private:
    GMEType m_type;
    cv::Mat m_prevGray;
    cv::Mat m_currGray;
    cv::Mat m_trans;
};

#endif // GME_H

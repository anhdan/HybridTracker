#ifndef KALMAN_H
#define KALMAN_H

#include <opencv2/opencv.hpp>

/**
 * @brief This class implements Kalman filter for position of a 2-D moving point
 */
class Kalman
{
public:
    /**
     * @brief Constructor
     */
    Kalman() {}

    /**
     * @brief Destructor
     */
    ~Kalman(){}

public:
    /**
     * @brief Sets default value for data of the Kalman model
     */
    void setDefaultModel();

    /**
     * @brief Sets Kalman initial state
     * @param _x    Initial x position
     * @param _y    Initial y position
     * @param _dx   Initial x velocity
     * @param _dy   Initial y velocity
     */
    void setState( const float &_x, const float &_y, const float &_dx, const float &_dy );

    /**
     * @brief Kalman prediction phase
     * @param _dx   Predicted x motion
     * @param _dy   Predicted y motion
     */
    void predictMotion( float *_dx, float *_dy );

    /**
     * @brief Kalman correction phase
     * @param _dx   Input x motion
     * @param _dy   Input y motion
     */
    void correctModel( const float &_dx, const float &_dy );

private:
    cv::Mat m_A;    /**< A matrix */
    cv::Mat m_H;    /**< H matrix */
    cv::Mat m_Q;    /**< Q matrix */
    cv::Mat m_R;    /**< R matrix */

    cv::Mat m_P;    /**< Prior P matrix */
    cv::Mat m_P_;   /**< Posterior P matrix */

    cv::Mat m_x;    /**< Prior x */
    cv::Mat m_x_;   /**< Posterior x */
    cv::Mat m_z;    /**< Mesurement z */
};

#endif // KALMAN_H

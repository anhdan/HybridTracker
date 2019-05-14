#include "kalman.h"

/**
 * @brief KalmanFilter::setDefaultModel
 */
void Kalman::setDefaultModel()
{
    m_A = (cv::Mat_<float>(4, 4) << 1, 0, 0.995, 0,
                                    0, 1, 0, 0.995,
                                    0, 0, 1, 0,
                                    0, 0, 0, 1);

    m_H = cv::Mat(2, 4, CV_32F);
    cv::setIdentity(m_H);

    m_Q = cv::Mat(4, 4, CV_32F);
    cv::setIdentity(m_Q, cv::Scalar::all(1e-2));

    m_R = (cv::Mat_<float>(2, 2) << 0.2845, 0.0045,
                                    0.0045, 0.0455);

    m_P = cv::Mat(4, 4, CV_32F);
    cv::setIdentity(m_P, cv::Scalar::all(0.1));
}


/**
 * @brief Kalman::setState
 */
void Kalman::setState(const float &x, const float &y, const float &dx, const float &dy)
{
    m_x = (cv::Mat_<float>(4, 1) << x, y, dx, dy);
    m_z = (cv::Mat_<float>(2, 1) << x, y);
}


/**
 * @brief KalmanFilter::predictMotion
 */
void Kalman::predictMotion( float *dx, float *dy )
{
    m_x_ = m_A * m_x;
    m_P_ = m_A * m_P * m_A.t() + m_Q;

    *dx  = m_x_.at<float>(2, 0);
    *dy  = m_x_.at<float>(3, 0);
}


/**
 * @brief KalmanFilter::correctModel
 */
void Kalman::correctModel(const float &dx, const float &dy)
{
    m_z.at<float>(0, 0) += dx;
    m_z.at<float>(1, 0) += dy;

    cv::Mat tmp = m_H * m_P_ * m_H.t() + m_R;
    cv::Mat K   = m_P_ *  m_H.t() * tmp.inv();
    m_x = m_x_ + K * (m_z - m_H * m_x_);

    cv::Mat I = cv::Mat::eye(4, 4, CV_32FC1);
    m_P = (I - K * m_H) * m_P_;
}

#include <iostream>
#include <math.h>
#include <random>
#include <time.h>
#include <stdlib.h>
#include <ctime>

#include "KCFTracker/kcfTracker.h"
#include "Utilities/saliency.h"

using namespace std;

struct TrackingEvent
{
    bool objectSelected;
    bool trackInited;
    cv::Point origin;

    TrackingEvent ()
    {
        objectSelected = false;
        trackInited = false;
        origin = cv::Point(0, 0);
    }
};

void onMouseCallBack( int _event, int _x, int _y, int /*_flag*/, void *_userData );


/** ===================================================
 *
 *              main
 *
 * ====================================================
 */
int main( int argc, char **argv )
{
    assert( argc == 4 );
    int openSize = atoi( argv[2] );
    int waitTime = atoi( argv[3] );
    cv::VideoCapture cap( argv[1] );
    if( !cap.isOpened() )
    {
        LOG_MSG( "[ERR] %s:%d: Failed to open video source\n", __FUNCTION__, __LINE__ );
        return EXIT_FAILURE;
    }

    //===== Create window with mouse interaction
    std::string windowName = "KCF Tracking";
    TrackingEvent trackEvent;
    cv::namedWindow( windowName );
    cv::setMouseCallback( windowName, onMouseCallBack, (void*)&trackEvent );

    KCFTracker tracker;
    cv::Mat inputFrame, dispFrame;
    std::vector<cv::Rect> trajectory;
    while( true )
    {
        if( !cap.read( inputFrame ) )
        {
            LOG_MSG( "[DEB] %s:%d: End of video source\n", __FUNCTION__, __LINE__ );
            break;
        }
        inputFrame.copyTo( dispFrame );

        if( trackEvent.objectSelected && !trackEvent.trackInited )
        {
            //----- Get initial image patch
            cv::Rect roiOpen;
            roiOpen.x       = ((trackEvent.origin.x - openSize) < 0) ? 0 : (trackEvent.origin.x - openSize);
            roiOpen.y       = ((trackEvent.origin.y - openSize) < 0) ? 0 : (trackEvent.origin.y - openSize);
            roiOpen.width   = ((roiOpen.x + 2*openSize) >= inputFrame.cols) ? (inputFrame.cols - roiOpen.x) : (2 * openSize);
            roiOpen.height  = ((roiOpen.y + 2*openSize) >= inputFrame.rows) ? (inputFrame.rows - roiOpen.y) : (2 * openSize);

            cv::Mat patchOpen = inputFrame(roiOpen).clone();

            //----- Get adjusted image patch by saliency
            cv::Rect roiSaliency = saliency( patchOpen );
            roiSaliency.x += roiOpen.x;
            roiSaliency.y += roiOpen.y;

            //----- Initialize tracking algorithm
            cv::Point initCenter = cv::Point(roiSaliency.x + roiSaliency.width/2, roiSaliency.y + roiSaliency.height/2);
            cv::Size initSize = cv::Size(roiSaliency.width, roiSaliency.height);
            tracker.init( inputFrame, initCenter, initSize );
            trackEvent.objectSelected = false;
            trackEvent.trackInited = true;
        }

        if( trackEvent.trackInited )
        {
            assert( tracker.process( inputFrame ) == htSuccess );
            if( tracker.getTrackStatus() != TrackStatus::TRACK_LOST )
            {
                cv::Rect objBound;
                tracker.getLocation( objBound );

                trajectory.push_back( objBound );
                int num = ((int)trajectory.size() < 15)? trajectory.size() : 15;
                for (int i = (int)trajectory.size()-1; i >= (int)trajectory.size()-num; i--){
                    cv::Point center(trajectory[i].x + trajectory[i].width / 2, trajectory[i].y + trajectory[i].height / 2);
                    cv::circle( dispFrame, center, 0.2, cv::Scalar(100, 80, 0), 8, 0);
                }
                cv::rectangle( dispFrame, objBound, cv::Scalar(0,0,0), 2, 4);
            }
        }

        cv::imshow( windowName, dispFrame );
        cv::waitKey( waitTime );
    }

    return 0;
}


/** ===================================================
 *
 *              function
 *
 * ====================================================
 */
void onMouseCallBack(int _event, int _x, int _y, int /*_flag*/, void *_userData)
{
    TrackingEvent *trackEvent = (TrackingEvent*)_userData;
    if( _event == cv::EVENT_LBUTTONDOWN )
    {
        trackEvent->origin.x = _x;
        trackEvent->origin.y = _y;
        printf( "Click location: [%d, %d]\n", trackEvent->origin.x, trackEvent->origin.y );
    }
    else if( _event == cv::EVENT_LBUTTONUP )
    {
        if( !trackEvent->trackInited )
            trackEvent->objectSelected = true;
    }
}

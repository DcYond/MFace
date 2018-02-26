#include "FD.h"
#include <iostream>
#include <sys/time.h>
#include <opencv2/opencv.hpp>

static float getElapse(struct timeval *tv1,struct timeval *tv2){
    float t = 0.0f;
    if (tv1->tv_sec == tv2->tv_sec)
        t = (tv2->tv_usec - tv1->tv_usec)/1000.0f;
    else
        t = ((tv2->tv_sec - tv1->tv_sec) * 1000 * 1000 + tv2->tv_usec - tv1->tv_usec)/1000.0f;
    return t;
}

int test_video()
{



    std::vector<std::string> model_file = {
        "./model/Det1.param",
        "./model/Det2.param",
        "./model/Det3.param"
    };

    std::vector<std::string> trained_file = {
        "./model/Det1.bin",
        "./model/Det2.bin",
        "./model/Det3.bin"
    };
    FD fd(model_file, trained_file);
    cv::VideoCapture mVideoCapture("./multi_face.avi");
    //cv::VideoCapture mVideoCapture(0);
    if(!mVideoCapture.isOpened()){
        return -1;
    }
    cv::Mat frame;

    struct timeval  tv1,tv2;
    struct timezone tz1,tz2;
    gettimeofday(&tv1,&tz1);

    int num = 0;
    mVideoCapture>>frame;
    while(!frame.empty())
    {
        mVideoCapture>>frame;
        if(frame.empty()){
            break;
        }


        std::vector<Bbox> finalBbox;

        ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(frame.data, ncnn::Mat::PIXEL_BGR, frame.cols, frame.rows);

        fd.detect(ncnn_img, finalBbox);
        for(int i = 0; i < finalBbox.size(); i++){
            cv::Rect box = cv::Rect(finalBbox[i].x1, finalBbox[i].y1, finalBbox[i].x2 - finalBbox[i].x1 + 1, finalBbox[i].y2 - finalBbox[i].y1 + 1);
            cv::rectangle(frame, box, cv::Scalar(0,0,255), 2,8,0);
            for(int j = 0; j < 5; j++)
            {
                cv::circle(frame,cv::Point(finalBbox[i].ppoint[j],finalBbox[i].ppoint[j+5]),2,cv::Scalar(255,0,0,0),2);
            }
        }
        imshow("face_detection",frame);
        num++;

        if(cv::waitKey(1)>1)
        {
            break;
        }
    }

    gettimeofday(&tv2,&tz2);
    printf( "%s = %g ms \n ", "Detection All time", getElapse(&tv1, &tv2));
    printf( "%s = %d, averagetime: %g ms \n ", "frame", num,getElapse(&tv1, &tv2)/num);
}




int main(int argc, char** argv)
{

    test_video();
    return 1;
}

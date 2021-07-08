#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>


using namespace cv;
using namespace std;

// PanoramicImage Class
class Detection{

// Methods

public:

    // Constructor
    Detection(String pattern);
    
    // Methods
    void pyramid(double scale, Mat image, int stepSize, int windowSize);
    void HOG(Mat image);
    void preprocessing(String pattern);
    
// Data

public:

    // List of test images
    vector<Mat> test_images;
};

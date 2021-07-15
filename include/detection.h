#include <opencv2/core.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <opencv2/ximgproc/segmentation.hpp>
#include <opencv2/objdetect.hpp>

#include <opencv2/photo.hpp>

using namespace cv;
using namespace std;

// PanoramicImage Class
class Detection{

// Methods

public:

    // Constructor
    Detection(String pattern);
    
    // Methods
    void dataset(String path);
    void pyramid(double scale, Mat image, int stepSize, int windowSize_rows, int windowSize_cols, String model_path_pb);
    void Kmeans_segmentation(int k);
    void otsu_segmentation();
    void showAndCompute_sift();
    void preprocessing(String pattern);
    
    vector<Rect> selective_search(Mat image, String method);
    
// Data

public:

    // List of test images
    vector<Mat> test_images;
};

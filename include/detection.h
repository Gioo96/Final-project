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
#include <filesystem>

using namespace cv;
using namespace std;

// Dataset Class
class Dataset {

public:

    // Constructor
    Dataset(String images_path, String model_CNN_path);
};

// Detection Class
class Detection {

public:

    // Constructor
    Detection(String pattern, int flag);
    
    // Methods
    void sliding_window(Mat image, int stepSize, int windowSize_rows, int windowSize_cols, String model_CNN_pb, bool use_cnn, vector<pair<Point, Point>> lines);
    vector<Rect> rect_return(vector<Rect> all_rects, vector<Rect> seed_rects, vector<double> input_probabilities, vector<double> &output_probabilities);
    void Kmeans_color_segmentation(int k);
    void otsu_segmentation();
    void preprocessing(String pattern);
    
    vector<Rect> selective_search(Mat image, String method);
    
// Data

public:

    // List of test images
    vector<Mat> test_images;
};

// Class implementing the Canny edge detector
class Canny_edge{

// Methods

public:

    // CANNY
    // Constructor
    Canny_edge(cv::Mat input_image, int th1, int th2, int apertureSize);
    
    // Performs Canny edge detector algorithm
    void doAlgorithm();
    
    // Set threshold1 for Canny
    void setThreshold1(int th1);
    
    // Set threshold2 for Canny
    void setThreshold2(int th2);
    
    // Get threshold1
    int getThreshold1();
    
    // Get threshold2
    int getThreshold2();
    
    // Get result
    cv::Mat getResult();
    
    static void onCannyThreshold_1(int pos, void *userdata);
    static void onCannyThreshold_2(int pos, void *userdata);

// Data

protected:

    // Aperture size of the Sobel operator (Canny)
    int aperture_size_Canny;
    
    // threshold_1
    int threshold1_Canny;
    
    // threshold_2
    int threshold2_Canny;
    
    // Input image
    Mat input_image_canny;
    
    // Result image
    Mat result_image_canny;
};

// HOUGH LINE
// Class implementing the Hough line detector
class HoughLine{

// Methods

public:

    // Constructor
    HoughLine(cv::Mat input_image, int rho, double theta, int threshold);
    
    // Performs Hough line detector algorithm
    void doAlgorithm();
    
    // Set rho for HoughLine
    void setRho(int rho);
    
    // Set theta for HoughLine
    void setTheta(double theta);
    
    // Set threshold for HoughLine
    void setThreshold(int threshold);
    
    // Get rho
    int getRho();
    
    // Get theta
    double getTheta();
    
    // Get threshold
    int getThreshold();
    
    std::vector<pair<Point, Point>> getLines();
    
    cv::Mat getResult();
    
    static void onHoughLineThetaden(int pos, void *userdata);
    
    void drawStraightLine(cv::Mat& img, cv::Point p1, cv::Point p2, cv::Scalar color);
        

// Data

protected:

    // rho
    int rho_HoughLine;
    
    // theta
    double theta_HoughLine;
    
    // threshold
    int threshold_HoughLine;
    
    std::vector<pair<Point, Point>> lines;
    
    // Input image
    Mat input_image_hough;
    
    // Result image
    Mat result_image_hough;
};

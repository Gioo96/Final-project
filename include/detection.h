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


int click_pair = 0;
std::vector<std::pair<cv::Point, cv::Point>> vec_pairs;
std::pair<cv::Point, cv::Point> p1_p2;
cv::Point p1, p2;
//cv::Mat mask;

using namespace cv;
using namespace std;

// DATASET CLASS
class Dataset {

public:

    // CONSTRUCTOR
    Dataset(String images_path, String model_CNN_path);
};

// BASE CLASS
class Base {
    
public:

    // CONSTRUCTOR
    Base(String pattern, int fl);
    
// DATA
public:

    // List of test images
    vector<Mat> test_images;
    // Kaggle dataset: flag = 1, Venice dataset: flag = 0
    int flag;
};

// DETECTION CLASS
class Detection : public Base {

public:

    // CONSTRUCTOR
    Detection(String pattern, int fl);
    
    // METHODS
    // sliding_window
    void sliding_window(Mat image, int stepSize, int windowSize_rows, int windowSize_cols, String model_CNN_pb);
    // rect_return
    vector<Rect> rect_return(vector<Rect> all_rects, vector<Rect> seed_rects, vector<double> input_probabilities, vector<double> &output_probabilities);
};

// SEGMENTATION CLASS
class Segmentation : public Base {

public:

    // CONSTRUCTOR
    Segmentation(String pattern, int fl);
    
    // METHODS
    // segmentation
    vector<Mat> segmentation();
    //click
    void click(Mat image);
    //onMouse
    static void onMouse(int event, int x, int y, int f, void* userdata);
    // pixel_accuracy
    vector<double> pixel_accuracy(String ground_truth_images, vector<Mat> segmentated_images);
    //swap_colors
    void swap_colors(Mat& image);
};

// CANNY_EDGE CLASS
class Canny_edge {

public:

    // CONSTRUCTOR
    Canny_edge(cv::Mat input_image, int th1, int th2, int apertureSize);
    
    // METHODS
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
    // onCannyThreshold_1
    static void onCannyThreshold_1(int pos, void *userdata);
    // onCannyThreshold_2
    static void onCannyThreshold_2(int pos, void *userdata);

// DATA
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

// HOUGH LINE CLASS
class HoughLine{

public:

    // CONSTRUCTOR
    HoughLine(cv::Mat input_image, int rho, double theta, int threshold);
    
    // METHODS
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
    // getLines
    std::vector<pair<Point, Point>> getLines();
    // getResult
    cv::Mat getResult();
    //onHoughLineThetaden
    static void onHoughLineThetaden(int pos, void *userdata);
    // drawStraightLine
    void drawStraightLine(cv::Mat& img, cv::Point p1, cv::Point p2, cv::Scalar color);
        

// DATA
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

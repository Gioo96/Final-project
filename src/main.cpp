#include "detection.cpp"
//#include <fstream>


using namespace std;
using namespace cv;

int main(int argc, const char * argv[]) {
    
    String pattern_venice = "/Users/gioel/Documents/Control System Engineering/Computer Vision/Final Project/data/venice_dataset";
    String pattern_kaggle = "/Users/gioel/Documents/Control System Engineering/Computer Vision/Final Project/data/Kaggle_ships";
    String model_CNN_pb = "/Users/gioel/Documents/Control\ System\ Engineering/Computer\ Vision/Final\ Project/Model/model_CNN.pb";
    String ground_truth_segmentation_path = "/Users/gioel/Documents/Control\ System\ Engineering/Computer\ Vision/Final\ Project/data/Ground_truth/Segmentation/";
    
    //
    // KAGGLE DATASET
    //
    cout<<"KAGGLE DATASET"<<endl<<endl<<"DETECTION";
    Detection kaggle(pattern_kaggle, 1);

    // Test images on which the performance is needed to be evaluated
    vector<Mat> test_images = kaggle.test_images;

    // Detection
    //kaggle.detection(test_images, model_CNN_pb);

    cout<<"--------------"<<endl<<"--------------"<<endl;

    // Segmentation
    cout<<"SEGMENTATION";
    Segmentation segmentation(pattern_kaggle, 1);
    //vector<Mat> segmented_images = segmentation.segmentation(ground_truth_segmentation_path);
    

    //
    // VENICE DATASET
    //
//    cout<<"VENICE DATASET"<<endl<<endl<<"DETECTION";
    Detection venice(pattern_venice, 0);

    vector<Mat> test_imagess = venice.test_images;
//
//    // Equalization on V channel
//    vector<Mat> image_equalized;
//    for (int i = 0; i<test_images.size(); i++){
//
//        vector<Mat> planes;
//        Mat image_equal;
//        Mat image_hsv;
//        vector<Mat> equalized_channel;
//
//        cvtColor(test_images.at(i), image_hsv, COLOR_BGR2HSV);
//        split(image_hsv, planes);
//        equalizeHist(planes[2], planes[2]);
//        merge(planes,image_equal);
//        cvtColor(image_equal, image_equal, COLOR_HSV2BGR);
//        image_equalized.push_back(image_equal);
//    }
//
//    // Smoothing
//    vector<Mat> image_smoothed;
//    for (int i = 0; i<test_images.size(); i++){
//
//        Mat image_smooth;
//        GaussianBlur(image_equalized.at(i), image_smooth, Size(3,3), 0);
//        image_smoothed.push_back(image_smooth);
//    }
//
//    // Detection
    //venice.detection(test_imagess, model_CNN_pb);
   
    // Segmentation
    cout<<"SEGMENTATION";
    Segmentation segmentationn(pattern_venice, 0);
    vector<Mat> segmented_images = segmentationn.segmentation(ground_truth_segmentation_path);
    
    return 0;
}

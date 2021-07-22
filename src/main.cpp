#include "detection.cpp"


using namespace std;
using namespace cv;

int main(int argc, const char * argv[]) {

    
    String pattern_venice = "/Users/gioel/Documents/Control System Engineering/Computer Vision/Final Project/data/venice_dataset";

    String pattern_kaggle = "/Users/gioel/Documents/Control System Engineering/Computer Vision/Final Project/data/Kaggle_ships";
    
    
    Detection detection(pattern_kaggle, 1);
    
    String model_CNN_pb = "/Users/gioel/Documents/Control\ System\ Engineering/Computer\ Vision/Final\ Project/Model/model_cnn.pb";

    int stepSize = 40;
    int windowSize_rows = 80;
    int windowSize_cols = 100;

    //vector<pair<Point, Point>> lines;
    //boat.sliding_window(boat.test_images.at(0), stepSize, windowSize_rows, windowSize_cols, model_CNN_pb, true, lines);
    
    //Mat im = imread("/Users/gioel/Documents/Control\ System\ Engineering/Computer\ Vision/Final\ Project/data/Kaggle_ships/boat-ferry-departure-crossing-sea-2733061.jpg", IMREAD_COLOR);

    //boat.flag = 0;
   
    vector<Mat> segmented_images;
    
    Segmentation segmentation(pattern_venice, 0);
    segmented_images = segmentation.segmentation();
    String ground_truth_segmentation_path = "/Users/gioel/Documents/Control\ System\ Engineering/Computer\ Vision/Final\ Project/data/Ground_truth/Segmentation/";
    vector<double> pixel_accuracy;
    //boat.click(boat.test_images.at(11));
    pixel_accuracy = segmentation.pixel_accuracy(ground_truth_segmentation_path, segmented_images);
    
    return 0;
}

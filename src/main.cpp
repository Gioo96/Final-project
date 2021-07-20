#include "detection.cpp"
#include <fstream>


using namespace std;
using namespace cv;

int main(int argc, const char * argv[]) {
    
    
    
    String pattern_venice = "/Users/gioel/Documents/Control System Engineering/Computer Vision/Final Project/data/venice_dataset";
//s
    String pattern_kaggle = "/Users/gioel/Documents/Control System Engineering/Computer Vision/Final Project/data/Kaggle_ships";
    
    
    Detection boat(pattern_kaggle, 1);
    
    String model_CNN_pb = "/Users/gioel/Documents/Control\ System\ Engineering/Computer\ Vision/Final\ Project/Model/model_cnn.pb";

    int stepSize = 40;
    int windowSize_rows = 120;
    int windowSize_cols = 120;

    vector<pair<Point, Point>> lines;
    boat.sliding_window(boat.test_images.at(4), stepSize, windowSize_rows, windowSize_cols, model_CNN_pb, true, lines);
    
    //Mat im = imread("/Users/gioel/Documents/Control\ System\ Engineering/Computer\ Vision/Final\ Project/data/Kaggle_ships/boat-ferry-departure-crossing-sea-2733061.jpg", IMREAD_COLOR);

    //boat.Kmeans_color_segmentation(2);
    
    
    
    //
    //
    //
//    // SMOOTHING
    
    
//    Mat image_test = boat.test_images.at(2).clone();
//    GaussianBlur(image_test, image_test, Size (13,13), 0, 0);
//    //
//    //
//    cout<<canny.getResult().size()<<endl;
//
//    // Detected lines (output of the hough line algorithm)
//    vector<Vec2f> detected_lines = hough_l.getLines(); // (rho,theta) coordinates
    
    
    
    
    
    
    
//    Mat im = boat.test_images.at(9).clone();
//    vector<Rect> rects = boat.selective_search(im, "fast");
//    vector<Rect> saved;
//    for(int i = 0; i < rects.size(); i ++) {
//
//        // Resize window (it has to be 300x300 since CNN was trained with images with such sizes)
//        Mat window = boat.test_images.at(9)(rects.at(i));
//        resize(window, window, Size(300, 300));
//        // Load CNN model
//        dnn::Net net = dnn::readNetFromTensorflow(model_path_pb);
//        Mat img_toNet_blob = cv::dnn::blobFromImage(window);
//        net.setInput(img_toNet_blob);
//        Mat prob;
//        net.forward(prob);
//        if (round(prob.at<float>(0)) == 1) {
//
//            saved.push_back(rects.at(i));
//        }
//    }
//    for (int i=0;i<saved.size();i++) {
//
//        rectangle(im, saved.at(i), cv::Scalar(0, 255, 0), 3);
//    }
//    Mat mer = boat.test_images.at(1);
//    imshow("all", im);
//    waitKey(0);
//    groupRectangles(saved, 3, 0.6);
//    for(int i = 0; i < saved.size(); i ++) {
//
//        rectangle(mer, rects.at(i), cv::Scalar(0, 255, 0), 3);
//    }
//    imshow("merging", mer);
//    waitKey(0);
    
//    Point classIdPoint;
//    double confidence;
//    minMaxLoc(prob.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
//    int classId = classIdPoint.x;
//
//    // Put efficiency information.
//    Mat frame;
//    vector<String> classes;
//    classes.push_back("Boat");
//    classes.push_back("No Boat");
//    std::vector<double> layersTimes;
//    double freq = getTickFrequency() / 1000;
//    double t = net.getPerfProfile(layersTimes) / freq;
//    std::string label = format("Inference time: %.2f ms", t);
//    putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
//    // Print predicted class.
//    label = format("%s: %.4f", (classes.empty() ? format("Class #%d", classId).c_str() :
//                                             classes[classId].c_str()),
//                          confidence);
//    putText(frame, label, Point(0, 40), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
//    imshow("Deep learning image classification in OpenCV", frame);
    
    return 0;
}

#include "detection.cpp"
#include <fstream>
#include <string>


using namespace std;
using namespace cv;

int main(int argc, const char * argv[]) {

    // PREPROCESSING DATASET IMAGES //
    //
    //
    //
    
    // Training images for dataset "MarDCT"
//    String MarDCT_training = "/Users/gioel/Documents/Control System Engineering/Computer Vision/Final Project/data/Classification/MarDCT/training/";
//    String pattern_dest_boat = MarDCT_training + "Boat/";
//    String pattern_dest_noboat = MarDCT_training + "No Boat/";
//
//    vector<String> categories = {"Alilaguna", "Ambulanza", "Barchino", "Cacciapesca", "Caorlina", "Gondola", "Lanciafino10m", "Lanciafino10mBianca", "Lanciafino10mMarrone", "Lanciamaggioredi10mBianca", "Lanciamaggioredi10mMarrone", "Motobarca", "Motopontonerettangolare", "MotoscafoACTV", "Mototopo", "Patanella", "Polizia", "Raccoltarifiuti", "Sandoloaremi", "Sanpierota", "Topa", "VaporettoACTV", "VigilidelFuoco", "Water"};
//
//    int num_samples = 0;
//    for (int i=0; i<categories.size(); i++) {
//
//        String MarDCT_training_cat = MarDCT_training + categories.at(i) + "/*.jpg";
//        vector<String> fn;
//        glob(MarDCT_training_cat, fn, false);
//
//        if (categories.at(i) == "Water") {
//
//            int count = fn.size(); // Number of images per category
//            for (int j = 0; j < count; j++) {
//
//                Mat single_img_gray = imread(fn[j], IMREAD_COLOR);
//                imwrite(pattern_dest_noboat  + "img" + to_string(num_samples) + ".jpg", single_img_gray);
//                num_samples += 1;
//            }
//        }
//        else {
//
//            int count = fn.size(); // Number of images per category
//            for (int j = 0; j < count; j++) {
//
//                Mat single_img_gray = imread(fn[j], IMREAD_COLOR);
//                imwrite(pattern_dest_boat  + "img" + to_string(num_samples) + ".jpg", single_img_gray);
//                num_samples += 1;
//            }
//        }
//    }
//
//    // Test images for dataset "MarDCT"
//    String MarDCT_test = "/Users/gioel/Documents/Control System Engineering/Computer Vision/Final Project/data/Classification/MarDCT/test/";
//    String line;
//    ifstream myfile (MarDCT_test + "ground_truth.txt");
//    int num_row = 0;
//    if (myfile.is_open()) {
//        while (getline(myfile,line)) {
//
//            String image;
//            String label;
//            for (int i=0; i<line.size(); i++) {
//
//                if (i < 25) {
//
//                    image += line.at(i);
//                }
//                else if (i > 25) {
//
//                    label += line.at(i);
//                }
//            }
//            Mat single_img_gray = imread(MarDCT_test + image, IMREAD_COLOR);
//            cout<<num_row<<"  "<<label<<" "<<image<<endl;
//            if (label.size() == 15 && label.at(0) == 'S') {
//
//                imwrite(MarDCT_test  + "No Boat/" + "img" + to_string(num_row) + ".jpg", single_img_gray);
//            }
//            else {
//
//                imwrite(MarDCT_test  + "Boat/" + "img" + to_string(num_row) + ".jpg", single_img_gray);
//            }
//
//            num_row ++;
//        }
//        myfile.close();
//      }
    
    
    
    String pattern = "/Users/gioel/Documents/Control System Engineering/Computer Vision/Final Project/data/venice_dataset";
//
    Detection boat(pattern);
//    int stepSize = 50;
//    int windowSize = 200;
//    double scale = 1.5;
//
//    Mat image = boat.test_images.at(1);
//    //boat.pyramid(scale, image, stepSize, windowSize);
//    boat.HOG(image);
    
    String model1 = "/Users/gioel/Documents/Control System Engineering/Computer Vision/Final Project/data/my_model/saved_model.pb";
    dnn::Net net = dnn::readNetFromTensorflow(model);
    Mat input = boat.test_images.at(0);
    net.setInput(input);
    if (net.empty()) {
        
        cout<<"Noooooo"<<endl;
    }
    Mat prob = net.forward();
    Point classIdPoint;
    double confidence;
    minMaxLoc(prob.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
    int classId = classIdPoint.x;
    
    // Put efficiency information.
    Mat frame;
    vector<String> classes;
    classes.push_back("Boat");
    classes.push_back("No Boat");
    std::vector<double> layersTimes;
    double freq = getTickFrequency() / 1000;
    double t = net.getPerfProfile(layersTimes) / freq;
    std::string label = format("Inference time: %.2f ms", t);
    putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
    // Print predicted class.
    label = format("%s: %.4f", (classes.empty() ? format("Class #%d", classId).c_str() :
                                             classes[classId].c_str()),
                          confidence);
    putText(frame, label, Point(0, 40), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
    imshow("Deep learning image classification in OpenCV", frame);
    
    return 0;
}

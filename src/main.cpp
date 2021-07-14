#include "detection.cpp"
#include <fstream>


using namespace std;
using namespace cv;

int main(int argc, const char * argv[]) {

    // PREPROCESSING DATASET IMAGES //
    //
    //
    //
    
//    String MarDCT_training = "/Users/gioel/Documents/Control System Engineering/Computer Vision/Final Project/data/Classification/MarDCT/training/";
//    String Kaggle_training_boat = "/Users/gioel/Desktop/Kaggle training Boat/";
//    String pattern_dest_training_boat = "/Users/gioel/Desktop/Training/Boat/";
//    String pattern_dest_training_noboat = "/Users/gioel/Desktop/Training/No Boat/";
//    // MAR DCT TRAINING
//    //String pattern_dest_noboat = MarDCT_training + "No Boat/";
//
//    vector<String> categories = {"Alilaguna", "Ambulanza", "Barchino", "Cacciapesca", "Caorlina", "Gondola", "Lanciafino10m", "Lanciafino10mBianca", "Lanciafino10mMarrone", "Lanciamaggioredi10mBianca", "Lanciamaggioredi10mMarrone", "Motobarca", "Motopontonerettangolare", "MotoscafoACTV", "Mototopo", "Patanella", "Polizia", "Raccoltarifiuti", "Sandoloaremi", "Sanpierota", "Topa", "VaporettoACTV", "VigilidelFuoco", "Water"};
//
//    int num_samples_training_boat = 0;
//    int num_samples_training_noboat = 0;
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
//                resize(single_img_gray, single_img_gray, Size(300, 300));
//                //single_img_gray.convertTo(single_img_gray, CV_32F, 1.0 / 255, 0);
//                imwrite(pattern_dest_training_noboat  + "img" + to_string(num_samples_training_noboat) + ".jpg", single_img_gray);
//                num_samples_training_noboat += 1;
//            }
//        }
//        else {
//
//            int count = fn.size(); // Number of images per category
//            for (int j = 0; j < static_cast<int>(count*0.6); j++) {
//
//                Mat single_img_gray = imread(fn[j], IMREAD_COLOR);
//                resize(single_img_gray, single_img_gray, Size(300, 300));
//                //single_img_gray.convertTo(single_img_gray, CV_32F, 1.0 / 255, 0);
//                imwrite(pattern_dest_training_boat  + "img" + to_string(num_samples_training_boat) + ".jpg", single_img_gray);
//                num_samples_training_boat += 1;
//            }
//        }
//    }
//
//    // KAGGLE TRAINING
//
//    String Kaggle_training_set = Kaggle_training_boat + "/*.jpg";
//    vector<String> fn1;
//    glob(Kaggle_training_set, fn1, false);
//
//    int count1 = fn1.size();
//    for (int i = 0; i <count1; i++) {
//
//        Mat single_img_gray = imread(fn1[i], IMREAD_COLOR);
//        resize(single_img_gray, single_img_gray, Size(300, 300));
//        //single_img_gray.convertTo(single_img_gray, CV_32F, 1.0 / 255, 0);
//        imwrite(pattern_dest_training_boat  + "img" + to_string(num_samples_training_boat) + ".jpg", single_img_gray);
//        num_samples_training_boat += 1;
//    }

    
    //Test images for dataset "MarDCT"
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
//            resize(single_img_gray, single_img_gray, Size(300, 300));
//            //single_img_gray.convertTo(single_img_gray, CV_32F, 1.0 / 255, 0);
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
    
//    String pattern_dest_test_boat = "/Users/gioel/Desktop/Test/Boat/";
//    String pattern_dest_test_noboat = "/Users/gioel/Desktop/Test/No Boat/";
//
//    int num_samples_test_boat = 0;
//    int num_samples_test_noboat = 0;
//
//    // MAR DCT TEST
//    String MarDCT_test = "/Users/gioel/Documents/Control System Engineering/Computer Vision/Final Project/data/Classification/MarDCT/test/";
//    String MarDCT_test_boat = MarDCT_test + "Boat/";
//    String MarDCT_test_noboat = MarDCT_test + "No Boat/";
//
//    String MarDCT_test_boat_set = MarDCT_test_boat + "/*.jpg";
//    String MarDCT_test_noboat_set = MarDCT_test_noboat + "/*.jpg";
//    vector<String> fn2;
//    glob(MarDCT_test_boat_set, fn2, false);
//
//    int count2 = fn2.size();
//    for (int i = 0; i <count2; i++) {
//
//        Mat single_img_gray = imread(fn2[i], IMREAD_COLOR);
//        resize(single_img_gray, single_img_gray, Size(300, 300));
//        //single_img_gray.convertTo(single_img_gray, CV_32F, 1.0 / 255, 0);
//        imwrite(pattern_dest_test_boat  + "img" + to_string(num_samples_test_boat) + ".jpg", single_img_gray);
//        num_samples_test_boat += 1;
//    }
//
//    vector<String> fn3;
//    glob(MarDCT_test_noboat_set, fn3, false);
//
//    int count3 = fn3.size();
//    for (int i = 0; i <count3; i++) {
//
//        Mat single_img_gray = imread(fn3[i], IMREAD_COLOR);
//        resize(single_img_gray, single_img_gray, Size(300, 300));
//        //single_img_gray.convertTo(single_img_gray, CV_32F, 1.0 / 255, 0);
//        imwrite(pattern_dest_test_noboat  + "img" + to_string(num_samples_test_noboat) + ".jpg", single_img_gray);
//        num_samples_test_noboat += 1;
//    }
//
//    // KAGGLE TEST
//    String Kaggle_test_boat_set = "/Users/gioel/Desktop/Kaggle test Boat/*.jpg";
//
//    vector<String> fn4;
//    glob(Kaggle_test_boat_set, fn4, false);
//
//    int count4 = fn4.size();
//    for (int i = 0; i <count4; i++) {
//
//        Mat single_img_gray = imread(fn4[i], IMREAD_COLOR);
//        resize(single_img_gray, single_img_gray, Size(300, 300));
//        //single_img_gray.convertTo(single_img_gray, CV_32F, 1.0 / 255, 0);
//        imwrite(pattern_dest_test_boat  + "img" + to_string(num_samples_test_boat) + ".jpg", single_img_gray);
//        num_samples_test_boat += 1;
//    }
    
    
    
    
    
     //Training images
//    String finaldataset_training_path = "/Users/gioel/Documents/Control System Engineering/Computer Vision/Final Project/data/FINAL_DATASET/TRAINING_DATASET/IMAGES";
//    String dest_path_training = "/Users/gioel/Documents/Control System Engineering/Computer Vision/Final Project/data/FINAL_DATASET/Training_300_300/";
//
//    int num_training_samples = 0;
//    String finaldataset_training = finaldataset_training_path + "/*.png";
//    vector<String> fn;
//    glob(finaldataset_training, fn, false);
//
//    int count = fn.size(); // Number of training images
//    for (int i = 0; i < count; i++) {
//
//        Mat single_img_gray = imread(fn[i], IMREAD_COLOR);
//        resize(single_img_gray, single_img_gray, Size(300, 300));
//        //single_img_gray.convertTo(single_img_gray, CV_32F, 1.0 / 255, 0);
//        imwrite(dest_path_training  + "img" + to_string(num_training_samples) + ".png", single_img_gray);
//        num_training_samples ++;
//    }
    
    
    String pattern_venice = "/Users/gioel/Documents/Control System Engineering/Computer Vision/Final Project/data/venice_dataset";
//
    String pattern_kaggle = "/Users/gioel/Documents/Control System Engineering/Computer Vision/Final Project/data/Kaggle_ships";
    
    String cut_dataset_venice = "/Users/gioel/Desktop/Cut\ images/Kaggle_ships";
//
//    String p = "/Users/gioel/Desktop/cartella\ senza\ nome/02.png";
//
//    Mat i = imread(p, IMREAD_COLOR);
//    cout<<i<<endl;
//
    Detection boat(pattern_kaggle);
//    boat.preprocessing(pattern_venice);
//    int stepSize = 50;
//    int windowSize = 200;
//    double scale = 1.5;
//
//    Mat image = boat.test_images.at(1);
//    //boat.pyramid(scale, image, stepSize, windowSize);
//    boat.HOG(image);
    
    String model_path_pb = "/Users/gioel/Documents/Control\ System\ Engineering/Computer\ Vision/Final\ Project/Model/boat_class.pb";
    
    String model_path_pbtxt = "/Users/gioel/Documents/Control\ System\ Engineering/Computer\ Vision/Final\ Project/Model/boat_class.pbtxt";
    
    String provaa = "/Users/gioel/Desktop/Cut\ images/blue-boat-freedom-horizon-ocean-2878.jpg";
    
    dnn::Net net = dnn::readNetFromTensorflow(model_path_pb);

//    Mat im = imread(provaa, IMREAD_COLOR);
//    resize(im, im, Size(300,300));
//    Mat imm = cv::dnn::blobFromImage(im);
//    net.setInput(imm);
//    Mat res;
//    net.forward(res);
//    cout<<"DDDD"<<res<<endl;
//    imshow("D", im);
//    waitKey(0);
    
    for (int i=0;i<boat.test_images.size();i++) {
        
            //Mat prova(Size(300,300), CV_32FC1);
        Mat prova;
        resize(boat.test_images.at(i), prova, Size(300,300));
        Mat a = cv::dnn::blobFromImage(prova);
        net.setInput(a);
        Mat res;
        net.forward(res);
       // cout<<res.size()<<endl;
        if (res.at<float>(0) < 0.5) {

            imshow("im", boat.test_images.at(i));
            waitKey(0);
        }
        cout<<round(res.at<float>(0))<<endl;
    }

    //boat.Kmeans_segmentation(3);
    //boat.otsu_segmentation();
    //boat.showAndCompute_sift();
    int stepSize = 50;
    int windowSize = 200;
    double scale = 1.5;

    boat.pyramid(scale, boat.test_images.at(1), stepSize, windowSize);
    
    
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

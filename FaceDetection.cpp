#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/objdetect.hpp>
#include<iostream>

using namespace std;
using namespace cv;

/*void main() {
	string path = "hero.jpeg";
	Mat	img = imread(path);
	imshow("Frame", img);
	waitKey(0);
}*/


#include <opencv2/opencv.hpp>

int main() {
    // Load pre-trained face detection model
    CascadeClassifier faceCascade;
    if (!faceCascade.load(cv::samples::findFile("haarcascades/haarcascade_frontalface_alt.xml"))) {
        std::cerr << "Error: Could not load face cascade." << std::endl;
        return -1;
    }

    // Open the default camera (usually the first webcam)
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Failed to open camera." << std::endl;
        return -1;
    }

    // Create a window to display the camera feed
    namedWindow("Face Detection", cv::WINDOW_NORMAL);

    // Main loop
    while (true) {
        Mat frame;
        cap >> frame; // Capture frame from camera

        // Convert frame to grayscale for face detection
        Mat gray;
        cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // Detect faces in the frame
        vector<cv::Rect> faces; 
        faceCascade.detectMultiScale(gray, faces, 1.1, 2, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));

        // Draw rectangles around detected faces and count them
        int num_faces = 0;
        for (const auto& face : faces) {
            cv::rectangle(frame, face, cv::Scalar(0, 255, 0), 2);
            num_faces++;
        }

        // Display the frame with detected faces and count
        putText(frame, "Number of faces: " + std::to_string(num_faces), cv::Point(20, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
        imshow("Face Detection", frame);

        // Check for key press to exit
        char key = waitKey(1);
        if (key == 27) // ASCII code for 'ESC'
            break;
    }

    // Release resources
    cap.release();
    destroyAllWindows();

    return 0;
}

#include "ROS_interface.h"
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>

extern "C" void demo_yolo(char *cfgfile, char *weightfile, float thresh, int cam_index,
				char *filename);
extern "C" void load_network(char *cfgfile, char *weightfile, float thresh, int cam_index);

using namespace std;
using namespace cv;
using namespace cv::gpu;

cv::Mat cv_ptr_copy;
//char *cfg = "/media/ubuntu/darknet/cfg/yolo-tiny_faces_9class.cfg";
char *cfg = "/media/ubuntu/darknet/cfg/yolo-tiny.cfg";
//char *cfg = "/media/ubuntu/darknet/cfg/yolo-small.cfg";
//char *cfg = "/media/ubuntu/darknet/cfg/yolo-tiny_heads.cfg";

char *weights = "/media/ubuntu/darknet/weights/yolo-tiny.weights";
//char *weights = "/media/ubuntu/darknet/weights/yolo-tiny_faces_3000.weights";
//char *weights = "/media/ubuntu/darknet/weights/yolo-small.weights";
//char *weights = "/media/ubuntu/darknet/face_detector_weights/yolo-tiny_heads_v2_2000.weights";

float thresh = 0.2;
int cam_index = 0;
char *filename = "/media/ubuntu/darknet/data/person.jpg";

//IplImage* get_Ipl_image()
//{
//  IplImage* ROS_img = new IplImage(cv_ptr_copy);
//  return ROS_img;
//}

Mat& get_Mat_image()
{
   return cv_ptr_copy;
}

void callback(const sensor_msgs::ImageConstPtr& msg)
{
cout << "usb image received" << endl;
  cv_bridge::CvImagePtr cv_ptr;
  try {
	cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  }
  catch (cv_bridge::Exception& e) {
	ROS_ERROR("cv_bridge exception: %s", e.what());
	return;
  }

  if (cv_ptr) {
	cv_ptr_copy = cv_ptr->image.clone();
        demo_yolo(cfg, weights, thresh, cam_index, filename);
  }
  return;
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "ROS_interface");

  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);
  image_transport::Subscriber image_sub;

  load_network(cfg, weights, thresh, cam_index);

  image_sub = it.subscribe("/usb_cam/image_raw", 1, callback);
  ros::spin();
  return 0;
}

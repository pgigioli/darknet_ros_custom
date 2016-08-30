#include "ROS_interface.h"
#include <geometry_msgs/Point.h>
#include <vector>
#include <iostream>
#include <pthread.h>
#include <face_tracker/rect.h>
#include <face_tracker/templMatch.h>
#include <face_tracker/imageArray.h>
#include <std_msgs/Int8.h>

extern "C" {
  #include "box.h"
}

// initialize YOLO functions that are called in this script
//extern "C" ROS_box *demo_yolo(char *cfgfile, char *weightfile, float thresh, int cam_index,
//				char *filename);
extern "C" ROS_box *demo_yolo();
extern "C" void load_network(char *cfgfile, char *weightfile, float thresh, int cam_index);

using namespace std;
using namespace cv;
using namespace cv::gpu;

// define demo_yolo inputs. cam_index and filename are placeholders and have arbitrary values
char *cfg = "/media/ubuntu/darknet/cfg/yolo-tiny_face_depth.cfg";
char *weights = "/media/ubuntu/darknet/face_depth_weights/yolo-tiny_face_depth_7000.weights";
float thresh = 0.1;
int cam_index = 0;
char *filename = "/media/ubuntu/darknet/data/person.jpg";
Mat cv_ptr_copy;
static ROS_box *boxes;

// define parameters
static const std::string OPENCV_WINDOW = "YOLO person detection";
int FRAME_W;
int FRAME_H;
int FRAME_AREA;
float TEMPL_SCALE = 0.5; //0.7
float ROI_SCALE = 3.0; //3.0
int FRAME_COUNT = 0;

// define a function that will replace CvVideoCapture.
// This function is called in yolo_kernels and allows YOLO to receive the ROS image
// message as an IplImage
//IplImage* get_Ipl_image()
//{
//   IplImage* ROS_img = new IplImage(cv_ptr_copy);
//   return ROS_img;
//}

Mat& get_Mat_image()
{
   return cv_ptr_copy;
}

class yoloPersonDetector
{
   ros::NodeHandle nh;

   image_transport::ImageTransport it;
   image_transport::Subscriber image_sub;
   ros::Publisher ROI_coordinate_pub;
   ros::Publisher template_match_pub;
   ros::Publisher found_object_pub;
   ros::Publisher cropped_faces_pub;

public:
   yoloPersonDetector() : it(nh)
   {
      image_sub = it.subscribe("/usb_cam/image_raw", 1,
	                       &yoloPersonDetector::callback,this);
      ROI_coordinate_pub = nh.advertise<geometry_msgs::Point>("ROI_coordinate", 1);
      template_match_pub = nh.advertise<face_tracker::templMatch>("YOLO_templates", 1);
      found_object_pub = nh.advertise<std_msgs::Int8>("found_object", 1);
      cropped_faces_pub = nh.advertise<face_tracker::imageArray>("cropped_faces", 1);

      namedWindow(OPENCV_WINDOW, WINDOW_NORMAL);
   }

   ~yoloPersonDetector()
   {
      destroyWindow(OPENCV_WINDOW);
   }

private:
   void publishCroppedFaces(vector<ROS_box> face_boxes, int num_faces)
   {
      face_tracker::imageArray img_msg_array;

      for (int i = 0; i < num_faces; i++) {
         // extract coordinates of face bboxes
         Point topLeftCorner = Point((face_boxes[i].x - face_boxes[i].w/2)*FRAME_W,
                                     (face_boxes[i].y - face_boxes[i].h/2)*FRAME_H);
         Point botRightCorner = Point((face_boxes[i].x + face_boxes[i].w/2)*FRAME_W,
                                      (face_boxes[i].y + face_boxes[i].h/2)*FRAME_H);
         Rect croppedFace = Rect(topLeftCorner.x, topLeftCorner.y,
                                 face_boxes[i].w*FRAME_W, face_boxes[i].h*FRAME_H);
         Mat cropped_frame = cv_ptr_copy.clone()(croppedFace);

    	 // convert cropped faces to image messages and publish
         cv_bridge::CvImage img_bridge;
         sensor_msgs::Image img_msg;
         std_msgs::Header header;
         header.seq = i;
         header.stamp = ros::Time::now();
         img_bridge = cv_bridge::CvImage(header, sensor_msgs::image_encodings::RGB8,
		                         cropped_frame);
         img_bridge.toImageMsg(img_msg);
   	 img_msg_array.images.push_back(img_msg);
      }
      cropped_faces_pub.publish(img_msg_array);
   }

   Rect getCroppedDimensions(ROS_box boxes, float scale)
   {
      int xmin = (boxes.x - scale*boxes.w/2)*FRAME_W;
      int ymin = (boxes.y - scale*boxes.h/2)*FRAME_H;
      int xmax = (boxes.x + scale*boxes.w/2)*FRAME_W;
      int ymax = (boxes.y + scale*boxes.h/2)*FRAME_H;

      if (xmin < 0) xmin = 0;
      if (ymin < 0) ymin = 0;
      if (xmax > FRAME_W) xmax = FRAME_W;
      if (ymax > FRAME_H) ymax = FRAME_H;

      Rect cropDim = Rect(xmin, ymin, (xmax - xmin), (ymax - ymin));
      return cropDim;
   }

   void getTemplates(ROS_box *boxes, Mat input_frame, int num)
   {
      Mat full_frame = input_frame.clone();

      // create message with templates and search ROI coordinates
      face_tracker::templMatch templ_match_msg;
      templ_match_msg.scale = TEMPL_SCALE;
      templ_match_msg.num = num;

      for (int i = 0; i < num; i++)
      {
         Rect template_size = getCroppedDimensions(boxes[i], TEMPL_SCALE);
	 Rect searchROI_rect = getCroppedDimensions(boxes[i], ROI_SCALE);

         // add template and search ROI coordinates to template matching message
         cv_bridge::CvImage templ_bridge;
         sensor_msgs::Image templ_msg;
         std_msgs::Header templ_header;
         templ_header.seq = i;
         templ_header.stamp = ros::Time::now();
         templ_bridge = cv_bridge::CvImage(templ_header, sensor_msgs::image_encodings::BGR8,
                                           full_frame(template_size));
         templ_bridge.toImageMsg(templ_msg);
         templ_match_msg.templates.push_back(templ_msg);

         templ_match_msg.classes.push_back(boxes[i].Class);

         face_tracker::rect ROI_msg;
         ROI_msg.x = searchROI_rect.x;
         ROI_msg.y = searchROI_rect.y;
         ROI_msg.w = searchROI_rect.width;
         ROI_msg.h = searchROI_rect.height;
         templ_match_msg.ROIcoords.push_back(ROI_msg);

         stringstream ss;
         ss << "template " << i;
         string window_name = ss.str();
         //imshow(window_name, templates[i]);
         //waitKey(3);

         stringstream ss2;
         ss2 << "search ROI " << i;
         string window_name2 = ss2.str();
         //imshow(window_name2, full_frame(searchROIcoordinates[i]));
         //waitKey(3);
      }

      // send array of templates and array of ROI coordinates to
      // template matching function
      template_match_pub.publish(templ_match_msg);
      return;
   }

   Mat drawBBoxes(Mat input_frame, vector<ROS_box> object_boxes, int num)
   {
      int largest_area = 0;
      geometry_msgs::Point focal_point;

      for (int i = 0; i < num; i++) {
         // define the focal point and area of found object
         focal_point.x = object_boxes[i].x*FRAME_W;
	 focal_point.y = object_boxes[i].y*FRAME_H;
	 focal_point.z = 0.0;

         // draw bounding box of first object found
         Point topLeftCorner = Point((object_boxes[i].x - object_boxes[i].w/2)*FRAME_W,
				     (object_boxes[i].y - object_boxes[i].h/2)*FRAME_H);
         Point botRightCorner = Point((object_boxes[i].x + object_boxes[i].w/2)*FRAME_W,
				      (object_boxes[i].y + object_boxes[i].h/2)*FRAME_H);
         switch (object_boxes[i].Class) {
            case 0:
	       rectangle(input_frame, topLeftCorner, botRightCorner, Scalar(255,255,0), 2);
	       break;
	    case 1:
	       rectangle(input_frame, topLeftCorner, botRightCorner, Scalar(0,255,255), 2);
	       break;
	    case 2:
	       rectangle(input_frame, topLeftCorner, botRightCorner, Scalar(255,0,255), 2);
	       break;
	    case 3:
	       rectangle(input_frame, topLeftCorner, botRightCorner, Scalar(155,155,0), 2);
	       break;
         }

         // check for the largest box
         if (object_boxes[i].h*object_boxes[i].w*FRAME_AREA > largest_area + largest_area*0.1) {
            largest_area = object_boxes[i].h*object_boxes[i].w*FRAME_AREA;
            focal_point.x = object_boxes[i].x*FRAME_W;
            focal_point.y = object_boxes[i].y*FRAME_H;
         }
      }

      // draw focal point on object of interest
      Point face_center(focal_point.x, focal_point.y);
      circle(input_frame, face_center, 2, Scalar(255,0,255), 2, 8, 0);

      // only publish focal point for faces. Change to == 1 to publish person focal point
      if (object_boxes[0].Class == 0) ROI_coordinate_pub.publish(focal_point);
      return input_frame;
   }

   void runYOLO(Mat full_frame)
   {
      Mat input_frame = full_frame.clone();

      // run yolo and get bounding boxes for objects
      //ROS_box *boxes = demo_yolo(cfg, weights, thresh, cam_index, filename);
      boxes = demo_yolo();

      // get the number of bounding boxes found
      int num = boxes[0].num;

      // if at least one bbox found, define center point and draw box
      if (num > 0  && num <= 50) {
         vector<ROS_box> boxes_100cm;
         vector<ROS_box> boxes_200cm;
	 vector<ROS_box> boxes_300cm;
	 vector<ROS_box> boxes_400cm;
         int num_100cm = 0;
         int num_200cm = 0;
	 int num_300cm = 0;
	 int num_400cm = 0;
	 cout << "# Objects: " << num << endl;

	 // split bounding boxes by class
         for (int i = 0; i < num; i++) {
            switch (boxes[i].Class) {
               case 0:
                  boxes_100cm.push_back(boxes[i]);
                  num_100cm++;
                  break;
               case 1:
                  boxes_200cm.push_back(boxes[i]);
                  num_200cm++;
                  break;
	       case 2:
		  boxes_300cm.push_back(boxes[i]);
		  num_300cm++;
		  break;
	       case 3:
		  boxes_400cm.push_back(boxes[i]);
		  num_400cm++;
		  break;
            }
         }

         //publishCroppedFaces(face_boxes, num_faces);

	 // send message that an object has been detected
         std_msgs::Int8 msg;
         msg.data = 1;
         found_object_pub.publish(msg);

	 // get templates and search ROI rectangles for template matching
	 getTemplates(boxes, input_frame, num);

         if (num_100cm > 0) input_frame = drawBBoxes(input_frame, boxes_100cm, num_100cm);
         if (num_200cm > 0) input_frame = drawBBoxes(input_frame, boxes_200cm, num_200cm);
	 if (num_300cm > 0) input_frame = drawBBoxes(input_frame, boxes_300cm, num_300cm);
	 if (num_400cm > 0) input_frame = drawBBoxes(input_frame, boxes_400cm, num_400cm);
      } else {
          std_msgs::Int8 msg;
          msg.data = 0;
          found_object_pub.publish(msg);
      }

      imshow(OPENCV_WINDOW, input_frame);
      cv::waitKey(3);
   }

   void callback(const sensor_msgs::ImageConstPtr& msg)
   {
      cout << "usb image received" << endl;

      cv_bridge::CvImagePtr cv_ptr;

      try {
         cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
      } catch (cv_bridge::Exception& e) {
         ROS_ERROR("cv_bridge exception: %s", e.what());
	 return;
      }

      if (cv_ptr) {
         cv_ptr_copy = cv_ptr->image.clone();

	 if (FRAME_COUNT == 0) {
            runYOLO(cv_ptr->image);
         }
	 //FRAME_COUNT++;
	 if (FRAME_COUNT == 1) FRAME_COUNT = 0;
      }
      return;
   }
};

int main(int argc, char** argv)
{
   ros::init(argc, argv, "ROS_interface");

   ros::param::get("/usb_cam/image_width", FRAME_W);
   ros::param::get("/usb_cam/image_height", FRAME_H);
   FRAME_AREA = FRAME_W * FRAME_H;

   load_network(cfg, weights, thresh, cam_index);

   yoloPersonDetector ypd;
   ros::spin();
   return 0;
}

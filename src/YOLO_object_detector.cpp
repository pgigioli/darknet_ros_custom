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
char *cfg = "/media/ubuntu/darknet/cfg/yolo-tiny.cfg";
char *weights = "/media/ubuntu/darknet/weights/yolo-tiny.weights";
float thresh = 0.3;
int cam_index = 0;
char *filename = "/media/ubuntu/darknet/data/person.jpg";
Mat cv_ptr_copy;
static ROS_box *boxes;

// define parameters
static const std::string OPENCV_WINDOW = "YOLO object detection";
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

class yoloObjectDetector
{
   ros::NodeHandle nh;

   image_transport::ImageTransport it;
   image_transport::Subscriber image_sub;
   ros::Publisher ROI_coordinate_pub;
   ros::Publisher template_match_pub;
   ros::Publisher found_object_pub;
   ros::Publisher cropped_faces_pub;

public:
   yoloObjectDetector() : it(nh)
   {
      image_sub = it.subscribe("/usb_cam/image_raw", 1,
	                       &yoloObjectDetector::callback,this);
      ROI_coordinate_pub = nh.advertise<geometry_msgs::Point>("ROI_coordinate", 1);
      template_match_pub = nh.advertise<face_tracker::templMatch>("YOLO_templates", 1);
      found_object_pub = nh.advertise<std_msgs::Int8>("found_object", 1);
      cropped_faces_pub = nh.advertise<face_tracker::imageArray>("cropped_faces", 1);

      namedWindow(OPENCV_WINDOW, WINDOW_NORMAL);
   }

   ~yoloObjectDetector()
   {
      destroyWindow(OPENCV_WINDOW);
   }

private:
   /*void publishCroppedFaces(vector<ROS_box> face_boxes, int num_faces)
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
   }*/

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

   Mat drawBBoxes(Mat input_frame, vector<ROS_box> object_boxes, int num, Scalar color,
		  string label)
   {
      int largest_area = 0;
      geometry_msgs::Point focal_point;

      for (int i = 0; i < num; i++) {
	 int xmin = (object_boxes[i].x - object_boxes[i].w/2)*FRAME_W;
	 int ymin = (object_boxes[i].y - object_boxes[i].h/2)*FRAME_H;
	 int xmax = (object_boxes[i].x + object_boxes[i].w/2)*FRAME_W;
	 int ymax = (object_boxes[i].y + object_boxes[i].h/2)*FRAME_H;
	 int width = object_boxes[i].w*FRAME_W;
	 int height = object_boxes[i].h*FRAME_H;

         // draw bounding box of first object found
         Point topLeftCorner = Point(xmin, ymin);
         Point botRightCorner = Point(xmax, ymax);
	 rectangle(input_frame, topLeftCorner, botRightCorner, color, 2);
         putText(input_frame, label, Point(xmin, ymax+15), FONT_HERSHEY_PLAIN,
		 1.0, color, 2.0);

         // check for the largest box
         if (object_boxes[0].Class == 14 && width*height*FRAME_AREA > largest_area + largest_area*0.1) {
            largest_area = width*height*FRAME_AREA;
            focal_point.x = object_boxes[i].x*FRAME_W;
            focal_point.y = object_boxes[i].y*FRAME_H;
	    focal_point.z = 0.0;
         }
      }

      // draw focal point only on people and publish
      if (object_boxes[0].Class == 14) {
         Point object_center(focal_point.x, focal_point.y);
         circle(input_frame, object_center, 2, Scalar(255,0,255), 2, 8, 0);
	 ROI_coordinate_pub.publish(focal_point);

         // send message that an object has been detected
         std_msgs::Int8 msg;
         msg.data = 1;
         found_object_pub.publish(msg);
      }
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
         vector<ROS_box> aeroplane_boxes;   int num_aeroplanes = 0;
         vector<ROS_box> bicycle_boxes;     int num_bicycles = 0;
	 vector<ROS_box> bird_boxes; 	    int num_birds = 0;
	 vector<ROS_box> boat_boxes;        int num_boats = 0;
	 vector<ROS_box> bottle_boxes; 	    int num_bottles = 0;
	 vector<ROS_box> bus_boxes; 	    int num_buses = 0;
	 vector<ROS_box> car_boxes; 	    int num_cars = 0;
	 vector<ROS_box> cat_boxes; 	    int num_cats = 0;
	 vector<ROS_box> chair_boxes;       int num_chairs = 0;
	 vector<ROS_box> cow_boxes;	    int num_cows = 0;
	 vector<ROS_box> diningtable_boxes; int num_diningtables = 0;
	 vector<ROS_box> dog_boxes;	    int num_dogs = 0;
	 vector<ROS_box> horse_boxes;	    int num_horses = 0;
	 vector<ROS_box> motorbike_boxes;   int num_motorbikes = 0;
	 vector<ROS_box> person_boxes;      int num_persons = 0;
	 vector<ROS_box> pottedplant_boxes; int num_pottedplants = 0;
	 vector<ROS_box> sheep_boxes;	    int num_sheeps = 0;
	 vector<ROS_box> sofa_boxes;        int num_sofas = 0;
	 vector<ROS_box> train_boxes;       int num_trains = 0;
	 vector<ROS_box> tvmonitor_boxes;   int num_tvmonitors = 0;

	 cout << "# Objects: " << num << endl;

	 // split bounding boxes by class
         for (int i = 0; i < num; i++) {
            switch (boxes[i].Class) {
               case 0:
                  aeroplane_boxes.push_back(boxes[i]);
                  num_aeroplanes++;
                  break;
               case 1:
                  bicycle_boxes.push_back(boxes[i]);
                  num_bicycles++;
                  break;
	       case 2:
		  bird_boxes.push_back(boxes[i]);
		  num_birds++;
		  break;
	       case 3:
		  boat_boxes.push_back(boxes[i]);
		  num_boats++;
		  break;
               case 4:
                  bottle_boxes.push_back(boxes[i]);
                  num_bottles++;
                  break;
               case 5:
                  bus_boxes.push_back(boxes[i]);
                  num_buses++;
                  break;
               case 6:
                  car_boxes.push_back(boxes[i]);
                  num_cars++;
                  break;
               case 7:
                  cat_boxes.push_back(boxes[i]);
                  num_cats++;
                  break;
               case 8:
                  chair_boxes.push_back(boxes[i]);
                  num_chairs++;
                  break;
               case 9:
                  cow_boxes.push_back(boxes[i]);
                  num_cows++;
                  break;
               case 10:
                  diningtable_boxes.push_back(boxes[i]);
                  num_diningtables++;
                  break;
               case 11:
                  dog_boxes.push_back(boxes[i]);
                  num_dogs++;
                  break;
               case 12:
                  horse_boxes.push_back(boxes[i]);
                  num_horses++;
                  break;
               case 13:
                  motorbike_boxes.push_back(boxes[i]);
                  num_motorbikes++;
                  break;
               case 14:
                  person_boxes.push_back(boxes[i]);
                  num_persons++;
                  break;
               case 15:
                  pottedplant_boxes.push_back(boxes[i]);
                  num_pottedplants++;
                  break;
               case 16:
                  sheep_boxes.push_back(boxes[i]);
                  num_sheeps++;
                  break;
               case 17:
                  sofa_boxes.push_back(boxes[i]);
                  num_sofas++;
                  break;
               case 18:
                  train_boxes.push_back(boxes[i]);
                  num_trains++;
                  break;
               case 19:
                  tvmonitor_boxes.push_back(boxes[i]);
                  num_tvmonitors++;
                  break;
            }
         }

         //publishCroppedFaces(face_boxes, num_faces);

	 // send message that an object has been detected
//         std_msgs::Int8 msg;
//         msg.data = 1;
//         found_object_pub.publish(msg);

	 // get templates and search ROI rectangles for template matching
	 getTemplates(boxes, input_frame, num);

         if (num_aeroplanes > 0) input_frame = drawBBoxes(input_frame, aeroplane_boxes, num_aeroplanes,
							  Scalar(255,255,0), "aeroplane");
         if (num_bicycles > 0) input_frame = drawBBoxes(input_frame, bicycle_boxes, num_bicycles,
							Scalar(200,255,0), "bicycle");
         if (num_birds > 0) input_frame = drawBBoxes(input_frame, bird_boxes, num_birds,
						     Scalar(150,255,0), "bird");
         if (num_boats > 0) input_frame = drawBBoxes(input_frame, boat_boxes, num_boats,
						     Scalar(100,255,0), "boat");
         if (num_bottles > 0) input_frame = drawBBoxes(input_frame, bottle_boxes, num_bottles,
						     Scalar(50,255,0), "bottle");
         if (num_buses > 0) input_frame = drawBBoxes(input_frame, bus_boxes, num_buses,
						     Scalar(0,255,0), "bus");
         if (num_cars > 0) input_frame = drawBBoxes(input_frame, car_boxes, num_cars,
						    Scalar(0,255,50), "car");
         if (num_cats > 0) input_frame = drawBBoxes(input_frame, cat_boxes, num_cats,
						    Scalar(0,255,100), "cat");
	 if (num_chairs > 0) input_frame = drawBBoxes(input_frame, chair_boxes, num_chairs,
						      Scalar(0,255,150), "chair");
         if (num_cows > 0) input_frame = drawBBoxes(input_frame, cow_boxes, num_cows,
						    Scalar(0,255,200), "cow");
         if (num_diningtables > 0) input_frame = drawBBoxes(input_frame, diningtable_boxes, num_diningtables,
							    Scalar(0,255,255), "dining table");
         if (num_dogs > 0) input_frame = drawBBoxes(input_frame, dog_boxes, num_dogs,
						    Scalar(0,200,255), "dog");
         if (num_horses > 0) input_frame = drawBBoxes(input_frame, horse_boxes, num_horses,
						      Scalar(0,150,255), "horse");
         if (num_motorbikes > 0) input_frame = drawBBoxes(input_frame, motorbike_boxes, num_motorbikes,
							  Scalar(0,100,255), "motorbike");
         if (num_persons > 0) input_frame = drawBBoxes(input_frame, person_boxes, num_persons,
						        Scalar(0,50,255), "person");
         if (num_pottedplants > 0) input_frame = drawBBoxes(input_frame, pottedplant_boxes, num_pottedplants,
							    Scalar(0,0,255), "potted plant");
         if (num_sheeps > 0) input_frame = drawBBoxes(input_frame, sheep_boxes, num_sheeps,
						      Scalar(50,0,255), "sheep");
         if (num_sofas > 0) input_frame = drawBBoxes(input_frame, sofa_boxes, num_sofas,
						     Scalar(100,0,255), "sofa");
         if (num_trains > 0) input_frame = drawBBoxes(input_frame, train_boxes, num_trains,
						      Scalar(150,0,255), "train");
         if (num_tvmonitors > 0) input_frame = drawBBoxes(input_frame, tvmonitor_boxes, num_tvmonitors,
							  Scalar(200,0,255), "tv monitor");
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

   yoloObjectDetector yod;
   ros::spin();
   return 0;
}

cmake_minimum_required(VERSION 2.8.12)
project(darknet_ros_custom)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  face_tracker
  cv_bridge
  roscpp
  rospy
  std_msgs
  image_transport
  message_generation
)

add_message_files(
   FILES
   bboxes.msg
   bbox.msg
   bbox_array.msg
)

generate_messages(
   DEPENDENCIES
   std_msgs
)

set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 -gencode arch=compute_20,code=sm_20 
        -gencode arch=compute_20,code=sm_21
        -gencode arch=compute_30,code=sm_30
        -gencode arch=compute_35,code=sm_35
        -gencode arch=compute_50,code=sm_50
        -gencode arch=compute_52,code=sm_52
#        -gencode arch=compute_53,code=sm_53
    )

add_definitions(-DGPU -DOPENCV)

catkin_package()

include_directories(
  /usr/local/cuda/include
  src
  ${catkin_INCLUDE_DIRS}
)

link_directories(
  /usr/local/cuda/lib
)

cuda_add_executable(yolo_face_detector
  src/YOLO_face_detector.cpp

  src/gemm.c                    src/utils.c
  src/cuda.c                    src/deconvolutional_layer.c
  src/convolutional_layer.c     src/list.c
  src/image.c                   src/activations.c
  src/im2col.c                  src/col2im.c
  src/blas.c                    src/crop_layer.c
  src/dropout_layer.c           src/maxpool_layer.c
  src/softmax_layer.c           src/data.c
  src/matrix.c                  src/network.c
  src/connected_layer.c         src/cost_layer.c
  src/parser.c                  src/option_list.c
  src/detection_layer.c         src/tag.c
  src/imagenet.c                src/captcha.c
  src/route_layer.c             src/writing.c
  src/box.c                     src/nightmare.c
  src/normalization_layer.c     src/avgpool_layer.c
  src/yolo_face_detector.c      src/layer.c
  src/compare.c                 src/classifier.c
  src/local_layer.c             src/shortcut_layer.c
  src/activation_layer.c        src/cifar.c
  src/crnn_layer.c              src/go.c
  src/rnn.c                     src/rnn_layer.c
  src/rnn_vid.c                  

  src/convolutional_kernels.cu  src/deconvolutional_kernels.cu
  src/activation_kernels.cu     src/im2col_kernels.cu
  src/col2im_kernels.cu         src/blas_kernels.cu
  src/crop_layer_kernels.cu     src/dropout_layer_kernels.cu
  src/maxpool_layer_kernels.cu  src/softmax_layer_kernels.cu
  src/network_kernels.cu        src/avgpool_layer_kernels.cu
  src/yolo_kernels_ROSobj_detector.cu)

cuda_add_executable(yolo_person_detector
  src/YOLO_person_detector.cpp

  src/gemm.c                    src/utils.c
  src/cuda.c                    src/deconvolutional_layer.c
  src/convolutional_layer.c     src/list.c
  src/image.c                   src/activations.c
  src/im2col.c                  src/col2im.c
  src/blas.c                    src/crop_layer.c
  src/dropout_layer.c           src/maxpool_layer.c
  src/softmax_layer.c           src/data.c
  src/matrix.c                  src/network.c
  src/connected_layer.c         src/cost_layer.c
  src/parser.c                  src/option_list.c
  src/detection_layer.c         src/tag.c
  src/imagenet.c                src/captcha.c
  src/route_layer.c             src/writing.c
  src/box.c                     src/nightmare.c
  src/normalization_layer.c     src/avgpool_layer.c
  src/yolo_face_detector.c      src/layer.c
  src/compare.c                 src/classifier.c
  src/local_layer.c             src/shortcut_layer.c
  src/activation_layer.c        src/cifar.c
  src/crnn_layer.c              src/go.c
  src/rnn.c                     src/rnn_layer.c
  src/rnn_vid.c

  src/convolutional_kernels.cu  src/deconvolutional_kernels.cu
  src/activation_kernels.cu     src/im2col_kernels.cu
  src/col2im_kernels.cu         src/blas_kernels.cu
  src/crop_layer_kernels.cu     src/dropout_layer_kernels.cu
  src/maxpool_layer_kernels.cu  src/softmax_layer_kernels.cu
  src/network_kernels.cu        src/avgpool_layer_kernels.cu
  src/yolo_kernels_ROSobj_detector.cu)

cuda_add_executable(yolo_face_depth_detector
  src/YOLO_face_depth_detector.cpp

  src/gemm.c                    src/utils.c
  src/cuda.c                    src/deconvolutional_layer.c
  src/convolutional_layer.c     src/list.c
  src/image.c                   src/activations.c
  src/im2col.c                  src/col2im.c
  src/blas.c                    src/crop_layer.c
  src/dropout_layer.c           src/maxpool_layer.c
  src/softmax_layer.c           src/data.c
  src/matrix.c                  src/network.c
  src/connected_layer.c         src/cost_layer.c
  src/parser.c                  src/option_list.c
  src/detection_layer.c         src/tag.c
  src/imagenet.c                src/captcha.c
  src/route_layer.c             src/writing.c
  src/box.c                     src/nightmare.c
  src/normalization_layer.c     src/avgpool_layer.c
  src/yolo_face_detector.c      src/layer.c
  src/compare.c                 src/classifier.c
  src/local_layer.c             src/shortcut_layer.c
  src/activation_layer.c        src/cifar.c
  src/crnn_layer.c              src/go.c
  src/rnn.c                     src/rnn_layer.c
  src/rnn_vid.c

  src/convolutional_kernels.cu  src/deconvolutional_kernels.cu
  src/activation_kernels.cu     src/im2col_kernels.cu
  src/col2im_kernels.cu         src/blas_kernels.cu
  src/crop_layer_kernels.cu     src/dropout_layer_kernels.cu
  src/maxpool_layer_kernels.cu  src/softmax_layer_kernels.cu
  src/network_kernels.cu        src/avgpool_layer_kernels.cu
  src/yolo_kernels_ROSobj_detector.cu)

add_dependencies(yolo_face_detector face_tracker_generate_messages_cpp ${catkin_EXPORTED_TARGETS})
add_dependencies(yolo_person_detector face_tracker_generate_messages_cpp ${catkin_EXPORTED_TARGETS})
add_dependencies(yolo_face_depth_detector face_tracker_generate_messages_cpp ${catkin_EXPORTED_TARGETS})

target_link_libraries(yolo_face_detector
   m
   pthread
   stdc++
   cuda
   cudart
   cublas
   curand
   ${catkin_LIBRARIES}
)

target_link_libraries(yolo_person_detector
   m
   pthread
   stdc++
   cuda
   cudart
   cublas
   curand
   ${catkin_LIBRARIES}
)

target_link_libraries(yolo_face_depth_detector
   m
   pthread
   stdc++
   cuda
   cudart
   cublas
   curand
   ${catkin_LIBRARIES}
)

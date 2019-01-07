#pragma once
/*
https://www.learnopencv.com/deep-learning-based-object-detection-using-yolov3-with-opencv-python-c/
*/

#include <recognitionLib/Types.h>

namespace rclib
{
namespace yolo
{

//!
int RunYolo3(
	const ProcessingType processingType,
	const std::string& fileToProcess,
	const std::string& outputPath);

} // namespace yolo
} // namespace rclib

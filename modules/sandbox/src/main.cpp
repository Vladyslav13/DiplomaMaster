#include "pch.h"

#include <recognitionLib/Yolo3.h>
#include <recognitionLib/Rcnn.h>
#include <recognitionLib/SelectiveSearch.h>
#include <recognitionLib/Goturn.h>

int main()
{
	try
	{
		{
			"/image/bird.jpg";
			"/video/run.mp4";
		}

		const std::string yoloFilesRoot = ASSETS_DIR;
		const std::string outputDataDir = "C:/Mine/Diploma/Programs/MainProj/processingResults";
		const std::string inputFileName = "/video/run.mp4";

		const auto typeOfProcessing = rclib::ProcessingType::VideoProcessing;
		const auto inputData = yoloFilesRoot + inputFileName;
		const auto outputData = outputDataDir + inputFileName;

		//rclib::yolo::RunYolo3(rclib::ProcessingType::VideoProcessing, inputData, outputData);
		//rclib::rcnn::RunRcnn();
		//rclib::search::SelectiveSearch(
		//	"C:/Mine/Diploma/Programs/MainProj/assets/rcnn/cars.jpg",
		//	rclib::search::SelectiveSearchType::Quality);

		rclib::goturn::RunGoturn(inputData);

	}
	catch (const std::exception& e)
	{
		std::cout << e.what() << std::endl;
	}
	system("pause");
	return 0;
}

#include "pch.h"

#include "MainWindow.h"
#include <recognitionLib/Types.h>


void TestAlgo();



int main(int argc, char *argv[])
{
	try
	{
		QApplication a(argc, argv);
		MainWindow w;
		w.show();

		return a.exec();
	}
	catch (const std::exception& e)
	{
		std::cout << e.what() << std::endl;
	}

	system("pause");
	return 0;
}

void TestAlgo()
{
	{
		"/images/bird.jpg";
		"/video/run.mp4";
	}

	const std::string yoloFilesRoot = ASSETS_DIR;
	const std::string outputDataDir = "C:/Mine/Diploma/Programs/MainProj/processingResults";
	const std::string inputFileName = "/images/bird.jpg";

	const auto inputData = yoloFilesRoot + "/video/run.mp4";
	const auto outputData = outputDataDir + inputFileName + ".avi";

	rclib::yolo::YOLO3 yolo3;
	rclib::yolo::YOLO3::ConfigFiles cfg;
	cfg.classesNamesFile_ = yoloFilesRoot + "/yolo/coco.names";
	cfg.modelWeights_ = yoloFilesRoot + "/yolo/yolov3.weights";
	cfg.moduleCfgFile_ = yoloFilesRoot + "/yolo/yolov3.cfg";
	yolo3.SetConfigs(cfg);
	yolo3.SetFrameProcessedCallback([](auto frame) {
		imshow("Test processing", *frame);
	});

	auto res = yolo3.Process(rclib::yolo::YOLO3::DataType::CaptureFromVideoCam, inputData);

	if (res.empty())
	{
		auto val = yolo3.GetProcessedData();
		cv::VideoWriter video;
		video.open("C:\\Users\\Vladyslav\\Desktop\\vidim2.avi", cv::VideoWriter::fourcc('M','J','P','G'), 28, yolo3.GetFrameSize());
		for (auto frame : val)
		{
			video.write(*frame);
		}
		video.release();

	}

	std::cout << res << std::endl;
	//rclib::yolo::RunYolo3(typeOfProcessing, inputData, outputData);
	//rclib::rcnn::RunRcnn();
	//rclib::search::SelectiveSearch(
	//	"C:/Mine/Diploma/Programs/MainProj/assets/rcnn/cars.jpg",
	//	rclib::search::SelectiveSearchType::Quality);
}
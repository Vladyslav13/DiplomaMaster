#include "pch.h"
#include <recognitionLib/Rcnn.h>

namespace rclib
{
namespace rcnn
{

MaskRcnn::MaskRcnn()
	: frameProcessedCallback_([](auto) {})
	, isRunning_(false)
	, netConfigured_(false)
{
}

MaskRcnn::MaskRcnn(const ConfigFiles& configFiles, const Settings& settings)
	: cfg_(configFiles)
	, frameProcessedCallback_([](auto) {})
	, isRunning_(false)
	, netConfigured_(false)
	, settings_(settings)
{
}

void MaskRcnn::DrawBox(FrameData& frame, int classId, float conf, cv::Rect box, FrameData& objectMask)
{
	//Get the label for the class name and its confidence
	std::string label = cv::format("%.2f", conf);
	if (!classes_.empty())
	{
		if (classId >= classes_.size()) {
			return;
		}

		const auto className = classes_[classId];

		if (!classesToDisplay_.empty())
		{
			auto it = std::find_if(
				classesToDisplay_.begin(),
				classesToDisplay_.end(),
				[&className](const std::string& name) {
				return className == name;
			});
			if (it == classesToDisplay_.end()) {
				return;
			}
		}

		label = className + ":" + label;
	}

	if (settings_.drawRectangle_)
	{
		//Draw a rectangle displaying the bounding box
		cv::rectangle(
			*frame,
			cv::Point(box.x, box.y),
			cv::Point(box.x + box.width, box.y + box.height),
			cv::Scalar(255, 178, 50),
			3);

		//Display the label at the top of the bounding box
		int baseLine;
		const cv::Size labelSize =
			cv::getTextSize(
				label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

		box.y = std::max(box.y, labelSize.height);
		cv::rectangle(
			*frame,
			cv::Point(box.x, box.y - round(1.5*labelSize.height)),
			cv::Point(box.x + round(1.5*labelSize.width),
				box.y + baseLine),
			cv::Scalar(255, 255, 255), cv::FILLED);

		cv::putText(*frame,
			label, cv::Point(box.x, box.y),
			cv::FONT_HERSHEY_SIMPLEX,
			0.75,
			cv::Scalar(0, 0, 0),
			1);
	}

	if (objectMask)
	{
		cv::Scalar color = colors_[classId % colors_.size()];

		// Resize the mask, threshold, color and apply it on the image
		cv::resize(*objectMask, *objectMask, cv::Size(box.width, box.height));
		cv::Mat mask = (*objectMask > settings_.maskThreshold_);
		cv::Mat coloredRoi = (0.3 * color + 0.7 * (*frame)(box));
		coloredRoi.convertTo(coloredRoi, CV_8UC3);

		// Draw the contours on the image
		std::vector<cv::Mat> contours;
		cv::Mat hierarchy;
		mask.convertTo(mask, CV_8U);

		findContours(
			mask, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

		drawContours(
			coloredRoi,
			contours,
			-1,
			color,
			5,
			cv::LINE_8,
			hierarchy,
			100);

		coloredRoi.copyTo((*frame)(box), mask);
	}
}

void MaskRcnn::InitDefaultConfiguration()
{
	const std::string assetsPath = std::string{ ASSETS_DIR } +"/rcnn";

	ConfigFiles cfg;
	cfg.classesNamesFile_ = assetsPath + "/mscoco_labels.names";
	cfg.textGraph_ = assetsPath + "/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt";
	cfg.modelWeights_ = assetsPath + "/mask_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb";
	cfg.colorsFile_ = assetsPath + "/colors.txt";

	SetConfigs(cfg);

	PrepareNet();
}

std::vector<std::string> MaskRcnn::GetOutputsNames(const cv::dnn::Net& net)
{
	static std::vector<std::string> names;
	if (names.empty())
	{
		//Get the indices of the output layers, i.e. the layers with unconnected outputs
		std::vector<int> outLayers = net.getUnconnectedOutLayers();

		//get the names of all the layers in the network
		std::vector<std::string> layersNames = net.getLayerNames();

		// Get the names of the output layers in names
		names.resize(outLayers.size());
		for (size_t i = 0; i < outLayers.size(); ++i)
			names[i] = layersNames[outLayers[i] - 1];
	}
	return names;
}

void MaskRcnn::PostProcess(FrameData& frame, const std::vector<cv::Mat>& outs)
{
	cv::Mat outDetections = outs[0];
	cv::Mat outMasks = outs[1];

	// Output size of masks is NxCxHxW where
	// N - number of detected boxes
	// C - number of classes (excluding background)
	// HxW - segmentation shape
	const int numDetections = outDetections.size[2];
	const int numClasses = outMasks.size[1];

	outDetections = outDetections.reshape(1, outDetections.total() / 7);
	for (int i = 0; i < numDetections; ++i)
	{
		float score = outDetections.at<float>(i, 2);
		if (score > settings_.confThreshold_)
		{
			// Extract the bounding box
			int classId = static_cast<int>(outDetections.at<float>(i, 1));
			int left = static_cast<int>(frame->cols * outDetections.at<float>(i, 3));
			int top = static_cast<int>(frame->rows * outDetections.at<float>(i, 4));
			int right = static_cast<int>(frame->cols * outDetections.at<float>(i, 5));
			int bottom = static_cast<int>(frame->rows * outDetections.at<float>(i, 6));

			left = std::max(0, std::min(left, frame->cols - 1));
			top = std::max(0, std::min(top, frame->rows - 1));
			right = std::max(0, std::min(right, frame->cols - 1));
			bottom = std::max(0, std::min(bottom, frame->rows - 1));
			cv::Rect box = cv::Rect(left, top, right - left + 1, bottom - top + 1);

			// Extract the mask for the object
			FrameData objectMask;
			if (settings_.drawMask_)
			{
				objectMask =
					std::make_shared<cv::Mat>(outMasks.size[2], outMasks.size[3], CV_32F, outMasks.ptr<float>(i, classId));
			}

			// Draw bounding box, colorize and show the mask on the image
			DrawBox(frame, classId, score, box, objectMask);
		}
	}
}

void MaskRcnn::PrepareNet()
{
	processingRes_.clear();

	if (netConfigured_) {
		return;
	}

	std::ifstream ifs(cfg_.classesNamesFile_);
	std::string line;
	classes_.clear();
	while (getline(ifs, line)) {
		classes_.push_back(line);
	}

	std::ifstream colorFptr(cfg_.colorsFile_);
	while (getline(colorFptr, line)) {
		char* pEnd;
		double r, g, b;
		r = strtod (line.c_str(), &pEnd);
		g = strtod (pEnd, NULL);
		b = strtod (pEnd, NULL);
		cv::Scalar color = cv::Scalar(r, g, b, 255.0);
		colors_.push_back(cv::Scalar(r, g, b, 255.0));
	}

	// Load the network
	net_ =
		cv::dnn::readNetFromTensorflow(cfg_.modelWeights_, cfg_.textGraph_);
	net_.setPreferableBackend(settings_.preferableBackend_);
	net_.setPreferableTarget(settings_.preferableTarget_);

	netConfigured_ = true;
}

std::string MaskRcnn::Process(
	const DataType processingDataType,
	const std::string& fileToProcess,
	const int deviceInd /*= 0*/)
{
	// TODO: It can be false but processing loop will be in the last iteration.
	if (isRunning_) {
		return "Can't start new processing, the previous one is in progress";
	}

	// TODO: add mutex synchronization while accessing this parameter?
	isRunning_ = true;
	std::string error;

	try
	{
		switch (processingDataType)
		{
		case DataType::CaptureFromVideoCam:
			ProcessStream(deviceInd, fileToProcess); // TODO: add possibility to chose cam.
			break;
		case DataType::VideoFile:
			ProcessVideo(fileToProcess);
			break;
		case DataType::Unknown:
		default:
			throw std::runtime_error{ "Unknown data type" };
		}
	}
	catch (const std::exception& e)
	{
		error = "Error occured in MaskRcnn work: ";
		error += e.what();
	}

	Stop();

	return error;
}

void MaskRcnn::ProcessStream(
	const int deviceInd, const std::string& fileToProcess)
{
	PrepareNet();

	cv::VideoCapture cap;
	cap.open(deviceInd);

	// Add normal logging.
	std::cout << "Start processing the object" << std::endl;

	ProcessVideoImpl(cap);

	cap.release();
}

void MaskRcnn::ProcessVideo(const std::string& fileToProcess)
{
	PrepareNet();

	cv::VideoCapture cap;
	cap.open(fileToProcess);

	// Add normal logging.
	std::cout << "Start processing the object" << std::endl;

	ProcessVideoImpl(cap);

	cap.release();
}

void MaskRcnn::ProcessVideoImpl(cv::VideoCapture& cap)
{
	frameSize_ = cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT));

	while (cv::waitKey(1) < 0 && isRunning_)
	{
		auto frame = std::make_shared<cv::Mat>();
		auto blob = std::make_shared<cv::Mat>();

		// Get frame from the video
		cap >> *frame;

		// Stop the program if reached end of video
		if (frame->empty()) {
			break;
		}

		// Create a 4D blob from a frame.
		cv::dnn::blobFromImage(
			*frame,
			*blob,
			1.0,
			cv::Size(frame->cols, frame->rows),
			cv::Scalar(),
			true,
			false);

		//Sets the input to the network
		net_.setInput(*blob);

		// Runs the forward pass to get output from the output layers
		std::vector<std::string> outNames(2);
		outNames[0] = "detection_out_final";
		outNames[1] = "detection_masks";
		std::vector<cv::Mat> outs;
		net_.forward(outs, outNames);

		// Extract the bounding box and mask for each of the detected objects
		PostProcess(frame, outs);

		// Write the frame with the detection boxes
		processingRes_.push_back(frame);

		frameProcessedCallback_(frame);
	}
}

bool MaskRcnn::IsRunning() const
{
	return isRunning_;
}

bool MaskRcnn::SetFrameProcessedCallback(const FrameProcessedCallback& callback)
{
	if (!callback) {
		return false;
	}

	frameProcessedCallback_ = callback;
	return true;
}

void MaskRcnn::SetConfigs(const ConfigFiles& cfg)
{
	cfg_ = cfg;
	netConfigured_ = false;
}

void MaskRcnn::SetSettings(const Settings& settings)
{
	settings_ = settings;
	netConfigured_ = false;
}

MaskRcnn::ConfigFiles MaskRcnn::GetConfigs() const
{
	return cfg_;
}

cv::Size MaskRcnn::GetFrameSize() const
{
	return frameSize_;
}

MaskRcnn::Settings MaskRcnn::GetSettings() const
{
	return settings_;
}

std::vector<MaskRcnn::FrameData> MaskRcnn::GetProcessedData() const
{
	if (isRunning_) {
		return {};
	}

	return processingRes_;
}

void MaskRcnn::Stop()
{
	isRunning_ = false;
}


























using namespace cv;
using namespace dnn;
using namespace std;

// Initialize the parameters
float confThreshold = 0.5; // Confidence threshold
float maskThreshold = 0.3; // Mask threshold

vector<string> classes;
vector<Scalar> colors;

// Draw the predicted bounding box
void drawBox(Mat& frame, int classId, float conf, Rect box, Mat& objectMask);

// Postprocess the neural network's output for each frame
void postprocess(Mat& frame, const vector<Mat>& outs);

int RunRcnn()
{
	const std::string assetsPath = std::string{ ASSETS_DIR } +"/rcnn";
	// Load names of classes
	string classesFile = assetsPath + "/mscoco_labels.names";
	ifstream ifs(classesFile.c_str());
	string line;
	while (getline(ifs, line)) classes.push_back(line);

	// Load the colors
	string colorsFile = assetsPath + "/colors.txt";
	ifstream colorFptr(colorsFile.c_str());
	while (getline(colorFptr, line)) {
		char* pEnd;
		double r, g, b;
		r = strtod (line.c_str(), &pEnd);
		g = strtod (pEnd, NULL);
		b = strtod (pEnd, NULL);
		Scalar color = Scalar(r, g, b, 255.0);
		colors.push_back(Scalar(r, g, b, 255.0));
	}

	// Give the configuration and weight files for the model
	String textGraph = assetsPath + "/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt";
	String modelWeights = assetsPath + "/mask_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb";

	// Load the network
	Net net = readNetFromTensorflow(modelWeights, textGraph);
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_CPU);

	// Open a video file or an image file or a camera stream.
	string str, outputFile;
	VideoCapture cap;
	VideoWriter video;
	Mat frame, blob;

	//
	// Programm hardcodede defines.
	//

	enum class WorkType
	{
		Unknown = 0,
		Image,
		Video,
		CaptureFromVideoCam // Currently unsupported
	};
	const WorkType currentWorkType = WorkType::Video;
	const std::string imagePath = assetsPath + "/cars.jpg";
	const std::string videoPath = assetsPath + "/cars.mp4";

	try {

		outputFile = "mask_rcnn_out_cpp.avi";
		if (currentWorkType == WorkType::Image)
		{
			// Open the image file
			str = imagePath;
			//cout << "Image file input : " << str << endl;
			ifstream ifile(str);
			if (!ifile) throw("error");
			cap.open(str);
			str.replace(str.end()-4, str.end(), "_mask_rcnn_out.jpg");
			outputFile = str;
		}
		else if (currentWorkType == WorkType::Video)
		{
			// Open the video file
			str = videoPath;
			ifstream ifile(str);
			if (!ifile) throw("error");
			cap.open(str);
			str.replace(str.end()-4, str.end(), "_mask_rcnn_out.avi");
			outputFile = str;
		}
		// Open the webcam
		//else cap.open(parser.get<int>("device"));

	}
	catch(...) {
		cout << "Could not open the input image/video stream" << endl;
		return 0;
	}

	// Get the video writer initialized to save the output video
	if (currentWorkType != WorkType::Image) {
		video.open(outputFile, VideoWriter::fourcc('M','J','P','G'), 28, Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)));
	}

	// Create a window
	static const string kWinName = "Deep learning object detection in OpenCV";
	namedWindow(kWinName, WINDOW_NORMAL);

	// Process frames.
	while (waitKey(1) < 0)
	{
		// get frame from the video
		cap >> frame;

		// Stop the program if reached end of video
		if (frame.empty()) {
			cout << "Done processing !!!" << endl;
			cout << "Output file is stored as " << outputFile << endl;
			waitKey(3000);
			break;
		}
		// Create a 4D blob from a frame.
		blobFromImage(frame, blob, 1.0, Size(frame.cols, frame.rows), Scalar(), true, false);
		//blobFromImage(frame, blob);

		//Sets the input to the network
		net.setInput(blob);

		// Runs the forward pass to get output from the output layers
		std::vector<String> outNames(2);
		outNames[0] = "detection_out_final";
		outNames[1] = "detection_masks";
		vector<Mat> outs;
		net.forward(outs, outNames);

		// Extract the bounding box and mask for each of the detected objects
		postprocess(frame, outs);

		// Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
		vector<double> layersTimes;
		double freq = getTickFrequency() / 1000;
		double t = net.getPerfProfile(layersTimes) / freq;
		string label = format("Mask-RCNN on 2.5 GHz Intel Core i7 CPU, Inference time for a frame : %0.0f ms", t);
		putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));

		// Write the frame with the detection boxes
		Mat detectedFrame;
		frame.convertTo(detectedFrame, CV_8U);
		if (currentWorkType == WorkType::Image) imwrite(outputFile, detectedFrame);
		else video.write(detectedFrame);

		imshow(kWinName, frame);

	}

	cap.release();
	if (currentWorkType != WorkType::Image) video.release();

	return 0;
}

// For each frame, extract the bounding box and mask for each detected object
void postprocess(Mat& frame, const vector<Mat>& outs)
{
	Mat outDetections = outs[0];
	Mat outMasks = outs[1];

	// Output size of masks is NxCxHxW where
	// N - number of detected boxes
	// C - number of classes (excluding background)
	// HxW - segmentation shape
	const int numDetections = outDetections.size[2];
	const int numClasses = outMasks.size[1];

	outDetections = outDetections.reshape(1, outDetections.total() / 7);
	for (int i = 0; i < numDetections; ++i)
	{
		float score = outDetections.at<float>(i, 2);
		if (score > confThreshold)
		{
			// Extract the bounding box
			int classId = static_cast<int>(outDetections.at<float>(i, 1));
			int left = static_cast<int>(frame.cols * outDetections.at<float>(i, 3));
			int top = static_cast<int>(frame.rows * outDetections.at<float>(i, 4));
			int right = static_cast<int>(frame.cols * outDetections.at<float>(i, 5));
			int bottom = static_cast<int>(frame.rows * outDetections.at<float>(i, 6));

			left = max(0, min(left, frame.cols - 1));
			top = max(0, min(top, frame.rows - 1));
			right = max(0, min(right, frame.cols - 1));
			bottom = max(0, min(bottom, frame.rows - 1));
			Rect box = Rect(left, top, right - left + 1, bottom - top + 1);

			// Extract the mask for the object
			Mat objectMask(outMasks.size[2], outMasks.size[3],CV_32F, outMasks.ptr<float>(i,classId));

			// Draw bounding box, colorize and show the mask on the image
			drawBox(frame, classId, score, box, objectMask);

		}
	}
}

// Draw the predicted bounding box, colorize and show the mask on the image
void drawBox(Mat& frame, int classId, float conf, Rect box, Mat& objectMask)
{
	//Draw a rectangle displaying the bounding box
	cv::rectangle(frame, Point(box.x, box.y), Point(box.x+box.width, box.y+box.height), Scalar(255, 178, 50), 3);

	//Get the label for the class name and its confidence
	string label = format("%.2f", conf);
	if (!classes.empty())
	{
		CV_Assert(classId < (int)classes.size());
		label = classes[classId] + ":" + label;
	}

	//Display the label at the top of the bounding box
	int baseLine;
	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	box.y = max(box.y, labelSize.height);
	rectangle(frame, Point(box.x, box.y - round(1.5*labelSize.height)), Point(box.x + round(1.5*labelSize.width), box.y + baseLine), Scalar(255, 255, 255), FILLED);
	putText(frame, label, Point(box.x, box.y), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,0),1);

	Scalar color = colors[classId%colors.size()];

	// Resize the mask, threshold, color and apply it on the image
	resize(objectMask, objectMask, Size(box.width, box.height));
	Mat mask = (objectMask > maskThreshold);
	Mat coloredRoi = (0.3 * color + 0.7 * frame(box));
	coloredRoi.convertTo(coloredRoi, CV_8UC3);

	// Draw the contours on the image
	vector<Mat> contours;
	Mat hierarchy;
	mask.convertTo(mask, CV_8U);
	findContours(mask, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
	drawContours(coloredRoi, contours, -1, color, 5, LINE_8, hierarchy, 100);
	coloredRoi.copyTo(frame(box), mask);

}
} // namespace rcnn
} // namespace rclib

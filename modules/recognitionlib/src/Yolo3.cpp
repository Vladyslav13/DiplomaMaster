#include "pch.h"
#include <recognitionLib/Yolo3.h>

namespace rclib
{
namespace yolo
{

YOLO3::YOLO3()
	: frameProcessedCallback_([](auto) {})
	, isRunning_(false)
{
	
}

YOLO3::YOLO3(const ConfigFiles& configFiles, const Settings& settings)
	: cfg_(configFiles)
	, frameProcessedCallback_([](auto) {})
	, isRunning_(false)
	, settings_(settings)
{
}

void YOLO3::DrawPred(
	int classId, float conf, int left, int top, int right, int bottom, FrameData& frame)
{
	//Draw a rectangle displaying the bounding box
	rectangle(
		*frame,
		cv::Point(left, top),
		cv::Point(right, bottom),
		cv::Scalar(255, 178, 50),
		3);

	//Get the label for the class name and its confidence
	std::string label = cv::format("%.2f", conf);
	if (!classes_.empty())
	{
		CV_Assert(classId < static_cast<int>(classes_.size()));
		label = classes_[classId] + ":" + label;
	}

	//Display the label at the top of the bounding box
	int baseLine;
	cv::Size labelSize =
		cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	top = std::max(top, labelSize.height);
	rectangle(
		*frame,
		cv::Point(left, top - round(1.5*labelSize.height)),
		cv::Point(left + round(1.5*labelSize.width), top + baseLine),
		cv::Scalar(255, 255, 255),
		cv::FILLED);

	putText(
		*frame,
		label,
		cv::Point(left, top),
		cv::FONT_HERSHEY_SIMPLEX,
		0.75,
		cv::Scalar(0, 0, 0),
		1);
}

std::vector<std::string> YOLO3::GetOutputsNames(const cv::dnn::Net& net)
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

void YOLO3::PostProcess(FrameData& frame, const std::vector<cv::Mat>& outs)
{
	std::vector<int> classIds;
	std::vector<float> confidences;
	std::vector<cv::Rect> boxes;

	for (size_t i = 0; i < outs.size(); ++i)
	{
		// Scan through all the bounding boxes output from the network and keep only the
		// ones with high confidence scores. Assign the box's class label as the class
		// with the highest score for the box.
		float* data = (float*)outs[i].data;
		for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
		{
			cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
			cv::Point classIdPoint;
			double confidence;
			// Get the value and location of the maximum score
			minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
			if (confidence > settings_.confThreshold_)
			{
				int centerX = (int)(data[0] * frame->cols);
				int centerY = (int)(data[1] * frame->rows);
				int width = (int)(data[2] * frame->cols);
				int height = (int)(data[3] * frame->rows);
				int left = centerX - width / 2;
				int top = centerY - height / 2;

				classIds.push_back(classIdPoint.x);
				confidences.push_back((float)confidence);
				boxes.push_back(cv::Rect(left, top, width, height));
			}
		}
	}

	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	std::vector<int> indices;
	cv::dnn::NMSBoxes(
		boxes, confidences,  settings_.confThreshold_, settings_.nmsThreshold_, indices);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		cv::Rect box = boxes[idx];
		DrawPred(
			classIds[idx],
			confidences[idx],
			box.x,
			box.y,
			box.x + box.width,
			box.y + box.height, 
			frame);
	}
}

void YOLO3::PrepareNet()
{
	// TODO: Move to function.
	std::ifstream ifs(cfg_.classesNamesFile_);
	std::string line;
	classes_.clear();
	while (getline(ifs, line)) {
		classes_.push_back(line);
	}

	// Load the network
	net_ =
		cv::dnn::readNetFromDarknet(cfg_.moduleCfgFile_, cfg_.modelWeights_);
	net_.setPreferableBackend(settings_.preferableBackend_);
	net_.setPreferableTarget(settings_.preferableTarget_);

	processingRes_.clear();
}

std::string YOLO3::Process(
	const DataType processingDataType,
	const std::string& fileToProcess)
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
			ProcessStream(0, fileToProcess); // TODO: add possibility to chose cam.
			break;
		case DataType::VideoProcessing:
			ProcessVideo(fileToProcess);
			break;
		case DataType::Unknown:
		default:
			throw std::runtime_error{ "Unknown data type" };
		}
	}
	catch (const std::exception& e)
	{
		error = "Error occured in YOLO3 work: ";
		error += e.what();
	}

	Stop();

	return error;
}

void YOLO3::ProcessStream(
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

void YOLO3::ProcessVideo(const std::string& fileToProcess)
{
	PrepareNet();
	
	cv::VideoCapture cap;
	cap.open(fileToProcess);

	// Add normal logging.
	std::cout << "Start processing the object" << std::endl;

	ProcessVideoImpl(cap);

	cap.release();
}

void YOLO3::ProcessVideoImpl(cv::VideoCapture& cap)
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
			1/255.0,
			cv::Size(settings_.inpWidth_, settings_.inpHeight_),
			cv::Scalar(0,0,0),
			true,
			false);

		// Sets the input to the network
		net_.setInput(*blob);

		// Runs the forward pass to get output of the output layers
		std::vector<cv::Mat> outs;
		net_.forward(outs, GetOutputsNames(net_));

		// Remove the bounding boxes with low confidence
		PostProcess(frame, outs);

		// Write the frame with the detection boxes
		processingRes_.push_back(frame);

		frameProcessedCallback_(frame);
	}
}

bool YOLO3::IsRunning() const
{
	return isRunning_;
}

bool YOLO3::SetFrameProcessedCallback(const FrameProcessedCallback& callback)
{
	if (!callback) {
		return false;
	}

	frameProcessedCallback_ = callback;
	return true;
}

void YOLO3::SetConfigs(const ConfigFiles& cfg)
{
	cfg_ = cfg;
}

void YOLO3::SetSettings(const Settings& settings)
{
	settings_ = settings;
}

YOLO3::ConfigFiles YOLO3::GetConfigs() const
{
	return cfg_;
}

cv::Size YOLO3::GetFrameSize() const
{
	return frameSize_;
}

YOLO3::Settings YOLO3::GetSettings() const
{
	return settings_;
}

std::vector<YOLO3::FrameData> YOLO3::GetProcessedData() const
{
	if (isRunning_) {
		return {};
	}

	return processingRes_;
}

void YOLO3::Stop()
{
	isRunning_ = false;
}















using namespace cv;
using namespace dnn;
using namespace std;

// Initialize the parameters
float confThreshold = 0.5; // Confidence threshold
float nmsThreshold = 0.4;  // Non-maximum suppression threshold
int inpWidth = 320;  // Width of network's input image 416 or 320
int inpHeight = 320; // Height of network's input image
vector<string> classes;

// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat& frame, const vector<Mat>& out);

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);

// Get the names of the output layers
vector<String> getOutputsNames(const Net& net);

int RunYolo3(
	const ProcessingType processingType,
	const std::string& fileToProcess,
	const std::string& outputPath)
{
	enum class WorkType
	{
		Unknown = 0,
		Image,
		Video,
		CaptureFromVideoCam // Currently unsupported
	};

	const WorkType currentWorkType = WorkType::CaptureFromVideoCam;
	const std::string yoloFilesRoot = ASSETS_DIR;
	const std::string imagePath = yoloFilesRoot + "/images/bird.jpg";
	const std::string videoPath = yoloFilesRoot + "/video/run.mp4";

	// Give the configuration and weight files for the model
	String modelConfiguration = yoloFilesRoot + "/yolo/yolov3.cfg";
	String modelWeights = yoloFilesRoot + "/yolo/yolov3.weights";
	// Load names of classes
	string classesFile = yoloFilesRoot + "/yolo/coco.names";

	ifstream ifs(classesFile.c_str());
	string line;
	while (getline(ifs, line)) classes.push_back(line);

	// Load the network
	Net net = readNetFromDarknet(modelConfiguration, modelWeights);
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(false ? DNN_TARGET_OPENCL : DNN_TARGET_CPU);

	// Open a video file or an image file or a camera stream.
	string str, outputFile;
	VideoCapture cap;
	VideoWriter video;
	Mat frame, blob;

	try {

		outputFile = "yolo_out_cpp.avi";
		if (currentWorkType == WorkType::Image)
		{
			// Open the image file
			str = imagePath;
			ifstream ifile(str);
			if (!ifile) throw("error");
			cap.open(str);
			str.replace(str.end()-4, str.end(), "_yolo_out_cpp.jpg");
			outputFile = str;
		}
		else if (currentWorkType == WorkType::Video)
		{
			// Open the video file
			str = videoPath;
			ifstream ifile(str);
			if (!ifile) throw("error");
			cap.open(str);
			str.replace(str.end()-4, str.end(), "_yolo_out_cpp.avi");
			outputFile = str;
		}
		// Open the webcaom
		else cap.open(0);

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


	std::cout << "Start processing the object" << std::endl;
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
		blobFromImage(frame, blob, 1/255.0, cv::Size(inpWidth, inpHeight), Scalar(0,0,0), true, false);

		//Sets the input to the network
		net.setInput(blob);

		// Runs the forward pass to get output of the output layers
		vector<Mat> outs;
		net.forward(outs, getOutputsNames(net));

		// Remove the bounding boxes with low confidence
		postprocess(frame, outs);

		// Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
		vector<double> layersTimes;
		double freq = getTickFrequency() / 1000;
		double t = net.getPerfProfile(layersTimes) / freq;
		string label = format("Inference time for a frame : %.2f ms", t);
		putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
		
		// Write the frame with the detection boxes
		Mat detectedFrame;
		frame.convertTo(detectedFrame, CV_8U);
		if (currentWorkType == WorkType::Image) imwrite(outputFile, detectedFrame);
		else video.write(detectedFrame);

		imshow(kWinName, frame);
	}

	cap.release();
	if (currentWorkType != WorkType::Image) video.release();
}

// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat& frame, const vector<Mat>& outs)
{
	vector<int> classIds;
	vector<float> confidences;
	vector<Rect> boxes;

	for (size_t i = 0; i < outs.size(); ++i)
	{
		// Scan through all the bounding boxes output from the network and keep only the
		// ones with high confidence scores. Assign the box's class label as the class
		// with the highest score for the box.
		float* data = (float*)outs[i].data;
		for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
		{
			Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
			Point classIdPoint;
			double confidence;
			// Get the value and location of the maximum score
			minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
			if (confidence > confThreshold)
			{
				int centerX = (int)(data[0] * frame.cols);
				int centerY = (int)(data[1] * frame.rows);
				int width = (int)(data[2] * frame.cols);
				int height = (int)(data[3] * frame.rows);
				int left = centerX - width / 2;
				int top = centerY - height / 2;

				classIds.push_back(classIdPoint.x);
				confidences.push_back((float)confidence);
				boxes.push_back(Rect(left, top, width, height));
			}
		}
	}

	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	vector<int> indices;
	NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];
		drawPred(classIds[idx], confidences[idx], box.x, box.y,
			box.x + box.width, box.y + box.height, frame);
	}
}

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
	//Draw a rectangle displaying the bounding box
	rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 3);

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
	top = max(top, labelSize.height);
	rectangle(frame, Point(left, top - round(1.5*labelSize.height)), Point(left + round(1.5*labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
	putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 1);
}

// Get the names of the output layers
vector<String> getOutputsNames(const Net& net)
{
	static vector<String> names;
	if (names.empty())
	{
		//Get the indices of the output layers, i.e. the layers with unconnected outputs
		vector<int> outLayers = net.getUnconnectedOutLayers();

		//get the names of all the layers in the network
		vector<String> layersNames = net.getLayerNames();

		// Get the names of the output layers in names
		names.resize(outLayers.size());
		for (size_t i = 0; i < outLayers.size(); ++i)
			names[i] = layersNames[outLayers[i] - 1];
	}
	return names;
}

} // namespace yolo
} // namespace rclib
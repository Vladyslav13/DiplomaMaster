#include "pch.h"
#include "MainWindow.h"

MainWindow::MainWindow(QWidget *parent)
	: QMainWindow(parent)
	, centralWidget_(new QFrame(this))
	, graphicsView_(new QGraphicsView(centralWidget_))
	, startBtn_(new QPushButton("start", centralWidget_))
{
	setCentralWidget(centralWidget_);

	graphicsView_->setScene(new QGraphicsScene(centralWidget_));
	graphicsView_->scene()->addItem(&pixmap_);
	graphicsView_->setBackgroundBrush(QBrush(Qt::black));

	auto controlsLay = new QVBoxLayout();
	controlsLay->addWidget(startBtn_, 0, Qt::AlignTop);

	auto mainLay = new QHBoxLayout(centralWidget_);
	mainLay->addLayout(controlsLay);
	mainLay->addWidget(graphicsView_);

	connect(startBtn_, &QPushButton::pressed, this, &MainWindow::OnStartButtonPressed);
	
	qRegisterMetaType<std::shared_ptr<cv::Mat>>();
	connect(this, &MainWindow::UpdateVideoFrame, this, &MainWindow::OnUpdateVideoFrame);

	resize(900, 500);
}

MainWindow::~MainWindow()
{
	if (yolo3_.IsRunning()) {
		yolo3_.Stop();
	}
	if (workingThread_ && workingThread_->joinable()) {
		workingThread_->join();
	}
}

void MainWindow::OnUpdateVideoFrame(std::shared_ptr<cv::Mat> frame)
{
	if (frame->empty()) {
		return;
	}

	QImage qimg(frame->data,
		frame->cols,
		frame->rows,
		frame->step,
		QImage::Format_RGB888);
	pixmap_.setPixmap(QPixmap::fromImage(qimg.rgbSwapped()));
	graphicsView_->fitInView(&pixmap_, Qt::KeepAspectRatio);

	qApp->processEvents();
}

void MainWindow::OnStartButtonPressed()
{
	if (workingThread_)
	{
		if (yolo3_.IsRunning()) {
			yolo3_.Stop();
		}
		if (workingThread_->joinable()) {
			workingThread_->join();
		}
		workingThread_.reset();

		startBtn_->setText("Start");

		return;
	}
	workingThread_ = std::make_shared<std::thread>([this]() {
		const std::string yoloFilesRoot = ASSETS_DIR;
		rclib::yolo::YOLO3::ConfigFiles cfg;
		cfg.classesNamesFile_ = yoloFilesRoot + "/yolo/coco.names";
		cfg.modelWeights_ = yoloFilesRoot + "/yolo/yolov3.weights";
		cfg.moduleCfgFile_ = yoloFilesRoot + "/yolo/yolov3.cfg";
		const auto inputData = yoloFilesRoot + "/video/run.mp4";
		yolo3_.SetConfigs(cfg);
		yolo3_.SetFrameProcessedCallback([this](auto frame) {
			emit UpdateVideoFrame(frame);
		});
		yolo3_.SetClassesToDisplay({ "cell phone" });
		yolo3_.Process(rclib::yolo::YOLO3::DataType::CaptureFromVideoCam, inputData);
	});

	startBtn_->setText("Stop");
}

#include "pch.h"
#include "MainWindow.h"

#include <Utils.h>

MainWindow::MainWindow(QWidget *parent)
	: QMainWindow(parent)
	, centralWidget_(new QFrame(this))
	, currentAlgorithm_(nullptr)
	, graphicsView_(new QGraphicsView(centralWidget_))
	, startBtn_(new QPushButton("start", centralWidget_))
	, workingThread_(nullptr)
{
	setCentralWidget(centralWidget_);

	graphicsView_->setScene(new QGraphicsScene(centralWidget_));
	graphicsView_->scene()->addItem(&pixmap_);
	graphicsView_->setBackgroundBrush(QBrush(Qt::black));

	auto groupBox = new QGroupBox(tr("Current algorithm"));
	auto yoloRatio = new QRadioButton(tr("YOLO"));
	auto algoSelectLay = new QVBoxLayout;
	algoSelectLay->addWidget(yoloRatio);
	groupBox->setLayout(algoSelectLay);

	auto controlsLay = new QVBoxLayout();
	controlsLay->addWidget(groupBox, 0, Qt::AlignTop);
	controlsLay->addWidget(startBtn_, 0, Qt::AlignTop);

	auto mainLay = new QHBoxLayout(centralWidget_);
	mainLay->addLayout(controlsLay);
	mainLay->addWidget(graphicsView_);

	connect(startBtn_, &QPushButton::pressed, this, &MainWindow::OnStartButtonPressed);
	
	qRegisterMetaType<std::shared_ptr<cv::Mat>>();
	connect(
		yoloRatio,
		&QRadioButton::toggled,
		this,
		[this](bool checked) {
			UpdateCurrentAlgo(rclib::NeroAlgoTypes::Yolo);
		});

	connect(this, &MainWindow::UpdateVideoFrame, this, &MainWindow::OnUpdateVideoFrame);

	resize(900, 500);
}

MainWindow::~MainWindow()
{
	try
	{
		StopAlgorithm();
	}
	catch (const std::exception& e)
	{
		std::cerr << "Error occurred while closing main window: "
			<< e.what();
	}
}

void MainWindow::OnStartButtonPressed()
{
	if (workingThread_)
	{
		StopAlgorithm();

		workingThread_.reset();

		startBtn_->setText("Start");

		return;
	}

	if (!currentAlgorithm_)
	{
		DisplayError("Chose algorithm first", this);
		return;
	}

	workingThread_ = std::make_shared<std::thread>([this]() {

		const std::string yoloFilesRoot = ASSETS_DIR;
		const auto inputData = yoloFilesRoot + "/video/run.mp4";

		currentAlgorithm_->Process(
			rclib::yolo::YOLO3::DataType::CaptureFromVideoCam, inputData);
	});

	startBtn_->setText("Stop");
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

void MainWindow::StopAlgorithm()
{
	if (currentAlgorithm_ && currentAlgorithm_->IsRunning()) {
		currentAlgorithm_->Stop();
	}

	if (workingThread_ && workingThread_->joinable()) {
		workingThread_->join();
	}
}

void MainWindow::UpdateCurrentAlgo(const rclib::NeroAlgoTypes type)
{
	try
	{
		StopAlgorithm();

		currentAlgorithm_ = rclib::CreateNeroAlgorithm(type);

		// TODO: Configure parameters later.
		currentAlgorithm_->InitDefaultConfiguration();
		currentAlgorithm_->SetFrameProcessedCallback([this](auto frame) {
			emit UpdateVideoFrame(frame);
		});
		currentAlgorithm_->SetClassesToDisplay({ "cell phone" });
	}
	catch (const std::exception& e)
	{
		const std::string errorMsg =
			std::string{ "Error occurred while updating algorithm: " } +e.what();
		std::cerr << errorMsg;
		DisplayError(errorMsg.c_str(), this);
	}
}

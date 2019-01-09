#include "pch.h"
#include "MainWindow.h"

#include <Utils.h>

MainWindow::MainWindow(QWidget *parent)
	: QMainWindow(parent)
	, centralWidget_(new QFrame(this))
	, currentAlgorithm_(nullptr)
	, graphicsView_(new QGraphicsView(centralWidget_))
	, startBtn_(new QPushButton("Start", centralWidget_))
	, workingThread_(nullptr)
{
	setCentralWidget(centralWidget_);

	graphicsView_->setScene(new QGraphicsScene(centralWidget_));
	graphicsView_->scene()->addItem(&pixmap_);
	graphicsView_->setBackgroundBrush(QBrush(Qt::black));

	//
	// Chose algo controls.
	//

	auto choseAlgoBox = new QGroupBox(tr("Current algorithm"));
	auto yoloRatio = new QRadioButton(tr("YOLO"));
	auto rcnnRatio = new QRadioButton(tr("RCNN"));
	auto algoSelectLay = new QVBoxLayout;
	algoSelectLay->addWidget(rcnnRatio);
	algoSelectLay->addWidget(yoloRatio);
	choseAlgoBox->setLayout(algoSelectLay);

	//
	// Chose source controls.
	//

	auto choseSource = new QGroupBox(tr("Current algorithm"));
	auto webCamRatio = new QRadioButton(tr("Web camera"));
	auto fileRation = new QRadioButton(tr("Video file"));
	auto choseSourceLay = new QVBoxLayout;
	choseSourceLay->addWidget(webCamRatio);
	choseSourceLay->addWidget(fileRation);
	choseSource->setLayout(choseSourceLay);

	//
	// Buttons controls.
	//
	auto buttonsGroup = new QGroupBox(tr("Current algorithm"));
	auto saveButton = new QPushButton(tr("Save"));
	auto buttonsLay = new QVBoxLayout;
	buttonsLay->addWidget(startBtn_, 0, Qt::AlignBottom);
	buttonsLay->addWidget(saveButton, 0, Qt::AlignBottom);
	buttonsGroup->setLayout(buttonsLay);

	//
	// All controls lay.
	//

	auto controlsLay = new QVBoxLayout();
	controlsLay->addWidget(choseAlgoBox, 0, Qt::AlignTop);
	controlsLay->addWidget(choseSource, 0, Qt::AlignTop);
	controlsLay->addWidget(buttonsGroup, 0, Qt::AlignBottom);

	//
	// Setting main lay.
	//

	auto mainLay = new QHBoxLayout(centralWidget_);
	mainLay->addLayout(controlsLay);
	mainLay->addWidget(graphicsView_);

	//
	// Connect signals handlers.
	//

	connect(startBtn_, &QPushButton::pressed, this, &MainWindow::OnStartButtonPressed);
	connect(saveButton, &QPushButton::pressed, this, &MainWindow::OnSaveButtonPressed);
	
	qRegisterMetaType<std::shared_ptr<cv::Mat>>();
	connect(
		yoloRatio,
		&QRadioButton::toggled,
		this,
		[this](bool checked) {
			UpdateCurrentAlgo(rclib::NeroAlgoTypes::Yolo);
		});

	connect(
		rcnnRatio,
		&QRadioButton::toggled,
		this,
		[this](bool checked) {
		UpdateCurrentAlgo(rclib::NeroAlgoTypes::MaskRcnn);
	});

	connect(
		webCamRatio,
		&QRadioButton::toggled,
		this,
		[this](bool checked) {
		algoDataSourceType_ = rclib::NeroAlgorithm::DataType::CaptureFromVideoCam;
	});

	connect(
		fileRation,
		&QRadioButton::toggled,
		this,
		[this](bool checked) {
		algoDataSourceType_ = rclib::NeroAlgorithm::DataType::VideoFile;
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

void MainWindow::OnSaveButtonPressed()
{
	if (!(currentAlgorithm_ && !currentAlgorithm_->GetProcessedData().empty()))
	{
		DisplayError("There is nothing to save", this);
		return;
	}

	if (currentAlgorithm_->IsRunning())
	{
		DisplayError("Stop video processing before saving", this);
		return;
	}

	auto filePath = QFileDialog::getSaveFileName(
		this,
		tr("Save processed video"),
		QDir::currentPath(),
		"*.avi");

	auto val =  currentAlgorithm_->GetProcessedData();
	cv::VideoWriter video;
	video.open(
		filePath.toStdString(),
		cv::VideoWriter::fourcc('M','J','P','G'),
		28,
		currentAlgorithm_->GetFrameSize());

	for (const auto& frame : val) {
		video.write(*frame);
	}

	video.release();
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

	std::string inputData;

	switch (algoDataSourceType_)
	{
	case rclib::NeroAlgorithm::DataType::VideoFile:
		inputData = QFileDialog::getOpenFileName(
			this,
			"Select color scale file",
			QDir::currentPath(),
			"*.mp4").toStdString();

		if (inputData.empty()) {
			return;
		}
		break;
	case rclib::NeroAlgorithm::DataType::CaptureFromVideoCam:
		break;
	default:
		DisplayError("Incorrect algo source is chosen", this);
		return;
	}

	workingThread_ = std::make_shared<std::thread>([this, inputData]() {
		currentAlgorithm_->Process(algoDataSourceType_, inputData);
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
		//currentAlgorithm_->SetClassesToDisplay({ "cell phone" });
	}
	catch (const std::exception& e)
	{
		const std::string errorMsg =
			std::string{ "Error occurred while updating algorithm: " } +e.what();
		std::cerr << errorMsg;
		DisplayError(errorMsg.c_str(), this);
	}
}

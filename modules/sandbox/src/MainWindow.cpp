#include "pch.h"
#include "MainWindow.h"

MainWindow::MainWindow(QWidget *parent)
	: QMainWindow(parent)
	, graphicsView(new QGraphicsView(this))
	, startBtn(new QPushButton("start", this))
{

	graphicsView->setScene(new QGraphicsScene(this));
	graphicsView->scene()->addItem(&pixmap);

	setCentralWidget(graphicsView);
	QHBoxLayout* lay = new QHBoxLayout(this);
	lay->addWidget(startBtn);

	connect(startBtn, &QPushButton::pressed, this, &MainWindow::on_startBtn_pressed);
	
	qRegisterMetaType<std::shared_ptr<cv::Mat>>();
	connect(this, &MainWindow::HasFrame, this, &MainWindow::on_startBtn_pressed2);
}

MainWindow::~MainWindow()
{
	yolo3.Stop();
	workingThread_->join();
}

void MainWindow::on_startBtn_pressed2(std::shared_ptr<cv::Mat> frame)
{
	if(!frame->empty())
	{
		QImage qimg(frame->data,
			frame->cols,
			frame->rows,
			frame->step,
			QImage::Format_RGB888);
		pixmap.setPixmap( QPixmap::fromImage(qimg.rgbSwapped()) );
		graphicsView->fitInView(&pixmap, Qt::KeepAspectRatio);
	}
	qApp->processEvents();
}

void MainWindow::on_startBtn_pressed()
{
	if (workingThread_)
	{
		yolo3.Stop();
		workingThread_->join();
		workingThread_.reset();
		startBtn->setText("Start");
		return;
	}
	workingThread_ = std::make_shared<std::thread>([this]() {
		const std::string yoloFilesRoot = ASSETS_DIR;
		rclib::yolo::YOLO3::ConfigFiles cfg;
		cfg.classesNamesFile_ = yoloFilesRoot + "/yolo/coco.names";
		cfg.modelWeights_ = yoloFilesRoot + "/yolo/yolov3.weights";
		cfg.moduleCfgFile_ = yoloFilesRoot + "/yolo/yolov3.cfg";
		const auto inputData = yoloFilesRoot + "/video/run.mp4";
		yolo3.SetConfigs(cfg);
		yolo3.SetFrameProcessedCallback([this](auto frame) {
			emit HasFrame(frame);
		});
		auto res = yolo3.Process(rclib::yolo::YOLO3::DataType::CaptureFromVideoCam, inputData);
	});

	startBtn->setText("Stop");

	//using namespace cv;

	//if(video.isOpened())
	//{
	//	startBtn->setText("Start");
	//	video.release();
	//	return;
	//}
	//video.open(0);
	//startBtn->setText("Stop");

	//Mat frame;
	//while(video.isOpened())
	//{
	//	video >> frame;
	//	if(!frame.empty())
	//	{
	//		QImage qimg(frame.data,
	//			frame.cols,
	//			frame.rows,
	//			frame.step,
	//			QImage::Format_RGB888);
	//		pixmap.setPixmap( QPixmap::fromImage(qimg.rgbSwapped()) );
	//		graphicsView->fitInView(&pixmap, Qt::KeepAspectRatio);
	//	}
	//	qApp->processEvents();
	//}

}

void MainWindow::closeEvent(QCloseEvent *event)
{
	if(video.isOpened())
	{
		QMessageBox::warning(this,
			"Warning",
			"Stop the video before closing the application!");
		event->ignore();
	}
	else
	{
		event->accept();
	}
}



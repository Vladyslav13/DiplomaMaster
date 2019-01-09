#pragma once

#include "pch.h"

#include <thread>
Q_DECLARE_METATYPE(std::shared_ptr<cv::Mat>)
class MainWindow : public QMainWindow
{
	Q_OBJECT

public:
	explicit MainWindow(QWidget *parent = 0);
	~MainWindow();

protected:
	void closeEvent(QCloseEvent *event);

public slots:
	void on_startBtn_pressed();
	void on_startBtn_pressed2(std::shared_ptr<cv::Mat> frame);

signals:
	void HasFrame(std::shared_ptr<cv::Mat>);

private:
	QGraphicsPixmapItem pixmap;
	cv::VideoCapture video;
	QGraphicsView* graphicsView;
	QPushButton* startBtn;
	std::shared_ptr<std::thread> workingThread_ = nullptr;
	rclib::yolo::YOLO3 yolo3;
};

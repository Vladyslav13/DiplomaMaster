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

public:
	//! Stops algorithm.
	void StopAlgorith();

public slots:
	void OnStartButtonPressed();
	void OnUpdateVideoFrame(std::shared_ptr<cv::Mat> frame);

signals:
	void UpdateVideoFrame(std::shared_ptr<cv::Mat>);

private:
	QFrame* centralWidget_;
	QGraphicsView* graphicsView_;
	QGraphicsPixmapItem pixmap_;
	QPushButton* startBtn_;
	rclib::yolo::YOLO3 yolo3_;
	std::shared_ptr<std::thread> workingThread_ = nullptr;
};

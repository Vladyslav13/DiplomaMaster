#pragma once

#include "pch.h"

#include <thread>
Q_DECLARE_METATYPE(std::shared_ptr<cv::Mat>)
class MainWindow : public QMainWindow
{
	Q_OBJECT

public:
	//!
	explicit MainWindow(QWidget *parent = 0);
	//!
	~MainWindow();

public:
	//! Stops algorithm.
	void StopAlgorithm();

public slots:
	//!
	void OnSaveButtonPressed();
	//!
	void OnStartButtonPressed();
	//!
	void OnUpdateVideoFrame(std::shared_ptr<cv::Mat> frame);

signals:
	//!
	void UpdateVideoFrame(std::shared_ptr<cv::Mat>);

	//!
private:
	void UpdateCurrentAlgo(const rclib::NeroAlgoTypes type);

private:
	//!
	QFrame* centralWidget_;
	//!
	rclib::NeroAlgoPtr currentAlgorithm_;
	//!
	QGraphicsView* graphicsView_;
	//!
	QGraphicsPixmapItem pixmap_;
	//!
	QPushButton* startBtn_;
	//!
	std::shared_ptr<std::thread> workingThread_ = nullptr;
	//!
	rclib::NeroAlgorithm::DataType algoDataSourceType_ = rclib::NeroAlgorithm::DataType::Unknown;
};

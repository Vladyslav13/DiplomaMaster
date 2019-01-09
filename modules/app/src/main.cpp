#include "pch.h"

#include "MainWindow.h"

int main(int argc, char *argv[])
{
	int ret = 0;

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
		ret = -1;
	}

	return ret;
}

#include <pch.h>

#include <Utils.h>

void DisplayError(const QString& msgTxt, QWidget* parent)
{
	QMessageBox::critical(parent, "Error", msgTxt);
}


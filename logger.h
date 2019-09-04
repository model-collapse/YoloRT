#ifndef _LOGGER_H_
#define _LOGGER_H_

#include <iostream>
#include "NvInfer.h"

class Logger : public nvinfer1::ILogger           
 {
     void log(nvinfer1::ILogger::Severity severity, const char* msg) override
     {
         // suppress info-level messages
         if (severity != nvinfer1::ILogger::Severity::kINFO)
             std::cout << msg << std::endl;
     }
 };

 #endif

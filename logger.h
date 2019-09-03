#ifndef _LOGGER_H_
#define _LOGGER_H_

#include <iostream>
#include "NvInfer.h"

class Logger : public ILogger           
 {
     void log(Severity severity, const char* msg) override
     {
         // suppress info-level messages
         if (severity != Severity::kINFO)
             std::cout << msg << std::endl;
     }
 };

 #endif _LOGGER_H_
/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */


#ifndef __TRT_UTILS_H__
#define __TRT_UTILS_H__

#include <set>
#include <map>
#include <string>
#include <vector>
#include <cassert>
#include <iostream>
#include <fstream>

#include "NvInfer.h"

#define UNUSED(expr) (void)(expr)

class YoloTinyMaxpoolPaddingFormula : public nvinfer1::IOutputDimensionsFormula
{

private:
    std::set<std::string> m_SamePaddingLayers;

    nvinfer1::DimsHW compute(nvinfer1::DimsHW inputDims, nvinfer1::DimsHW kernelSize,
                             nvinfer1::DimsHW stride, nvinfer1::DimsHW padding,
                             nvinfer1::DimsHW dilation, const char* layerName) const override
    {
        assert(inputDims.d[0] == inputDims.d[1]);
        assert(kernelSize.d[0] == kernelSize.d[1]);
        assert(stride.d[0] == stride.d[1]);
        assert(padding.d[0] == padding.d[1]);

        int outputDim;
        // Only layer maxpool_12 makes use of same padding
        if (m_SamePaddingLayers.find(layerName) != m_SamePaddingLayers.end())
        {
            outputDim = (inputDims.d[0] + 2 * padding.d[0]) / stride.d[0];
        }
        // Valid Padding
        else
        {
            outputDim = (inputDims.d[0] - kernelSize.d[0]) / stride.d[0] + 1;
        }
        return nvinfer1::DimsHW{outputDim, outputDim};
    }

public:
    void addSamePaddingLayer(std::string input) { m_SamePaddingLayers.insert(input); }
};

std::string trim(std::string s);
float clamp(const float val, const float minVal, const float maxVal);
bool fileExists(const std::string fileName, bool verbose = true);
std::vector<float> loadWeights(const std::string weightsFilePath, const std::string& networkType);
std::string dimsToString(const nvinfer1::Dims d);
void displayDimType(const nvinfer1::Dims d);
int getNumChannels(nvinfer1::ITensor* t);
uint64_t get3DTensorVolume(nvinfer1::Dims inputDims);

// Helper functions to create yolo engine
nvinfer1::ILayer* netAddMaxpool(int layerIdx, std::map<std::string, std::string>& block,
                                nvinfer1::ITensor* input, nvinfer1::INetworkDefinition* network);
nvinfer1::ILayer* netAddAvgpool(int layerIdx, std::map<std::string, std::string>& block,
                                nvinfer1::ITensor* input, nvinfer1::INetworkDefinition* network);                            

nvinfer1::ILayer* netAddConvLinear(int layerIdx, std::map<std::string, std::string>& block,
                                   std::vector<float>& weights,
                                   std::vector<nvinfer1::Weights>& trtWeights, int& weightPtr,
                                   int& inputChannels, nvinfer1::ITensor* input,
                                   nvinfer1::INetworkDefinition* network);
nvinfer1::ILayer* netAddConvBNLeaky(int layerIdx, std::map<std::string, std::string>& block,
                                    std::vector<float>& weights,
                                    std::vector<nvinfer1::Weights>& trtWeights, int& weightPtr,
                                    int& inputChannels, nvinfer1::ITensor* input,
                                    nvinfer1::INetworkDefinition* network);
nvinfer1::ILayer* netAddUpsample(int layerIdx, std::map<std::string, std::string>& block,
                                 std::vector<float>& weights,
                                 std::vector<nvinfer1::Weights>& trtWeights, int& inputChannels,
                                 nvinfer1::ITensor* input, nvinfer1::INetworkDefinition* network);
void printLayerInfo(std::string layerIndex, std::string layerName, std::string layerInput,
                    std::string layerOutput, std::string weightPtr);

#endif

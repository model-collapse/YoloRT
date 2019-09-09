#ifndef _JBUF_H_
#define _JBUF_H_

#include "NvInfer.h"
#include "half.h"
#include "common.h"
#include <cuda_runtime_api.h>
#include <cassert>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <string>
#include <vector>
#include <new>
#include "nvdsinfer.h"

template <typename AllocFunc, typename FreeFunc>
class GenericBuffer
{
public:
    //!
    //! \brief Construct an empty buffer.
    //!
    GenericBuffer()
        : mByteSize(0)
        , mBuffer(nullptr)
    {
    }

    //!
    //! \brief Construct a buffer with the specified allocation size in bytes.
    //!
    GenericBuffer(size_t size)
        : mByteSize(size)
    {
        if (!allocFn(&mBuffer, mByteSize))
            throw std::bad_alloc();
    }

    GenericBuffer(GenericBuffer&& buf)
        : mByteSize(buf.mByteSize)
        , mBuffer(buf.mBuffer)
    {
        buf.mByteSize = 0;
        buf.mBuffer = nullptr;
    }

    GenericBuffer& operator=(GenericBuffer&& buf)
    {
        if (this != &buf)
        {
            freeFn(mBuffer);
            mByteSize = buf.mByteSize;
            mBuffer = buf.mBuffer;
            buf.mByteSize = 0;
            buf.mBuffer = nullptr;
        }
        return *this;
    }

    //!
    //! \brief Returns pointer to underlying array.
    //!
    void* data() { return mBuffer; }

    //!
    //! \brief Returns pointer to underlying array.
    //!
    const void* data() const { return mBuffer; }

    //!
    //! \brief Returns the size (in bytes) of the buffer.
    //!
    size_t size() const { return mByteSize; }

    ~GenericBuffer()
    {
        freeFn(mBuffer);
    }

private:
    size_t mByteSize;
    void* mBuffer;
    AllocFunc allocFn;
    FreeFunc freeFn;
};

class ManagedAllocator
{
public:
    bool operator()(void** ptr, size_t size) const { return cudaMallocManaged(ptr, size) == cudaSuccess; }
};

class ManagedFree
{
public:
    void operator()(void* ptr) const { cudaFree(ptr); }
};

using ManagedBuffer = GenericBuffer<ManagedAllocator, ManagedFree>;

class UnifiedBufManager {
    public:
    static const size_t kINVALID_SIZE_VALUE = ~size_t(0);

    //!
    //! \brief Create a BufferManager for handling buffer interactions with engine.
    //!
    UnifiedBufManager(std::shared_ptr<nvinfer1::ICudaEngine> engine, const int& batchSize)
        : mEngine(engine)
        , mBatchSize(batchSize)
    {
        for (int i = 0; i < mEngine->getNbBindings(); i++)
        {
            // Create host and device buffers
            auto dims = mEngine->getBindingDimensions(i);
            size_t vol = samplesCommon::volume(dims);
            auto dataType = mEngine->getBindingDataType(i);
            size_t elementSize = samplesCommon::getElementSize(dataType);
            size_t allocationSize = static_cast<size_t>(mBatchSize) * vol * elementSize;
            std::unique_ptr<ManagedBuffer> manBuf{new ManagedBuffer(allocationSize)};
            mDeviceBindings.emplace_back(manBuf->data());
            NvDsInferLayerInfo layerInfo;
            layerInfo.buffer = manBuf->data();
            layerInfo.dataType = NvDsInferDataType(int(dataType));
            layerInfo.layerName = mEngine->getBindingName(i);
            std::cerr << "binding layer@" << i << ": " << layerInfo.layerName << endl;
            layerInfo.bindingIndex = i;
            layerInfo.dims.numDims = dims.nbDims;
            layerInfo.dims.numElements = 1;
            layerInfo.isInput = 0;
            for (int32_t i = 0; i < int32_t(layerInfo.dims.numDims); i++) {
                layerInfo.dims.d[i] = dims.d[i];
                layerInfo.dims.numElements *= dims.d[i];
            }
            mManagedBuffers.emplace_back(std::move(manBuf));
            mLayerInfos.emplace_back(layerInfo);
        }
    }

    std::vector<void*>& getDeviceBindings() { return mDeviceBindings; }

    const std::vector<void*>& getDeviceBindings() const { return mDeviceBindings; }

    size_t size(const std::string& tensorName) const
    {
        int index = mEngine->getBindingIndex(tensorName.c_str());
        if (index == -1)
            return kINVALID_SIZE_VALUE;
        return mManagedBuffers[index]->size();
    }

    /*void dumpBuffer(std::ostream& os, const std::string& tensorName)
    {
        int index = mEngine->getBindingIndex(tensorName.c_str());
        if (index == -1)
        {
            os << "Invalid tensor name" << std::endl;
            return;
        }
        void* buf = mManagedBuffers[index]->data();
        size_t bufSize = mManagedBuffers[index]->size();
        nvinfer1::Dims bufDims = mEngine->getBindingDimensions(index);
        size_t rowCount = static_cast<size_t>(bufDims.nbDims >= 1 ? bufDims.d[bufDims.nbDims - 1] : mBatchSize);

        os << "[" << mBatchSize;
        for (int i = 0; i < bufDims.nbDims; i++)
            os << ", " << bufDims.d[i];
        os << "]" << std::endl;
        switch (mEngine->getBindingDataType(index))
        {
        case nvinfer1::DataType::kINT32: print<int32_t>(os, buf, bufSize, rowCount); break;
        case nvinfer1::DataType::kFLOAT: print<float>(os, buf, bufSize, rowCount); break;
        case nvinfer1::DataType::kHALF: print<half_float::half>(os, buf, bufSize, rowCount); break;
        case nvinfer1::DataType::kINT8: assert(0 && "Int8 network-level input and output is not supported"); break;
        }
    }*/

    ~UnifiedBufManager() = default;

    void* getBuffer(const std::string& tensorName) const
    {
        int index = mEngine->getBindingIndex(tensorName.c_str());
        if (index == -1)
            return nullptr;
        return mManagedBuffers[index]->data();
    }

    NvDsInferLayerInfo getLayerInfo(const std::string& tensorName) const 
    {
        int index = mEngine->getBindingIndex(tensorName.c_str());
        if (index == -1) {
            NvDsInferLayerInfo emp;
	    return emp;
	}
        return mLayerInfos[index];
    }
    
private:
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;              //!< The pointer to the engine
    int mBatchSize;                                              //!< The batch size
    std::vector<std::unique_ptr<ManagedBuffer>> mManagedBuffers; //!< The vector of pointers to managed buffers
    std::vector<void*> mDeviceBindings;                          //!< The vector of device buffers needed for engine execution
    std::vector<NvDsInferLayerInfo> mLayerInfos;
};

#endif

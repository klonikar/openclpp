/**
	Name: opencl++.cpp
	Author: Kiran Lonikar (klonikar)
	Description: OpenCL Wrapper class implementation
*/

#include "opencl++.h"
#if defined(_WIN32) || defined(_WIN64)
    // Headers needed for Windows
    #include <windows.h>
	#define CL_PLACEMENT_NEW(p, T) (p)->T::T
	#define CL_PLACEMENT_NEW_TEMPL(p, T, C) (p)->T::C
#else
	#define CL_PLACEMENT_NEW(p, T) new (p) T
	#define CL_PLACEMENT_NEW_TEMPL(p, T, C) new (p) T
#endif

#include <iostream>

CLPlatform* CLPlatform::g_allPlatforms = NULL;
cl_uint CLPlatform::g_numPlatforms = CLPlatform::initLib();

CLPlatform::CLPlatform(cl_platform_id __id) : _id(__id), _devices(NULL) {
	cl_int ciErrNum = 0;
	ciErrNum = clGetPlatformInfo (_id, CL_PLATFORM_NAME, sizeof(_name), &_name, NULL);
	ciErrNum = clGetPlatformInfo (_id, CL_PLATFORM_PROFILE, sizeof(_profile), &_profile, NULL);
	ciErrNum = clGetPlatformInfo (_id, CL_PLATFORM_VERSION, sizeof(_version), &_version, NULL);
	ciErrNum = clGetPlatformInfo (_id, CL_PLATFORM_VENDOR, sizeof(_vendor), &_vendor, NULL);
	ciErrNum = clGetPlatformInfo (_id, CL_PLATFORM_EXTENSIONS, sizeof(_extensions), &_extensions, NULL);
	ciErrNum = clGetPlatformInfo (_id, CL_PLATFORM_ICD_SUFFIX_KHR, sizeof(_icd_suffix), &_icd_suffix, NULL);
	ciErrNum = clGetDeviceIDs(_id, CL_DEVICE_TYPE_ALL, 0, NULL, &_numDevices);

	cl_device_id* cdDeviceIds = new cl_device_id[_numDevices];
	if (cdDeviceIds == NULL) {
		return;
	}

	void* raw_memory = operator new[]( _numDevices * sizeof( CLDevice ) );
    _devices = static_cast<CLDevice*>(raw_memory);

	ciErrNum = clGetDeviceIDs(_id, CL_DEVICE_TYPE_ALL, _numDevices, cdDeviceIds, NULL);
    for( cl_uint i = 0; i < _numDevices; ++i ) {
#ifndef _WINDOWS_
        new (&_devices[i]) CLDevice(cdDeviceIds[i]);
#else
		(&_devices[i])->CLDevice::CLDevice( cdDeviceIds[i] );
#endif
	}
	delete[] cdDeviceIds;
}

cl_uint CLPlatform::initLib() {
    cl_uint num_platforms = 0; 
    cl_int ciErrNum = 0;

	ciErrNum = clGetPlatformIDs (0, NULL, &num_platforms);
	if (ciErrNum != CL_SUCCESS || num_platforms == 0)
		return num_platforms;

	cl_platform_id* clPlatformIDs = new cl_platform_id[num_platforms];
	if (clPlatformIDs == NULL) {
		return 0;
	}

	void* raw_memory = operator new[]( num_platforms * sizeof( CLPlatform ) );
    g_allPlatforms = static_cast<CLPlatform*>( raw_memory );

	ciErrNum = clGetPlatformIDs (num_platforms, clPlatformIDs, NULL);
    for( cl_uint i = 0; i < num_platforms; ++i ) {
#ifndef _WINDOWS_
        new (&g_allPlatforms[i]) CLPlatform(clPlatformIDs[i]);
#else
		(&g_allPlatforms[i])->CLPlatform::CLPlatform( clPlatformIDs[i] );
#endif
    }

	delete[] clPlatformIDs;
	return num_platforms;
}

const CLPlatform *CLPlatform::getAllPlatforms() {
	return g_allPlatforms;
}

CLDevice::CLDevice(cl_device_id __id) : _id(__id), _devType(0) {
	cl_int ciErrNum = 0;
    ciErrNum = clGetDeviceInfo(_id, CL_DEVICE_NAME, sizeof(_name), &_name, NULL);
    ciErrNum = clGetDeviceInfo(_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(_numComputeUnits), &_numComputeUnits, NULL);
	ciErrNum = clGetDeviceInfo(_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(_maxWorkGroupSize), &_maxWorkGroupSize, NULL);
	ciErrNum = clGetDeviceInfo(_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(_maxWorkItemSizes), _maxWorkItemSizes, NULL);
	ciErrNum = clGetDeviceInfo(_id, CL_DEVICE_TYPE, sizeof(_devType), &_devType, NULL);
	ciErrNum = clGetDeviceInfo(_id, CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE, sizeof(_nativeDoubleSupport), &_nativeDoubleSupport, NULL);
	ciErrNum = clGetDeviceInfo(_id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, sizeof(_preferredDoubleSupport), &_preferredDoubleSupport, NULL);
}

CLContext::CLContext(const CLDevice *devices, cl_uint numDevices) : _devices(devices), _numDevices(numDevices) {
	cl_device_id* cdDeviceIds = new cl_device_id[numDevices];
	for(cl_uint i = 0;i < numDevices;i++)
		cdDeviceIds[i] = devices[i].id();
	cl_int ciErrNum = 0;
	_id = clCreateContext(0, numDevices, cdDeviceIds, callback, this, &ciErrNum);
	delete[] cdDeviceIds;
}

CLContext::~CLContext() {
	clReleaseContext(_id);
}

void CLContext::callback(const char *errInfo, const void * /* private_info */, size_t /* pvtInfoSize */, void *userData) {
	CLContext *this_ptr = static_cast<CLContext*>(userData);
	std::cerr << "Error on context: " << this_ptr->id() << ": " << errInfo << std::endl;
}

CLCommandQueue::CLCommandQueue(CLContext *ctx, CLDevice *device) : _ctx(ctx), _device(device), _ciErrNum(0) {
	init();
}

void CLCommandQueue::init() {
	_id = clCreateCommandQueue(_ctx->id(), _device ? _device->id() : _ctx->devices()->id(), 0, &_ciErrNum);
}

CLCommandQueue::~CLCommandQueue() {
	clReleaseCommandQueue(_id);
}

CLCommandQueue* CLCommandQueue::enqueueWriteBuffer(CLMem *srcMem, bool blocking, size_t offset, size_t cb, void *src) {
	_ciErrNum = clEnqueueWriteBuffer(_id, srcMem->id(), blocking, offset, cb, src, 0, NULL, NULL);
	return this;
}

CLCommandQueue* CLCommandQueue::enqueueNDRangeKernel(CLKernel *kernel, cl_uint dim, const size_t* global_work_offset,
                       const size_t* global_work_size,
					   const size_t* local_work_size) {
	_ciErrNum = clEnqueueNDRangeKernel(_id, kernel->id(), dim, global_work_offset, global_work_size, local_work_size, 0, NULL, NULL);
	return this;
}

CLCommandQueue* CLCommandQueue::enqueueReadBuffer(CLMem *dstMem, bool blocking, size_t offset, size_t cb, void *dst) {
	_ciErrNum = clEnqueueReadBuffer(_id, dstMem->id(), blocking, offset, cb, dst, 0, NULL, NULL);
	return this;
}

CLMem::CLMem(CLContext *ctx, cl_mem_flags flags, size_t size, void *hostPtr) : _ctx(ctx), _flags(flags), _size(size), _hostPtr(hostPtr) {
	cl_int ciErrNum = 0;
	_id = clCreateBuffer(_ctx->id(), _flags, _size, _hostPtr, &ciErrNum);
}

CLMem::~CLMem() {
	clReleaseMemObject(_id);
}

CLReadOnlyMem::CLReadOnlyMem(CLContext *ctx, size_t size, void *hostPtr) : CLMem(ctx, CL_MEM_READ_ONLY, size, hostPtr) {

}

CLReadOnlyMem::~CLReadOnlyMem() {
}

CLWriteOnlyMem::CLWriteOnlyMem(CLContext *ctx, size_t size, void *hostPtr) : CLMem(ctx, CL_MEM_WRITE_ONLY, size, hostPtr) {

}

CLWriteOnlyMem::~CLWriteOnlyMem() {
}

CLProgram::CLProgram(CLContext *ctx, cl_uint count, const char **strings, const size_t *lengths) : _ctx(ctx), _count(count), _strings(strings), _lengths(lengths) {
	_ciErrNum = 0;
	_id = clCreateProgramWithSource(_ctx->id(), _count, _strings, _lengths, &_ciErrNum);
}

CLProgram::~CLProgram() {
	clReleaseProgram(_id);
}

CLProgram* CLProgram::build(const char *options) {
	_ciErrNum = clBuildProgram(_id, 0, NULL, options, NULL, NULL);
	if(_ciErrNum == CL_SUCCESS)
		return this;
	else
		return NULL;
}

CLKernel::CLKernel(CLProgram *program, const char *name) : _program(program), _name(name), _ciErrNum(0), _argNum(0) {
	_id = clCreateKernel(_program->id(), _name, &_ciErrNum);
}

CLKernel::~CLKernel() {
	clReleaseKernel(_id);
}

CLKernel* CLKernel::setArg(CLMem* arg, int argNum) {
	if(argNum != -1)
		_argNum = (cl_uint) argNum;

	_ciErrNum = clSetKernelArg(_id, _argNum++, sizeof(cl_mem), (void*)&arg->id());
	return this;
}

CLKernel* CLKernel::setArg(cl_int& arg, int argNum) {
	if(argNum != -1)
		_argNum = (cl_uint) argNum;

	_ciErrNum = clSetKernelArg(_id, _argNum++, sizeof(cl_int), (void*)&arg);
	return this;
}

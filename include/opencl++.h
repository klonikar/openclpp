/**
	Name: opencl++.h
	Author: Kiran Lonikar (klonikar)
	Description: OpenCL C++ Wrapper classes header - Fluent style. This interface is different from the one described on 
	https://www.khronos.org/registry/cl/specs/opencl-cplusplus-1.2.pdf
	Notable differences:
	1. CLPlatform::getAllPlatforms() returns pointer to first platform in the array instead of STL vector
	2. CLPlatform.getDevices() returns pointer to first device in the array instead of STL vector
	3. CLPlatform, CLDevice contain the attributes instead of having to call getPlatformInfo or getDeviceInfo.
	   In general total independence from functions and macros defined in cl.h.
    4. Fluent style functions so that functions can be chained.
	   Functions like CLProgram.build, CLKernel.setArg, CLCommandQueue.enqueue* return the "this" object.
	   See sample code to know how it simplifies the usage.

	Implementation Notes: 
	1. Creating context from device type is not yet supported (clCreateContextFromType)
	2. Command queues can be created using the devices used for creating context.
*/
#ifndef _OPENCLPP_H_
#define _OPENCLPP_H_
// All OpenCL headers
#if defined (__APPLE__) || defined(MACOSX)
    #include <OpenCL/opencl.h>
#else
    #include <CL/opencl.h>
#endif 

#define MAX_CLPLATFORM_NAME_LEN 128
#define MAX_CLPLATFORM_PROFILE_LEN 128
#define MAX_CLPLATFORM_VENDOR_LEN 128
#define MAX_CLPLATFORM_VERSION_LEN 128
#define MAX_CLPLATFORM_EXTENSIONS_LEN 1024
#define MAX_CLPLATFORM_ICD_SUFFIX_LEN 16
#define MAX_DEVICE_NAME 128

class CLDevice;
class CLContext;
class CLCommandQueue;
class CLProgram;
class CLKernel;

// Initial implementations. Change later to impl pattern for binary compatibility
class CLPlatform {
private:
	// data members
	cl_platform_id _id;
	char _name[MAX_CLPLATFORM_NAME_LEN];
	char _profile[MAX_CLPLATFORM_PROFILE_LEN];
	char _vendor[MAX_CLPLATFORM_VENDOR_LEN];
	char _version[MAX_CLPLATFORM_VERSION_LEN];
	char _extensions[MAX_CLPLATFORM_EXTENSIONS_LEN];
	char _icd_suffix[MAX_CLPLATFORM_ICD_SUFFIX_LEN];
	cl_uint _numDevices;
	CLDevice *_devices;

	// static data members
	static CLPlatform* g_allPlatforms;

	// constructor: private
	CLPlatform(cl_platform_id);
	// static library initializer to initialize static data members.
	static cl_uint initLib();

public:
	static cl_uint g_numPlatforms;
	static const CLPlatform *getAllPlatforms();

	CLDevice *devices() const { return _devices; }

	const char *name() const { return _name; }
	const char *profile() const { return _profile; }
	const char *vendor() const { return _vendor; }
	const char *version() const { return _version; }
	const char *extensions() const { return _extensions; }
	const char *icd_suffix() const { return _icd_suffix; }
	cl_platform_id id() const { return _id; }
	cl_uint numDevices() const { return _numDevices; }
};

class CLDevice {
private:
	cl_device_id _id;
	char _name[MAX_DEVICE_NAME];
	cl_uint _numComputeUnits;          // Number of compute units (SM's on NV GPU)
	cl_uint _maxWorkGroupSize;         // Max work group size
	cl_uint _maxWorkItemSizes[3];      // Max work item sizes
	cl_device_type _devType;
	cl_uint _nativeDoubleSupport;
	cl_uint _preferredDoubleSupport;

	CLDevice(cl_device_id);
public:
	cl_uint numComputeUnits() const { return _numComputeUnits; }
	cl_uint maxWorkGroupSize() const { return _maxWorkGroupSize; }
	const cl_uint *maxWorkItemSizes() const { return _maxWorkItemSizes; }
	const char *name() const { return _name; }
	cl_device_id id() const { return _id; }
	bool isGpu() const { return (_devType & CL_DEVICE_TYPE_GPU) ? true : false; }
	bool isCpu() const { return (_devType & CL_DEVICE_TYPE_CPU)  ? true : false; }
	bool isAccelerator() const { return (_devType & CL_DEVICE_TYPE_ACCELERATOR)  ? true : false; }
	cl_device_type devType() const { return _devType; }
	cl_uint nativeDoubleSupport() const { return _nativeDoubleSupport; }
	cl_uint preferredDoubleSupport() const { return _preferredDoubleSupport; }

	// add CLPlatform as friend class
	friend class CLPlatform;
};

class CLContext {
private:
	cl_context _id;
	const CLDevice *_devices;
	cl_uint _numDevices;

	static void CL_CALLBACK callback(const char *, const void *, size_t, void *);
public:
	CLContext(const CLDevice *devices, cl_uint numDevices);
	~CLContext();

	cl_context id() const { return _id; }
	const CLDevice *devices() const { return _devices; }
	cl_uint numDevices() const { return _numDevices; }
};

class CLMem {
private:
	cl_mem _id;
	cl_mem_flags _flags;
	CLContext *_ctx;
	size_t _size;
	void *_hostPtr;
public:
	// Constructor
	CLMem(CLContext *ctx, cl_mem_flags flags, size_t size, void *hostPtr = NULL);
	virtual ~CLMem();

	cl_mem& id() { return _id; }
	cl_mem_flags flags() const { return _flags; }
	const CLContext *ctx() const { return _ctx; }
	size_t size() const { return _size; }
	void *hostPtr() const { return _hostPtr; }
};

class CLReadOnlyMem : public CLMem {
public:
	CLReadOnlyMem(CLContext *ctx, size_t size, void *hostPtr = NULL);
	virtual ~CLReadOnlyMem();
};

class CLWriteOnlyMem : public CLMem {
public:
	CLWriteOnlyMem(CLContext *ctx, size_t size, void *hostPtr = NULL);
	virtual ~CLWriteOnlyMem();
};

class CLProgram {
private:
	cl_program _id;
	CLContext *_ctx;
	cl_uint _count;
	const char **_strings;
	const size_t *_lengths;
	cl_int _ciErrNum;
public:
	CLProgram(CLContext *ctx, cl_uint count, const char **strings, const size_t *lengths = NULL);
	~CLProgram();

	CLProgram* build(const char *options = NULL);

	cl_program id() { return _id; }
	const CLContext *ctx() const { return _ctx; }
	cl_int ciErrNum() const { return _ciErrNum; }
};

class CLKernel {
private:
	cl_kernel _id;
	CLProgram *_program;
	const char* _name;
	cl_int _ciErrNum;
	cl_uint _argNum;
public:
	CLKernel(CLProgram *program, const char *name);
	~CLKernel();

	cl_kernel id() { return _id; }
	const CLProgram *program() const { return _program; }
	const char* name() const { return _name; }
	cl_int ciErrNum() const { return _ciErrNum; }
	cl_uint argNum() const { return _argNum; }

	CLKernel* setArg(CLMem* arg, int argNum = -1);
	CLKernel* setArg(cl_int& arg, int argNum = -1);
};

class CLCommandQueue {
private:
	cl_command_queue _id;
	const CLContext *_ctx;
	const CLDevice *_device;
	cl_int _ciErrNum;

	void init();
public:
	// Construct using context and a device of the context
	CLCommandQueue(CLContext *ctx, CLDevice *device = NULL);
	~CLCommandQueue();

	// Getters
	cl_command_queue id() const { return _id; }
	const CLContext *ctx() const { return _ctx; }
	const CLDevice *device() const { return _device; }
	cl_int ciErrNum() const { return _ciErrNum; }

	// Functionality
	CLCommandQueue* enqueueWriteBuffer(CLMem *srcMem, bool blocking, size_t offset, size_t cb, void *src);
	CLCommandQueue* enqueueNDRangeKernel(CLKernel *kernel, cl_uint dim, const size_t* global_work_offset,
                       const size_t* global_work_size,
                       const size_t* local_work_size);
	CLCommandQueue* enqueueReadBuffer(CLMem *dstMem, bool blocking, size_t offset, size_t cb, void *dst);

};

#endif /* _OPENCLPP_H_ */
openclpp
=======

OpenCL C++ Wrapper classes header - Fluent style (http://en.wikipedia.org/wiki/Fluent_interface)
OpenCL C++ Wrapper classes header - Fluent style. This interface is different from the one described on https://www.khronos.org/registry/cl/specs/opencl-cplusplus-1.2.pdf
    Notable differences:
   l. CLPlatform::getAllPlatforms() returns pointer to first platform in the array instead of STL vector
   l. CLPlatform.getDevices() returns pointer to first device in the array instead of STL vector
   l. CLPlatform, CLDevice contain the attributes instead of having to call getPlatformInfo or getDeviceInfo.
	   In general total independence from functions and macros defined in cl.h.
   l. Fluent style functions so that functions can be chained.
	   Functions like CLProgram.build, CLKernel.setArg, CLCommandQueue.enqueue* return the "this" object.
	   See sample code to know how it simplifies the usage.

	Implementation Notes: 
   l. Creating context from device type is not yet supported (clCreateContextFromType)
   l. Command queues can be created using the devices used for creating context.
   l. Does not support all the attributes and functors yet.

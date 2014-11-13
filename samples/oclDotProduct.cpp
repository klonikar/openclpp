/*
 File: oclDotProduct.cpp
 compilation and execution:
 Windows:
 call "\Program Files (x86)\Microsoft Visual Studio 9.0"\Common7\Tools\vsvars32.bat
 cd samples
 cl -I. -I .. -I ..\include oclDotProduct.cpp ..\src\opencl++.cpp ..\lib\Win32\OpenCL.lib
 oclDotProduct.exe [-local 8/16/32/64/128/256/512/1024]
 Linux:
 TBD
*/
#include <opencl++.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#ifdef _WIN32
    // Headers needed for Windows
    #include <windows.h>
#endif

#define CONFIG_USE_DOUBLE
#ifdef CONFIG_USE_DOUBLE
#define DOUBLE_SUPPORT_AVAILABLE
#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif
#endif // CONFIG_USE_DOUBLE

#if defined(DOUBLE_SUPPORT_AVAILABLE)

// double
typedef cl_double real_t;
typedef cl_double2 real2_t;
typedef cl_double3 real3_t;
typedef cl_double4 real4_t;
typedef cl_double8 real8_t;
typedef cl_double16 real16_t;
#define PI 3.14159265358979323846

#else

// float
typedef cl_float real_t;
typedef cl_float2 real2_t;
typedef cl_float3 real3_t;
typedef cl_float4 real4_t;
typedef cl_float8 real8_t;
typedef cl_float16 real16_t;
#define PI 3.14159265359f

#endif

// Name of the file with the source code for the computation kernel
// *********************************************************************
const char* cSourceFile = "DotProduct.cl";

// Host buffers for demo
// *********************************************************************
void *srcA, *srcB, *dst;        // Host buffers for OpenCL test
void* Golden;                   // Host buffer for host golden processing cross check

// OpenCL Vars
CLContext *cxGPUContextP;  // OpenCL context
CLCommandQueue *cqCommandQueueP;// OpenCL command que
CLProgram *cpProgramP;           // OpenCL program
CLKernel *ckKernelP;             // OpenCL kernel
CLReadOnlyMem *cmDevSrcAP;               // OpenCL device source buffer A
CLReadOnlyMem *cmDevSrcBP;               // OpenCL device source buffer B 
CLWriteOnlyMem *cmDevDstP;                // OpenCL device destination buffer 
size_t szGlobalWorkSize;        // Total # of work items in the 1D range
size_t szLocalWorkSize;		    // # of work items in the 1D work group	
size_t szParmDataBytes;			// Byte size of context information
size_t szKernelLength;			// Byte size of kernel code
cl_int ciErrNum;			    // Error code var
char* cPathAndName = NULL;      // var for full paths to data, src, etc.
char* cSourceCL = NULL;         // Buffer to hold source for compilation 
const char* cExecutableName = NULL;

// demo config vars
int iNumElements= 12779440;	    // Length of float arrays to process (odd # for illustration)
bool bNoPrompt = false;  

// Forward Declarations
// *********************************************************************
void DotProductHost(const real_t* pfData1, const real_t* pfData2, real_t* pfResult, int iNumElements);
void Cleanup (int iExitCode);
void (*pCleanup)(int) = &Cleanup;

int *gp_argc = NULL;
char ***gp_argv = NULL;

//////////////////////////////////////////////////////////////////////////////
//! Loads a Program file and prepends the cPreamble to the code.
//!
//! @return the source string if succeeded, 0 otherwise
//! @param cFilename        program filename
//! @param cPreamble        code that is prepended to the loaded file, typically a set of #defines or a header
//! @param szFinalLength    returned length of the code string
//////////////////////////////////////////////////////////////////////////////
char* oclLoadProgSource(const char* cFilename, const char* cPreamble, size_t* szFinalLength)
{
    // locals 
    FILE* pFileStream = NULL;
    size_t szSourceLength;

    // open the OpenCL source code file
    #ifdef _WIN32   // Windows version
        if(fopen_s(&pFileStream, cFilename, "rb") != 0) 
        {       
            return NULL;
        }
    #else           // Linux version
        pFileStream = fopen(cFilename, "rb");
        if(pFileStream == 0) 
        {       
            return NULL;
        }
    #endif

    size_t szPreambleLength = strlen(cPreamble);

    // get the length of the source code
    fseek(pFileStream, 0, SEEK_END); 
    szSourceLength = ftell(pFileStream);
    fseek(pFileStream, 0, SEEK_SET); 

    // allocate a buffer for the source code string and read it in
    char* cSourceString = (char *)malloc(szSourceLength + szPreambleLength + 1); 
    memcpy(cSourceString, cPreamble, szPreambleLength);
    if (fread((cSourceString) + szPreambleLength, szSourceLength, 1, pFileStream) != 1)
    {
        fclose(pFileStream);
        free(cSourceString);
        return 0;
    }

    // close the file and return the total length of the combined (preamble + source) string
    fclose(pFileStream);
    if(szFinalLength != 0)
    {
        *szFinalLength = szSourceLength + szPreambleLength;
    }
    cSourceString[szSourceLength + szPreambleLength] = '\0';

    return cSourceString;
}

// Round Up Division function
size_t shrRoundUp(int group_size, int global_size) 
{
    int r = global_size % group_size;
    if(r == 0) 
    {
        return global_size;
    } else 
    {
        return global_size + group_size - r;
    }
}

void shrFillArray(real_t* pfData, int iSize)
{
    int i; 
    const real_t fScale = 1.0f / (real_t)RAND_MAX;
    for (i = 0; i < iSize; ++i) 
    {
        pfData[i] = fScale * rand();
    }
}
// Main function 
// *********************************************************************
int main(int argc, char **argv)
{
    gp_argc = &argc;
    gp_argv = &argv;

    // Get the NVIDIA platform
	const CLPlatform *platforms = CLPlatform::getAllPlatforms();
	const CLPlatform *nvidiaPlatformP = NULL;
	const CLDevice *targetDeviceP = NULL;
	for(cl_uint i = 0;i < CLPlatform::g_numPlatforms;i++) {
		printf("platform name: %s, profile: %s, version: %s, vendor: %s, extensions: %s, icd suffix: %s, devices: %u\n", platforms[i].name(), platforms[i].profile(), platforms[i].version(), platforms[i].vendor(), platforms[i].extensions(), platforms[i].icd_suffix(), platforms[i].numDevices()); 
		if(strstr(platforms[i].name(), "NVIDIA")) {
			nvidiaPlatformP = &platforms[i];
			targetDeviceP = platforms[i].devices(); // first device of the platform
		}
		for(cl_uint j = 0;j < platforms[i].numDevices();++j) {
		    printf("\n device %s # of Compute Units = %u, work group size %u, sizes (%u, %u, %u), Type %u, GPU %d, CPU %d, Accelerator %d, native double support %u, preferred double support %u\n", 
				platforms[i].devices()[j].name(),
				platforms[i].devices()[j].numComputeUnits(),
				platforms[i].devices()[j].maxWorkGroupSize(),
				platforms[i].devices()[j].maxWorkItemSizes()[0], 
				platforms[i].devices()[j].maxWorkItemSizes()[1],
				platforms[i].devices()[j].maxWorkItemSizes()[2],
				(int) platforms[i].devices()[j].devType(),
				(int) platforms[i].devices()[j].isGpu(),
				(int) platforms[i].devices()[j].isCpu(),
				(int) platforms[i].devices()[j].isAccelerator(),
				platforms[i].devices()[j].nativeDoubleSupport(),
				platforms[i].devices()[j].preferredDoubleSupport()
				); 
		}

	}

    // get command line arg for quick test, if provided
    // bNoPrompt = shrCheckCmdLineFlag(argc, (const char**)argv, "noprompt");

    // start logs
	cExecutableName = argv[0];

    // set and log Global and Local work size dimensions
    szLocalWorkSize = 256;
	if(argc > 2 && strcmp(argv[1], "-local") == 0) {
		szLocalWorkSize = atoi(argv[2]);
	}
    szGlobalWorkSize = shrRoundUp((int)szLocalWorkSize, iNumElements);  // rounded up to the nearest multiple of the LocalWorkSize
    // Allocate and initialize host arrays
    srcA = (void *)malloc(sizeof(real4_t) * szGlobalWorkSize);
    srcB = (void *)malloc(sizeof(real4_t) * szGlobalWorkSize);
    dst = (void *)malloc(sizeof(real_t) * szGlobalWorkSize);
    Golden = (void *)malloc(sizeof(real_t) * iNumElements);
    shrFillArray((real_t*)srcA, 4 * iNumElements);
    shrFillArray((real_t*)srcB, 4 * iNumElements);

	cxGPUContextP = new CLContext(targetDeviceP, 1);

    // Create a command-queue
	cqCommandQueueP = new CLCommandQueue(cxGPUContextP);

    // Allocate the OpenCL buffer memory objects for source and result on the device GMEM
    cmDevSrcAP = new CLReadOnlyMem(cxGPUContextP, sizeof(real_t)*szGlobalWorkSize*4);
    cmDevSrcBP = new CLReadOnlyMem(cxGPUContextP, sizeof(real_t)*szGlobalWorkSize*4);
    cmDevDstP = new CLWriteOnlyMem(cxGPUContextP, sizeof(real_t)*szGlobalWorkSize);

    // Read the OpenCL kernel in from source file
    cSourceCL = oclLoadProgSource(cSourceFile, "", &szKernelLength);

    // Build the program with 'mad' Optimization option
    #ifdef MAC
        char* flags = "-cl-fast-relaxed-math -DMAC";
    #else
        char* flags = "-cl-fast-relaxed-math";
		flags = "-D CONFIG_USE_DOUBLE";
    #endif

	cpProgramP = (new CLProgram(cxGPUContextP, 1, (const char **)&cSourceCL, &szKernelLength))->build(flags);

    // Create the kernel
    ckKernelP = (new CLKernel(cpProgramP, "DotProduct"))
					->setArg(cmDevSrcAP)
					->setArg(cmDevSrcBP)
					->setArg(cmDevDstP)
					->setArg(iNumElements);

    // --------------------------------------------------------
    // Core sequence... copy input data to GPU, compute, copy results back

    // Asynchronous write of data to GPU device
    // Launch kernel
	SYSTEMTIME t1_g, t2_g;
	GetSystemTime(&t1_g);
    cqCommandQueueP->enqueueWriteBuffer(cmDevSrcAP, CL_FALSE, 0, sizeof(real_t) * szGlobalWorkSize * 4, srcA)
				   ->enqueueWriteBuffer(cmDevSrcBP, CL_FALSE, 0, sizeof(real_t) * szGlobalWorkSize * 4, srcB)
				   ->enqueueNDRangeKernel(ckKernelP, 1, NULL, &szGlobalWorkSize, &szLocalWorkSize)
				   ->enqueueReadBuffer(cmDevDstP, CL_TRUE, 0, sizeof(real_t) * szGlobalWorkSize, dst)
				   ;

	GetSystemTime(&t2_g);
	printf("kernel %d secs, %d mili\n", t2_g.wSecond-t1_g.wSecond, t2_g.wMilliseconds-t1_g.wMilliseconds);

    // Compute and compare results for golden-host and report errors and pass/fail
	SYSTEMTIME t1, t2;
	GetSystemTime(&t1);
    DotProductHost ((const real_t*)srcA, (const real_t*)srcB, (real_t*)Golden, iNumElements);
	GetSystemTime(&t2);
	printf("host %d secs, %d mili\n", t2.wSecond-t1.wSecond, t2.wMilliseconds-t1.wMilliseconds);

    // Cleanup and leave
    Cleanup (EXIT_SUCCESS);
}

// "Golden" Host processing dot product function for comparison purposes
// *********************************************************************
void DotProductHost(const real_t* pfData1, const real_t* pfData2, real_t* pfResult, int iNumElements)
{
    int i, j, k;
    for (i = 0, j = 0; i < iNumElements; i++) 
    {
		//pfResult[i] =  pfData1[i] * sin(pfData2[i]) + 1;
		
        pfResult[i] = 0.0f;
        for (k = 0; k < 4; k++, j++) 
        {
            pfResult[i] += pfData1[j] * pfData2[j]; 
        } 
		
    }
}

// Cleanup and exit code
// *********************************************************************
void Cleanup(int iExitCode)
{
    // Cleanup allocated objects
    if(cPathAndName)free(cPathAndName);
    if(cSourceCL)free(cSourceCL);
	if(ckKernelP) delete ckKernelP;  
    if(cpProgramP) delete cpProgramP;
    if(cqCommandQueueP) delete cqCommandQueueP;
    if(cxGPUContextP) delete cxGPUContextP;
    if (cmDevSrcAP) delete cmDevSrcAP;
    if (cmDevSrcBP) delete cmDevSrcBP;
    if (cmDevDstP) delete cmDevDstP;

    // Free host memory
    free(srcA); 
    free(srcB);
    free (dst);
    free(Golden);

}

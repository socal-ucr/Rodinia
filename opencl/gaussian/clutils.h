#ifndef __CL_UTILS_H__
#define __CL_UTILS_H__

/*!
	\file clutils.h
*/

// All OpenCL headers
#if defined (__APPLE__) || defined(MACOSX)
    #include <OpenCL/opencl.h>
#else
    #include <CL/opencl.h>
#endif 

cl_context cl_init_context(int platform,int dev,int quiet=0);

cl_kernel * Kernel_Precompile(int n);
cl_program cl_CompileProgram(char * kernelPath, char * compileoptions, bool verboseoptions = 0);

void    cl_sync();
void    cl_cleanup();

cl_mem  cl_allocDevice(unsigned int mem_size);

cl_mem * cl_allocDeviceConst(unsigned int mem_size, void * host_ptr);

void    cl_freeDevice(cl_mem *mem);
void    cl_copyToDevice(cl_mem dst, void *src, unsigned mem_size, unsigned int event_id  = 999);
void    cl_copyToHost(void *dst, cl_mem src, unsigned mem_size);

int     cl_errChk(const cl_int status, const char *msg);
int     cl_errChk_sync(const cl_int status, const char *msg);


void    cl_printBinaries(cl_program program);
void    cl_copyTextureToDevice(cl_mem dst, void* src, int width, int height);
cl_mem * cl_allocTexture(int width, int height, void *data, size_t elementSize,
                        cl_channel_type type);

cl_program cl_getProgram();
cl_context cl_getContext();
cl_command_queue cl_getCommandQueue();
cl_device_id cl_getDeviceId();
cl_program buildProgram(char *kernelPath, char * compileoptions);

void cl_KernelTime(cl_event );
void cl_KernelTimeSync(cl_event );
void cl_TimeStart(cl_event, cl_profiling_info,char * );
void cl_TimeStop(cl_event, cl_profiling_info, char * );

//define vector types
typedef struct{
  int x;
  int y;
} int2;

typedef struct{
  float x;
  float y;
}float2;

typedef struct{
	float x;
	float y;
	float z;
	float w;
}float4;

#define MAX_ERR_VAL 64

#define FALSE 0
#define TRUE 1

#endif

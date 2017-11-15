
/***********************************************************
*  OpenCL implementation of OpenSURF                       *
*  By Chris Gregg and Perhaad Mistry                       *
*  under the direction of Norm Rubin                       *
*  Contact: chg5w@virginia.edu, pmistry@ece.neu.edu        *
*  Advanced Micro Devices                                  *
*  August 2010                                             *
*                                                          *
*  Modified from OpenSURF code developed by C. Evans:      *
*  --- OpenSURF ---                                        *
*  This library is distributed under the GNU GPL. Please   *
*  contact chris.evans@irisys.co.uk for more information.  *
*                                                          *
*  C. Evans, Research Into Robust Visual Features,         *
*  MSc University of Bristol, 2008.                        *
*                                                          *
************************************************************/
/**
	\file clutils.cpp
*/


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "clutils.h"

#include "gettimeofday.h"


void    cl_sync();
void    cl_cleanup();


//! Global pointer to list of precompiled kernels
cl_kernel * kernel_list;

//! Number of events declared in event_table class
#define EVENT_COUNT 20

//! Globally visible list of events
cl_event * event_list;


cl_kernel * Kernel_Precompile(int n);



cl_mem cl_allocDevice(unsigned int mem_size);
cl_mem *cl_allocDeviceConst(unsigned int mem_size, void * host_ptr);

void    cl_freeDevice(cl_mem *mem);
//void    cl_copyToDevice(cl_mem dst, void *src, unsigned mem_size, unsigned int event_id );
void    cl_copyToHost(void* dst, cl_mem src, unsigned mem_size);
void    cl_printBinaries(cl_program program);

//! Globally visible OpenCL program
cl_program clProgram;
//! Globally visible OpenCL contexts
cl_context clGPUContext;
//! Globally visible OpenCL cmd queue
cl_command_queue clCommandQueue;

cl_device_id device;


//! Return a cl_program
cl_program cl_getProgram()
{
	return clProgram;
}

//! Returns a reference to the command queue
/*!
	Returns a reference to the command queue \n
	Used for any OpenCl call that needs the command queue declared in clutils.cpp
*/
cl_command_queue cl_getCommandQueue()
{
	return clCommandQueue;
}

cl_device_id cl_getDeviceId(){
	return device;
}

cl_context cl_getContext()
{
	return clGPUContext;
}



void  cl_cleanup()
{
	if(clProgram) {
		clReleaseProgram(clProgram);
	}
	if(clCommandQueue) {
		clReleaseCommandQueue(clCommandQueue);
	}
	if(clGPUContext) {
		clReleaseContext(clGPUContext);
	}
	free(kernel_list);
	free(event_list);
}

/*!
	Wait till all pending commands in queue are finished
*/
void cl_sync()
{
	clFinish(clCommandQueue);
}




cl_mem *cl_allocTexture(int width, int height, void *data, size_t elementSize, cl_channel_type type) {
	cl_int status;

	cl_mem *mem;
	mem = (cl_mem *) malloc(sizeof(cl_mem));

	cl_image_format image_format;
	image_format.image_channel_order = CL_R;
	image_format.image_channel_data_type = type;

	*mem = clCreateImage2D(clGPUContext,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		&image_format,
		width,
		height,
		width*elementSize,  // TODO make sure this is good
		data,
		&status);

	if(cl_errChk(status, "creating texture")) {
		exit(1);
	}

	return mem;
}

//! Allocate memory on device
/*!
	\param mem_size Size of memory in bytes
	\param ptr_name Optional parameter for pointer name
	\return Returns a cl_mem object that points to device memory
*/
cl_mem cl_allocDevice(unsigned mem_size)
{
	cl_mem mem;
	mem = (cl_mem) malloc(sizeof(cl_mem));
	cl_int status;
	//cl_ulong memAvail;

	static int allocationCount = 1;

	allocationCount+=mem_size;
	//printf("Allocation count: %d\n",allocationCount);
	//printf("ALLOCATING %u BYTES\n", mem_size);

	mem = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, mem_size, NULL,
		&status);
	//printf("creating buffer prob.\n");

	if(cl_errChk(status, "creating buffer")) {
		exit(1);
	}
	return mem;
}


cl_mem * cl_allocDeviceConst(unsigned mem_size, void * host_ptr)
{
	cl_mem * mem;
	mem = (cl_mem *) malloc(sizeof(cl_mem));
	cl_int status;


	*mem = clCreateBuffer(clGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		mem_size, host_ptr,
		&status);
	if(cl_errChk(status, "Error creating Const Mem buffer")) {
		printf("Error Allocating %u BYTES in Const Memory\n", mem_size);
		exit(1);
	}
	return mem;
}

//! This function will recieve information of where in the program we are as "event_id"
void cl_copyToDevice(cl_mem dst, void* src, unsigned mem_size, unsigned int event_id )
{
	cl_int status;
    if(event_id == 999)
    {
        status = clEnqueueWriteBuffer(clCommandQueue, dst, CL_TRUE, 0,
                mem_size, src, 0, NULL, NULL);
        if(cl_errChk(status, "write buffer")) {
            exit(1);
        }
    }
}

void cl_copyTextureToDevice(cl_mem dst, void* src, int width, int height)
{
	cl_int status;
	const size_t szTexOrigin[3] = {0, 0, 0};
	const size_t szTexRegion[3] = {height, width, 1};
	status = clEnqueueWriteImage(clCommandQueue, dst, CL_TRUE, szTexOrigin,
		szTexRegion, 0, 0, src, 0, NULL, NULL);
	if(cl_errChk(status, "write buffer texture")) {
		exit(1);
	}
}

void cl_copyToHost(void* dst, cl_mem src, unsigned mem_size)
{
	cl_int status;
	status = clEnqueueReadBuffer(clCommandQueue, src, CL_TRUE, 0,
		mem_size, dst, 0, NULL, NULL);
	cl_sync();
	if(cl_errChk(status, "read buffer")) {
		exit(1);
	}
}

void cl_freeDevice(cl_mem *mem)
{
	cl_int status;
	printf("cl_mem:%p\n",*mem);
	status = clReleaseMemObject(*mem);
	if(cl_errChk(status, "releasing mem object")) {
		exit(1);
	}
	free(mem);
}


//! OpenCl error code list
/*!
	An array of character strings used to give the error corresponding to the error code \n

	The error code is the index within this array
*/
const char *cl_errs[MAX_ERR_VAL] = {
	"CL_SUCCESS",                       //0
	"CL_DEVICE_NOT_FOUND",              //-1
	"CL_DEVICE_NOT_AVAILABLE",          //-2
	"CL_COMPILER_NOT_AVAILABLE",        //-3
	"CL_MEM_OBJECT_ALLOCATION_FAILURE", //-4
	"CL_OUT_OF_RESOURCES",              //-5
	"CL_OUT_OF_HOST_MEMORY",            //-6
	"CL_PROFILING_INFO_NOT_AVAILABLE",  //-7
	"CL_MEM_COPY_OVERLAP",              //-8
	"CL_IMAGE_FORMAT_MISMATCH",         //-9
	"CL_IMAGE_FORMAT_NOT_SUPPORTED",    //-10
	"CL_BUILD_PROGRAM_FAILURE",         //-11
	"CL_MAP_FAILURE",                   //-12
	"",               //-13
	"",               //-14
	"",               //-15
	"",               //-16
	"",               //-17
	"",               //-18
	"",               //-19
	"",               //-20
	"",               //-21
	"",               //-22
	"",               //-23
	"",               //-24
	"",               //-25
	"",               //-26
	"",               //-27
	"",               //-28
	"",               //-29
	"CL_INVALID_VALUE", //-30
	"CL_INVALID_DEVICE_TYPE", //-31
	"CL_INVALID_PLATFORM", //-32
	"CL_INVALID_DEVICE", //-33
	"CL_INVALID_CONTEXT",
	"CL_INVALID_QUEUE_PROPERTIES",
	"CL_INVALID_COMMAND_QUEUE",
	"CL_INVALID_HOST_PTR",
	"CL_INVALID_MEM_OBJECT",
	"CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",
	"CL_INVALID_IMAGE_SIZE",
	"CL_INVALID_SAMPLER",
	"CL_INVALID_BINARY",
	"CL_INVALID_BUILD_OPTIONS",
	"CL_INVALID_PROGRAM",
	"CL_INVALID_PROGRAM_EXECUTABLE",
	"CL_INVALID_KERNEL_NAME",
	"CL_INVALID_KERNEL_DEFINITION",
	"CL_INVALID_KERNEL",
	"CL_INVALID_ARG_INDEX",
	"CL_INVALID_ARG_VALUE",
	"CL_INVALID_ARG_SIZE",
	"CL_INVALID_KERNEL_ARGS",
	"CL_INVALID_WORK_DIMENSION ",
	"CL_INVALID_WORK_GROUP_SIZE",
	"CL_INVALID_WORK_ITEM_SIZE",
	"CL_INVALID_GLOBAL_OFFSET",
	"CL_INVALID_EVENT_WAIT_LIST",
	"CL_INVALID_EVENT",
	"CL_INVALID_OPERATION",
	"CL_INVALID_GL_OBJECT",
	"CL_INVALID_BUFFER_SIZE",
	"CL_INVALID_MIP_LEVEL",
	"CL_INVALID_GLOBAL_WORK_SIZE"};


//! OpenCl Error checker
/*!
Checks for error code as per cl_int returned by OpenCl
\param status Error value as cl_int
\param msg User provided error message
\return True if Error Seen, False if no error
*/
int cl_errChk(const cl_int status, const char * msg)
{

	if(status != CL_SUCCESS) {
		printf("OpenCL Error: %d %s %s\n", status, cl_errs[-status], msg);
		return TRUE;
	}
	return FALSE;
}

//! Synchronous OpenCl Error checker
/*!
Checks for error code as per cl_int returned by OpenCl, Waits till all
commands finish before checking error code
\param status Error value as cl_int
\param msg User provided error message
\return True if Error Seen, False if no error
*/

int cl_errChk_sync(const cl_int status, const char * msg)
{
	cl_sync();
	if(status != CL_SUCCESS) {
		printf("OpenCL Error: %d %s %s\n", status, cl_errs[-status], msg);
		return TRUE;
	}
	return FALSE;
}


	void cl_printBinaries(cl_program program) {

		cl_uint program_num_devices;

		clGetProgramInfo( program,
			CL_PROGRAM_NUM_DEVICES,
			sizeof(cl_uint),
			&program_num_devices,
			NULL
			);

		printf("Number of devices: %d\n", program_num_devices);

		//size_t binaries_sizes[program_num_devices];
		size_t * binaries_sizes = (size_t *)malloc(sizeof(size_t)*program_num_devices);

		clGetProgramInfo( program,
			CL_PROGRAM_BINARY_SIZES,
			program_num_devices*sizeof(size_t),
			binaries_sizes,
			NULL
			);

		char** binaries = (char**)malloc(sizeof(char*)*program_num_devices);

		for (unsigned int i = 0; i < program_num_devices; i++)
			binaries[i] = (char*)malloc(sizeof(char)*(binaries_sizes[i]+1));

		clGetProgramInfo(program, CL_PROGRAM_BINARIES, program_num_devices*sizeof(size_t), binaries, NULL);

		for (unsigned int i = 0; i < program_num_devices; i++)
		{
			binaries[i][binaries_sizes[i]] = '\0';

			printf("Program %d\n", i);
			printf("%s\n", binaries[i]);

		}


		for (unsigned int i = 0; i < program_num_devices; i++)
			free(binaries[i]);

		free(binaries);
	}


	//! Initialize OpenCl environment on one device
	/*!
		Init function for one device. Looks for supported devices and creates a context
		\return returns a context initialized
	*/
cl_context cl_init_context(int platform, int dev,int quiet) {
    int printInfo=1;
    if (platform >= 0 && dev >= 0) printInfo = 0;
	cl_int status;
	// Used to iterate through the platforms and devices, respectively
	cl_uint numPlatforms;
	cl_uint numDevices;

	// These will hold the platform and device we select (can potentially be
	// multiple, but we're just doing one for now)
	// cl_platform_id platform = NULL;

	status = clGetPlatformIDs(0, NULL, &numPlatforms);
	if (printInfo) printf("Number of platforms detected:%d\n", numPlatforms);

	// Print some information about the available platforms
	cl_platform_id *platforms = NULL;
	cl_device_id * devices = NULL;
	if (numPlatforms > 0)
	{
		// get all the platforms
		platforms = (cl_platform_id*)malloc(numPlatforms *
			sizeof(cl_platform_id));
		status = clGetPlatformIDs(numPlatforms, platforms, NULL);

		// Traverse the platforms array
		if (printInfo) printf("Checking For OpenCl Compatible Devices\n");
		for(unsigned int i = 0; i < numPlatforms ; i++)
		{
			char pbuf[100];
			if (printInfo) printf("Platform %d:\t", i);
			status = clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR,
				sizeof(pbuf), pbuf, NULL);
			if (printInfo) printf("Vendor: %s\n", pbuf);

			//unsigned int numDevices;

			status = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
			if(cl_errChk(status, "checking for devices"))
				exit(1);
			if(numDevices == 0) {
				printf("There are no devices for Platform %d\n",i);
				exit(0);
			}
			else
			{
				if (printInfo) printf("\tNo of devices for Platform %d is %u\n",i, numDevices);
				//! Allocate an array of devices of size "numDevices"
				devices = (cl_device_id*)malloc(sizeof(cl_device_id)*numDevices);
				//! Populate Arrray with devices
				status = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, numDevices,
					devices, NULL);
				if(cl_errChk(status, "getting device IDs")) {
					exit(1);
				}
			}
			for( unsigned int j = 0; j < numDevices; j++)
			{
				char dbuf[100];
				char deviceStr[100];
				if (printInfo) printf("\tDevice: %d\t", j);
				status = clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR, sizeof(dbuf),
					deviceStr, NULL);
				cl_errChk(status, "Getting Device Info\n");
			    if (printInfo) printf("Vendor: %s", deviceStr);
				status = clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(dbuf),
					dbuf, NULL);
				if (printInfo) printf("\n\t\tName: %s\n", dbuf);
			}
		}
	}
	else
	{
		// If no platforms are available, we're sunk!
		printf("No OpenCL platforms found\n");
		exit(0);
	}

	int platform_touse;
	unsigned int device_touse;
	if (printInfo) printf("Enter Platform and Device No (Seperated by Space) \n");
	if (printInfo) scanf("%d %d", &platform_touse, &device_touse);
	else {
	  platform_touse = platform;
	  device_touse = dev;
	}
	if (!quiet) printf("Using Platform %d \t Device No %d \n",platform_touse, device_touse);

	//! Recheck how many devices does our chosen platform have
	status = clGetDeviceIDs(platforms[platform_touse], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);

	if(device_touse > numDevices)
	{
		printf("Invalid Device Number\n");
		exit(1);
	}
	
	//! Populate devices array with all the visible devices of our chosen platform
	devices = (cl_device_id *)malloc(sizeof(cl_device_id)*numDevices);
	status = clGetDeviceIDs(platforms[platform_touse],
					CL_DEVICE_TYPE_ALL, numDevices,
					devices, NULL);
	if(cl_errChk(status,"Error in Getting Devices\n")) exit(1);


	//!Check if Device requested is a CPU or a GPU
	cl_device_type dtype;
	device = devices[device_touse];
	status = clGetDeviceInfo(devices[device_touse],
					CL_DEVICE_TYPE,
					sizeof(dtype),
					(void *)&dtype,
					NULL);
	if(cl_errChk(status,"Error in Getting Device Info\n")) exit(1);
	if(dtype == CL_DEVICE_TYPE_GPU) {
	  if (!quiet) printf("Creating GPU Context\n\n");
	}
	else if (dtype == CL_DEVICE_TYPE_CPU) {
      if (!quiet) printf("Creating CPU Context\n\n");
	}
	else perror("This Context Type Not Supported\n");

	cl_context_properties cps[3] = {CL_CONTEXT_PLATFORM,
		(cl_context_properties)(platforms[platform_touse]), 0};

	cl_context_properties *cprops = cps;

	clGPUContext = clCreateContextFromType(
					cprops, (cl_device_type)dtype,
					NULL, NULL, &status);
	if(cl_errChk(status, "creating Context")) {
		exit(1);
	}

#define PROFILING

#ifdef PROFILING

	clCommandQueue = clCreateCommandQueue(clGPUContext,
						devices[device_touse], CL_QUEUE_PROFILING_ENABLE, &status);

#else

	clCommandQueue = clCreateCommandQueue(clGPUContext,
						devices[device_touse], NULL, &status);

#endif // PROFILING

	if(cl_errChk(status, "creating command queue")) {
		exit(1);
	}
	return clGPUContext;
}


void cl_TimeStart(cl_event event_time,
			cl_profiling_info profile_mode,
			char * event_name)
{
	cl_int kerneltimer;
	cl_ulong startTime;
	kerneltimer = clGetEventProfilingInfo(event_time,
		profile_mode,
		sizeof(cl_ulong), &startTime, NULL);
	if(cl_errChk(kerneltimer, "Error in Profiling\n"))exit(1);
	printf("\t%s\t %lu\n",event_name,(unsigned long)startTime);

}


void cl_TimeStop(cl_event event_time,
				cl_profiling_info profile_mode,
				char * event_name)
{
	cl_int kerneltimer;
	cl_ulong endTime;
	kerneltimer = clGetEventProfilingInfo(event_time,
		profile_mode,
		sizeof(cl_ulong), &endTime, NULL);
	if(cl_errChk(kerneltimer, "Error in Profiling\n"))exit(1);
	printf("%s\t%lu\n",event_name,(unsigned long)endTime);

}

//! Time kernel execution using cl_event
/*!
	Prints out the time taken between the start and end of an event
	\param event_time
*/
void cl_KernelTime(cl_event event_time)
{
	cl_int kerneltimer;
	cl_ulong starttime;
	cl_ulong endtime;

	kerneltimer = clGetEventProfilingInfo(event_time,
		CL_PROFILING_COMMAND_START,
		sizeof(cl_ulong), &starttime, NULL);

	if(cl_errChk(kerneltimer, "Error in Start Time \n"))exit(1);

	kerneltimer = clGetEventProfilingInfo(event_time,
		CL_PROFILING_COMMAND_END  ,
		sizeof(cl_ulong), &endtime, NULL);

	if(cl_errChk(kerneltimer, "Error in End Time \n"))exit(1);
	unsigned long elapsed =  (unsigned long)(endtime - starttime);
	printf("\tKernel Execution\t%ld ns\n",elapsed);
}

//! Synchronously Time kernel execution using cl_event
/*!
	Prints out the time taken between the start and end of an event.\n
	Adds synchronization in order to be sure that events have
	occured otherwise profiling calls will fail \n

	Shouldnt be used on critical path due to the necessary flushing of the queue
	\param event_time
*/
void cl_KernelTimeSync(cl_event event_time)
{
	cl_int kerneltimer;
	clFlush(cl_getCommandQueue());
	clFinish(cl_getCommandQueue());

	cl_ulong starttime;
	cl_ulong endtime;

	kerneltimer = clGetEventProfilingInfo(event_time,
		CL_PROFILING_COMMAND_START,
		sizeof(cl_ulong), &starttime, NULL);

	if(cl_errChk(kerneltimer, "Error in Start Time \n"))exit(1);

	kerneltimer = clGetEventProfilingInfo(event_time,
		CL_PROFILING_COMMAND_END  ,
		sizeof(cl_ulong), &endtime, NULL);

	if(cl_errChk(kerneltimer, "Error in Start Time \n"))exit(1);
	unsigned long elapsed =  (unsigned long)(endtime - starttime);
	printf("\tTime Elapsed in Kernel is %ld ns\n",elapsed);
}


//! Convert source code file into cl_program
/*!
Compile Opencl source file into a cl_program. The cl_program will be made into a kernel in Kernel_Precompile

\param kernelPath  Filename of OpenCl code
\param compileoptions Compilation options
\param verbosebuild Switch to enable verbose Output
*/
cl_program cl_CompileProgram(char * kernelPath, char * compileoptions, bool verbosebuild )
	{
		cl_int status;
		FILE *fp;
		char *source;
		long int size;

		//printf("Only Compiler Function: Kernel file is: %s\n", kernelPath);

		fp = fopen(kernelPath, "rb");
		if(!fp) {
			printf("Could not open kernel file\n");
			exit(-1);
		}
		status = fseek(fp, 0, SEEK_END);
		if(status != 0) {
			printf("Error seeking to end of file\n");
			exit(-1);
		}
		size = ftell(fp);
		//printf("size:**********************%ld\n",size);
		if(size < 0) {
			printf("Error getting file position\n");
			exit(-1);
		}
		/*status = fseek(fp, 0, SEEK_SET);
		if(status != 0) {
		printf("Error seeking to start of file\n");
		exit(-1);
		}*/
		rewind(fp);

		source = (char *)malloc(size + 1);
		// fill with NULLs
		for (int i=0;i<size+1;i++) source[i]='\0';
		if(source == NULL) {
			printf("Error allocating space for the kernel source\n");
			exit(-1);
		}

		//fread(source, size, 1, fp);   // TODO add error checking here
		fread(source,1,size,fp);
		source[size] = '\0';
		//printf("source:%s",source);
		cl_program clProgramReturn = clCreateProgramWithSource(clGPUContext, 1,
			(const char **)&source, NULL, &status);
		if(cl_errChk(status, "creating program")) {
			//       exit(1);
		}

		free(source);
		fclose(fp);

		status = clBuildProgram(clProgramReturn, 0, NULL,compileoptions, NULL, NULL);
		if(cl_errChk(status, "building program") || verbosebuild == 1)
		{

			cl_build_status build_status;

			clGetProgramBuildInfo(clProgramReturn, device, CL_PROGRAM_BUILD_STATUS,
				sizeof(cl_build_status), &build_status, NULL);

			if(build_status == CL_SUCCESS && verbosebuild == 0) {
				return clProgramReturn;
			}

			//char *build_log;
			size_t ret_val_size;
			printf("Device: %p",device);
			clGetProgramBuildInfo(clProgramReturn, device, CL_PROGRAM_BUILD_LOG, 0,
				NULL, &ret_val_size);

			char *build_log = (char *) malloc(ret_val_size+1);
			if(build_log == NULL){ printf("Couldnt Allocate Build Log of Size %zu \n",ret_val_size); exit(1);}

			clGetProgramBuildInfo(clProgramReturn, device, CL_PROGRAM_BUILD_LOG,
				ret_val_size+1, build_log, NULL);

			printf("After build log call\n");
			// to be careful, terminate with \0
			// there's no information in the reference whether the string is 0
			// terminated or not
			build_log[ret_val_size] = '\0';

			printf("Build log:\n %s...\n", build_log);
			if(build_status != CL_SUCCESS) {
				exit(1);
			}
			else
				return clProgramReturn;
		}

		// print the ptx information
		//   cl_printBinaries(clProgram);
		//    printf("Done Compiling the Program\n");
		return clProgramReturn;
	}



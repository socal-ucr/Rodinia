/* UPDATE
   ------------------------------
   2009.12 Lukasz G. Szafaryn
   -- entire code written
   2010.01 Lukasz G. Szafaryn
   -- comments

   DESCRIPTION
   ------------------------------
   The Heart Wall application tracks the movement of a mouse heart over a sequence
   of 104 609x590 ultrasound images to record response to the stimulus. In its
   initial stage, the program performs image processing operations on the first
   image to detect initial, partial shapes of inner and outer heart walls. These
   operations include: edge detection, SRAD despeckling (also part of Rodinia suite),
   morphological transformation and dilation. In order to reconstruct approximated full
   shapes of heart walls, the program generates ellipses that are superimposed over
   the image and sampled to mark points on the heart walls (Hough Search). In its
   final stage (Heart Wall Tracking presented here), program tracks movement of
   surfaces by detecting the movement of image areas under sample points as the shapes
   of the heart walls change throughout the sequence of images.

   Tracking is the final stage of the Heart Wall application. It takes the positions of
   heart walls from the first ultrasound image in the sequence as determined by the
   initial detection stage in the application. Tracking code is implemented in the form
   of multiple nested loops that process batches of 10 frames and 51 points in each image.
   Displacement of heart walls is detected by comparing currently processed frame to the
   template frame which is updated after processing a batch of frames. There is a sequential
   dependency between processed frames. The processing of each point consist of a large
   number of small serial steps with interleaved control statements. Each of the steps
   involves a small amount of computation performed only on a subset of entire image.
   This stage of the application accounts for almost all of the execution time (the exact
   ratio depends on the number of ultrasound images).

   For more information, see:

   Papers:
   ~ L. G. Szafaryn, K. Skadron, and J. J. Saucerman. "Experiences Accelerating MATLAB Systems
   Biology Applications." In Proceedings of the Workshop on Biomedicine in Computing: Systems,
   Architectures, and Circuits (BiC) 2009, in conjunction with the 36th IEEE/ACM International
   Symposium on Computer Architecture (ISCA), June 2009.
   <http://www.cs.virginia.edu/~skadron/Papers/BiC09.pdf>


   WHERE TO GET THE PROGRAM
   ------------------------------------------------------------------------------------------

   Download:
   Rodinia Benchmark Suite
   <https://www.cs.virginia.edu/~skadron/wiki/rodinia/index.php/Main_Page>

   The code takes the following input files that need to be located in the
   same directory as the source files:
   1) video file (input.avi)
   2) text file with parameters (input.txt)


   PARAMETERS
   ------------------------------------------------------------------------------------------
   The following are the command parameters to the application:
   1) Number of frames to process. Needs to be integer <= to the number of frames in the input file.

   Example:
   a.out 104

*/

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "OpenCL.h"
#include "avilib.h"
#include "avimod.h"
#include "define.c"
#include "file.c"
#include "timer.c"

int main(int argc, char *argv[])
{
	// time
	long long time0;
	long long time1;
	long long time2;
	long long time3;
	long long time4;
	long long time5;
	long long time6;
	long long time7;
	long long time8;
	long long time9;
	long long time10;
	long long time11;

	time0 = get_time();

	// Counter.
	int i;
	int frames_processed;

	// Frames.
	char* video_file_name;
	avi_t* frames;
	float* frame;

	time1 = get_time();

	// Open the movie file.
	video_file_name = (char*) "../../data/heartwall/test.avi";
	frames = (avi_t*) AVI_open_input_file(video_file_name, 1);
	if (frames == NULL)
	{
		AVI_print_error((char*) "Error with AVI_open_input_file");
		return -1;
	}

	// Common.
	params_common common;
	params_common_change common_change;
	common.no_frames = AVI_video_frames(frames);
	common.frame_rows = AVI_video_height(frames);
	common.frame_cols = AVI_video_width(frames);
	common.frame_elem = common.frame_rows * common.frame_cols;
	common.frame_mem = sizeof(float) * common.frame_elem;
	size_t globalWorkSize = common.frame_elem;      // Holds the size, for OpenCL purposes.

	// Create and initialize the OpenCL object.
	OpenCL cl(1);       // 1 means to display output (debugging mode).
	cl.init(1);         // 1 means to use GPU. 0 means use CPU.
	cl.gwSize(globalWorkSize);

	// Create and build the kernel.
	string kn = "hwKernel";     // the kernel name, for future use.
	cl.createKernel(kn);

	// Create the memory on the device that will hold each frame.
	common_change.d_frame = clCreateBuffer(cl.ctxt(),                            // The OpenCL context.
	                                       CL_MEM_READ_WRITE,                    // Memory flags. Enable read/write.
	                                       sizeof(cl_float)*common.frame_elem,   // The device memory size.
	                                       NULL,                                 // NULL - nothing to copy.
	                                       NULL);                                // A pointer to a variable to return an error code into. None in this case.


	// Structures, global structure variables.
	// Problem here: cannot dynamically allocate constant memory in GPU,
	// so have to use constant value here.
	struct_common_change d_common_change;
	struct_common d_common;
	params_unique unique[ALL_POINTS];
	struct_unique d_unique[ALL_POINTS];

	time2 = get_time();

	// Check the input arguments...
	if (argc != 2)
	{
		printf("ERROR: missing argument (number of frames to processed) or too many arguments\n");
		return 0;
	}
	else
	{
		frames_processed = atoi(argv[1]);
		if (frames_processed < 0 || frames_processed > common.no_frames)
		{
			printf("ERROR: %d is an incorrect number of frames specified, select in the range of 0-%d\n",
			       frames_processed, common.no_frames);
			return 0;
		}
	}

	time3 = get_time();

	// Read parameters from the input file.
	read_parameters((char*) "../../data/heartwall/input.txt", &common.tSize, &common.sSize, &common.maxMove, &common.alpha);
	read_header((char*) "../../data/heartwall/input.txt", &common.endoPoints, &common.epiPoints);

	common.allPoints = common.endoPoints + common.epiPoints;

	time4 = get_time();

	// Endo points memory allocation.
	common.endo_mem = sizeof(int) * common.endoPoints;
	common.endoRow = (int *) malloc(common.endo_mem);

	// TODO: Delete this... cudaMalloc((void **) &common.d_endoRow, common.endo_mem);
	common.d_endoRow = clCreateBuffer(cl.ctxt(),                               // The OpenCL context.
	                                  CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, // Memory flags. Enable read/write.
	                                  sizeof(cl_int) * common.endoPoints,      // The device memory size.
	                                  common.endoRow,                          // The memory on the host to copy over.
	                                  NULL);                                   // A pointer to a variable to return an error code into. None in this case.
	
	common.endoCol = (int *) malloc(common.endo_mem);

	// TODO: Delete this... cudaMalloc((void **) &common.d_endoCol, common.endo_mem);
	common.d_endoCol = clCreateBuffer(cl.ctxt(),                               // The OpenCL context.
	                                  CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, // Memory flags. Enable read/write.
	                                  sizeof(cl_int) * common.endoPoints,      // The device memory size.
	                                  common.endoCol,                          // The memory on the host to copy over.
	                                  NULL);                                   // A pointer to a variable to return an error code into. None in this case.
	
	common.tEndoRowLoc = (int *) malloc(common.endo_mem * common.no_frames);

	// TODO: Delete this... cudaMalloc((void **) &common.d_tEndoRowLoc, common.endo_mem * common.no_frames);
	common.d_tEndoRowLoc = clCreateBuffer(cl.ctxt(),                                             // The OpenCL context.
	                                      CL_MEM_READ_WRITE,                                     // Memory flags. Enable read/write.
	                                      sizeof(cl_int) * common.endoPoints * common.no_frames, // The device memory size.
	                                      NULL,                                                  // NULL - nothing to copy.
	                                      NULL);                                                 // A pointer to a variable to return an error code into. None in this case.
	common.tEndoColLoc = (int *) malloc(common.endo_mem * common.no_frames);

	// TODO: Delete this... cudaMalloc((void **) &common.d_tEndoColLoc, common.endo_mem * common.no_frames);
	common.d_tEndoColLoc = clCreateBuffer(cl.ctxt(),                                             // The OpenCL context.
	                                      CL_MEM_READ_WRITE,                                     // Memory flags. Enable read/write.
	                                      sizeof(cl_int) * common.endoPoints * common.no_frames, // The device memory size.
	                                      NULL,                                                  // NULL - nothing to copy.
	                                      NULL);                                                 // NULL - no error code used.

	// EPI points memory allocation
	common.epi_mem = sizeof(int) * common.epiPoints;
	common.epiRow = (int *) malloc(common.epi_mem);

	common.d_epiRow = clCreateBuffer(cl.ctxt(),                               // The OpenCL context.
	                                 CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, // Memory flags. Enable read/write.
	                                 sizeof(cl_int) * common.epiPoints,       // The device memory size.
	                                 common.epiRow,                           // The memory on the host to copy over.
	                                 NULL);                                   // NULL - no error code used.

	common.epiCol = (int *) malloc(common.epi_mem);

	common.d_epiCol = clCreateBuffer(cl.ctxt(),                               // The OpenCL context.
	                                 CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, // Memory flags. Enable read/write.
	                                 sizeof(cl_int) * common.epiPoints,       // The device memory size.
	                                 common.epiCol,                           // The memory on the host to copy over.
	                                 NULL);                                   // NULL - no error code used.

	common.tEpiRowLoc = (int *) malloc(common.epi_mem * common.no_frames);

	common.d_tEpiRowLoc = clCreateBuffer(cl.ctxt(),                                            // The OpenCL context.
	                                     CL_MEM_READ_WRITE,                                    // Memory flags. Enable read/write.
	                                     sizeof(cl_int) * common.epiPoints * common.no_frames, // The device memory size.
	                                     NULL,                                                 // NULL - nothing to copy.
	                                     NULL);                                                // NULL - no error code used.

	common.tEpiColLoc = (int *) malloc(common.epi_mem * common.no_frames);
	common.d_tEpiColLoc = clCreateBuffer(cl.ctxt(),                                            // The OpenCL context.
	                                     CL_MEM_READ_WRITE,                                    // Memory flags. Enable read/write.
	                                     sizeof(cl_int) * common.epiPoints * common.no_frames, // The device memory size.
	                                     NULL,                                                 // NULL - nothing to copy.
	                                     NULL);                                                // NULL - no error code used.

	time5 = get_time();
	
	// Read data from file.
	read_data((char*)"../../data/heartwall/input.txt", common.endoPoints, common.endoRow, common.endoCol, common.epiPoints, common.epiRow,
	          common.epiCol);

	time6 = get_time();

	// Template sizes:

	// Common.
	common.in_rows = common.tSize + 1 + common.tSize;
	common.in_cols = common.in_rows;
	common.in_elem = common.in_rows * common.in_cols;     // Total number of elements in a frame.

	d_common.in_rows = common.tSize + 1 + common.tSize;
	d_common.in_cols = common.in_rows;
	d_common.in_elem = common.in_rows * common.in_cols;   // Total number of elements in a frame.

	common.in_mem = sizeof(float) * common.in_elem;

	// Create array of templates for all points:

	// Common.
	common.d_endoT = clCreateBuffer(cl.ctxt(),                                             // The OpenCL context.
	                                CL_MEM_READ_WRITE,                                     // Memory flags. Enable read/write.
	                                sizeof(cl_float) * common.in_elem * common.endoPoints, // The device memory size.
	                                NULL,                                                  // NULL - nothing to copy.
	                                NULL);                                                 // NULL - no error code used.

	common.d_epiT = clCreateBuffer(cl.ctxt(),                                             // The OpenCL context.
	                               CL_MEM_READ_WRITE,                                     // Memory flags. Enable read/write.
	                               sizeof(cl_float) * common.in_elem * common.endoPoints, // The device memory size.
	                               NULL,                                                  // NULL - nothing to copy.
	                               NULL);                                                 // NULL - no error code used.

	// Specific to endo or epi to be set here.
	for (i = 0; i < common.endoPoints; i++)
	{
		unique[i].point_no = i;
		d_unique[i].point_no = i;
	}
	for (i = common.endoPoints; i < common.allPoints; i++)
	{
		unique[i].point_no = i - common.endoPoints;
		d_unique[i].point_no = i - common.endoPoints;
	}

	// Right template from template array.
	// pointers
	int* h_in_pointer = (int*)malloc(common.allPoints * sizeof(int));
	for (i = 0; i < common.allPoints; i++)
	{
		// TODO:
		// unique[i].in_pointer = unique[i].point_no * common.in_elem;
		h_in_pointer[i] = unique[i].point_no * common.in_elem;
	}
	cl_mem d_in_pointer = clCreateBuffer(cl.ctxt(),                                // The OpenCL context.
	                                     CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,  // Memory flags. Enable read/write.
	                                     sizeof(int) * common.allPoints,           // The device memory size.
	                                     h_in_pointer,                             // NULL - nothing to copy.
	                                     NULL);                                    // NULL - no error code used.

	// Area around point from frame.
	// common
	common.in2_rows = 2 * common.sSize + 1;
	common.in2_cols = 2 * common.sSize + 1;
	common.in2_elem = common.in2_rows * common.in2_cols;
	d_common.in2_rows = 2 * common.sSize + 1;
	d_common.in2_cols = 2 * common.sSize + 1;
	d_common.in2_elem = common.in2_rows * common.in2_cols;
	common.in2_mem = sizeof(float) * common.in2_elem;

	// ************************************************************************************
	// POINTERS
	// ************************************************************************************
	// NOTE: Since this was all cudaMalloc'd in a loop originally, we need to create a 
	//       memory buffer on the device which has the total size from the for-loop,
	//       then use the for-loop to copy into the large buffer using clEnqueueWriteBuffer
	//       with offsets determined by the for-loop.
	int d_in2_memSize = sizeof(cl_float) * common.in2_elem * common.allPoints;
	unique[0].d_in2 = clCreateBuffer(cl.ctxt(),                 // The OpenCL context.
	                                 CL_MEM_READ_WRITE,         // Memory flags. Enable read/write.
	                                 d_in2_memSize,             // The device memory size.
	                                 NULL,                      // NULL - nothing to copy.
	                                 NULL);                     // NULL - no error code used.

	// Convolution...

	// common
	common.conv_rows = common.in_rows + common.in2_rows - 1;   // number of rows in I
	common.conv_cols = common.in_cols + common.in2_cols - 1;   // number of columns in I
	common.conv_elem = common.conv_rows * common.conv_cols;    // number of elements
	d_common.conv_rows = common.in_rows + common.in2_rows - 1; // number of rows in I
	d_common.conv_cols = common.in_cols + common.in2_cols - 1; // number of columns in I
	d_common.conv_elem = common.conv_rows * common.conv_cols;  // number of elements
	common.conv_mem = sizeof(float) * common.conv_elem;
	common.ioffset = 0;
	common.joffset = 0;
	d_common.ioffset = 0;
	d_common.joffset = 0;

	// pointers
	int d_conv_memSize = sizeof(cl_float) * common.conv_elem * common.allPoints;
	unique[0].d_conv = clCreateBuffer(cl.ctxt(),                // The OpenCL context.
	                                  CL_MEM_READ_WRITE,        // Memory flags. Enable read/write.
	                                  d_conv_memSize,           // The device memory size.
	                                  NULL,                     // NULL - nothing to copy.
	                                  NULL);                    // NULL - no error code used.

	// Cumulative sum...

	// Padding of array, vertical cumulative sum.

	// common
	common.in2_pad_add_rows = common.in_rows;
	common.in2_pad_add_cols = common.in_cols;
	common.in2_pad_cumv_rows = common.in2_rows + 2* common .in2_pad_add_rows;
	common.in2_pad_cumv_cols = common.in2_cols + 2* common .in2_pad_add_cols;
	common.in2_pad_cumv_elem = common.in2_pad_cumv_rows * common.in2_pad_cumv_cols;
	d_common.in2_pad_add_rows = common.in_rows;
	d_common.in2_pad_add_cols = common.in_cols;
	d_common.in2_pad_cumv_rows = common.in2_rows + 2* common .in2_pad_add_rows;
	d_common.in2_pad_cumv_cols = common.in2_cols + 2* common .in2_pad_add_cols;
	d_common.in2_pad_cumv_elem = common.in2_pad_cumv_rows * common.in2_pad_cumv_cols;
	common.in2_pad_cumv_mem = sizeof(float) * common.in2_pad_cumv_elem;

	// pointers
	int d_in2_pad_cumv_size = sizeof(cl_float) * common.in2_pad_cumv_elem * common.allPoints;
	unique[0].d_in2_pad_cumv = clCreateBuffer(cl.ctxt(),             // The OpenCL context.
	                                          CL_MEM_READ_WRITE,     // Memory flags. Enable read/write.
	                                          d_in2_pad_cumv_size,   // The device memory size.
	                                          NULL,                  // NULL - nothing to copy.
	                                          NULL);                 // NULL - no error code used.

	// Selection...
	
	// common
	// (1 to n+1)
	common.in2_pad_cumv_sel_rowlow = 1 + common.in_rows;
	common.in2_pad_cumv_sel_rowhig = common.in2_pad_cumv_rows - 1;
	common.in2_pad_cumv_sel_collow = 1;
	common.in2_pad_cumv_sel_colhig = common.in2_pad_cumv_cols;
	common.in2_pad_cumv_sel_rows = common.in2_pad_cumv_sel_rowhig - common.in2_pad_cumv_sel_rowlow + 1;
	common.in2_pad_cumv_sel_cols = common.in2_pad_cumv_sel_colhig - common.in2_pad_cumv_sel_collow + 1;
	common.in2_pad_cumv_sel_elem = common.in2_pad_cumv_sel_rows * common.in2_pad_cumv_sel_cols;
	d_common.in2_pad_cumv_sel_rowlow = 1 + common.in_rows;
	d_common.in2_pad_cumv_sel_rowhig = common.in2_pad_cumv_rows - 1;
	d_common.in2_pad_cumv_sel_collow = 1;
	d_common.in2_pad_cumv_sel_colhig = common.in2_pad_cumv_cols;
	d_common.in2_pad_cumv_sel_rows = common.in2_pad_cumv_sel_rowhig - common.in2_pad_cumv_sel_rowlow + 1;
	d_common.in2_pad_cumv_sel_cols = common.in2_pad_cumv_sel_colhig - common.in2_pad_cumv_sel_collow + 1;
	d_common.in2_pad_cumv_sel_elem = common.in2_pad_cumv_sel_rows * common.in2_pad_cumv_sel_cols;
	common.in2_pad_cumv_sel_mem = sizeof(float) * common.in2_pad_cumv_sel_elem;

	// pointers
	int	d_in2_pad_cumv_sel_memSize = sizeof(cl_float) * common.in2_pad_cumv_sel_elem * common.allPoints;
	unique[0].d_in2_pad_cumv_sel = clCreateBuffer(cl.ctxt(),                   // The OpenCL context.
	                                              CL_MEM_READ_WRITE,           // Memory flags. Enable read/write.
	                                              d_in2_pad_cumv_sel_memSize,  // The device memory size.
	                                              NULL,                        // NULL - nothing to copy.
	                                              NULL);                       // NULL - no error code used.

	// Selection 2, subtraction, horizontal cumulative sum.

	// common
	common.in2_pad_cumv_sel2_rowlow = 1;
	common.in2_pad_cumv_sel2_rowhig = common.in2_pad_cumv_rows - common.in_rows - 1;
	common.in2_pad_cumv_sel2_collow = 1;
	common.in2_pad_cumv_sel2_colhig = common.in2_pad_cumv_cols;
	common.in2_sub_cumh_rows = common.in2_pad_cumv_sel2_rowhig - common.in2_pad_cumv_sel2_rowlow + 1;
	common.in2_sub_cumh_cols = common.in2_pad_cumv_sel2_colhig - common.in2_pad_cumv_sel2_collow + 1;
	common.in2_sub_cumh_elem = common.in2_sub_cumh_rows * common.in2_sub_cumh_cols;
	d_common.in2_pad_cumv_sel2_rowlow = 1;
	d_common.in2_pad_cumv_sel2_rowhig = common.in2_pad_cumv_rows - common.in_rows - 1;
	d_common.in2_pad_cumv_sel2_collow = 1;
	d_common.in2_pad_cumv_sel2_colhig = common.in2_pad_cumv_cols;
	d_common.in2_sub_cumh_rows = common.in2_pad_cumv_sel2_rowhig - common.in2_pad_cumv_sel2_rowlow + 1;
	d_common.in2_sub_cumh_cols = common.in2_pad_cumv_sel2_colhig - common.in2_pad_cumv_sel2_collow + 1;
	d_common.in2_sub_cumh_elem = common.in2_sub_cumh_rows * common.in2_sub_cumh_cols;
	common.in2_sub_cumh_mem = sizeof(float) * common.in2_sub_cumh_elem;

	// pointers
	int d_in2_sub_cumh_memSize = sizeof(cl_float) * common.in2_sub_cumh_elem * common.allPoints;
	unique[0].d_in2_sub_cumh = clCreateBuffer(cl.ctxt(),              // The OpenCL context.
	                                          CL_MEM_READ_WRITE,      // Memory flags. Enable read/write.
	                                          d_in2_sub_cumh_memSize, // The device memory size.
	                                          NULL,                   // NULL - nothing to copy.
	                                          NULL);                  // NULL - no error code used.


	// Selection

	// common
	common.in2_sub_cumh_sel_rowlow = 1;
	common.in2_sub_cumh_sel_rowhig = common.in2_sub_cumh_rows;
	common.in2_sub_cumh_sel_collow = 1 + common.in_cols;
	common.in2_sub_cumh_sel_colhig = common.in2_sub_cumh_cols - 1;
	common.in2_sub_cumh_sel_rows = common.in2_sub_cumh_sel_rowhig - common.in2_sub_cumh_sel_rowlow + 1;
	common.in2_sub_cumh_sel_cols = common.in2_sub_cumh_sel_colhig - common.in2_sub_cumh_sel_collow + 1;
	common.in2_sub_cumh_sel_elem = common.in2_sub_cumh_sel_rows * common.in2_sub_cumh_sel_cols;

	d_common.in2_sub_cumh_sel_rowlow = 1;
	d_common.in2_sub_cumh_sel_rowhig = common.in2_sub_cumh_rows;
	d_common.in2_sub_cumh_sel_collow = 1 + common.in_cols;
	d_common.in2_sub_cumh_sel_colhig = common.in2_sub_cumh_cols - 1;
	d_common.in2_sub_cumh_sel_rows = common.in2_sub_cumh_sel_rowhig - common.in2_sub_cumh_sel_rowlow + 1;
	d_common.in2_sub_cumh_sel_cols = common.in2_sub_cumh_sel_colhig - common.in2_sub_cumh_sel_collow + 1;
	d_common.in2_sub_cumh_sel_elem = common.in2_sub_cumh_sel_rows * common.in2_sub_cumh_sel_cols;
	
	common.in2_sub_cumh_sel_mem = sizeof(float) * common.in2_sub_cumh_sel_elem;

	// pointers
	int d_in2_sub_cumh_sel_memSize = sizeof(cl_float) * common.in2_sub_cumh_sel_elem * common.allPoints;
	unique[0].d_in2_sub_cumh_sel = clCreateBuffer(cl.ctxt(),                  // The OpenCL context.
	                                              CL_MEM_READ_WRITE,          // Memory flags. Enable read/write.
	                                              d_in2_sub_cumh_sel_memSize, // The device memory size.
	                                              NULL,                       // NULL - nothing to copy.
	                                              NULL);                      // NULL - no error code used.

	// Selection 2, subtraction.

	// common
	common.in2_sub_cumh_sel2_rowlow = 1;
	common.in2_sub_cumh_sel2_rowhig = common.in2_sub_cumh_rows;
	common.in2_sub_cumh_sel2_collow = 1;
	common.in2_sub_cumh_sel2_colhig = common.in2_sub_cumh_cols - common.in_cols - 1;
	common.in2_sub2_rows = common.in2_sub_cumh_sel2_rowhig - common.in2_sub_cumh_sel2_rowlow + 1;
	common.in2_sub2_cols = common.in2_sub_cumh_sel2_colhig - common.in2_sub_cumh_sel2_collow + 1;
	common.in2_sub2_elem = common.in2_sub2_rows * common.in2_sub2_cols;

	d_common.in2_sub_cumh_sel2_rowlow = 1;
	d_common.in2_sub_cumh_sel2_rowhig = common.in2_sub_cumh_rows;
	d_common.in2_sub_cumh_sel2_collow = 1;
	d_common.in2_sub_cumh_sel2_colhig = common.in2_sub_cumh_cols - common.in_cols - 1;
	d_common.in2_sub2_rows = common.in2_sub_cumh_sel2_rowhig - common.in2_sub_cumh_sel2_rowlow + 1;
	d_common.in2_sub2_cols = common.in2_sub_cumh_sel2_colhig - common.in2_sub_cumh_sel2_collow + 1;
	d_common.in2_sub2_elem = common.in2_sub2_rows * common.in2_sub2_cols;
	
	common.in2_sub2_mem = sizeof(float) * common.in2_sub2_elem;

	// pointers
	int d_in2_sub2_memSize = sizeof(cl_float) * common.in2_sub2_elem * common.allPoints;
	unique[0].d_in2_sub2 = clCreateBuffer(cl.ctxt(),          // The OpenCL context.
	                                      CL_MEM_READ_WRITE,  // Memory flags. Enable read/write.
	                                      d_in2_sub2_memSize, // The device memory size.
	                                      NULL,               // NULL - nothing to copy.
	                                      NULL);              // NULL - no error code used.

	// Cumulative sum 2:
	// Multiplication...

	// common
	common.in2_sqr_rows = common.in2_rows;
	common.in2_sqr_cols = common.in2_cols;
	common.in2_sqr_elem = common.in2_elem;

	d_common.in2_sqr_rows = common.in2_rows;
	d_common.in2_sqr_cols = common.in2_cols;
	d_common.in2_sqr_elem = common.in2_elem;
	
	common.in2_sqr_mem = common.in2_mem;

	// pointers
	int d_in2_sqr_memSize = sizeof(cl_float) * common.in2_sqr_elem * common.allPoints;
	unique[0].d_in2_sqr = clCreateBuffer(cl.ctxt(),                           // The OpenCL context.
	                                     CL_MEM_READ_WRITE,                   // Memory flags. Enable read/write.
	                                     d_in2_sqr_memSize,                   // The device memory size.
	                                     NULL,                                // NULL - nothing to copy.
	                                     NULL);                               // NULL - no error code used.

	// Selection 2, subtraction

	// common
	common.in2_sqr_sub2_rows = common.in2_sub2_rows;
	common.in2_sqr_sub2_cols = common.in2_sub2_cols;
	common.in2_sqr_sub2_elem = common.in2_sub2_elem;

	d_common.in2_sqr_sub2_rows = common.in2_sub2_rows;
	d_common.in2_sqr_sub2_cols = common.in2_sub2_cols;
	d_common.in2_sqr_sub2_elem = common.in2_sub2_elem;
	
	common.in2_sqr_sub2_mem = common.in2_sub2_mem;

	// pointers
	int d_in2_sqr_sub2_memSize = sizeof(cl_float) * common.in2_sqr_sub2_elem * common.allPoints;
	unique[0].d_in2_sqr_sub2 = clCreateBuffer(cl.ctxt(),              // The OpenCL context.
	                                          CL_MEM_READ_WRITE,      // Memory flags. Enable read/write.
	                                          d_in2_sqr_sub2_memSize, // The device memory size.
	                                          NULL,                   // NULL - nothing to copy.
	                                          NULL);                  // NULL - no error code used.

	// Final!
	// common
	common.in_sqr_rows = common.in_rows;
	common.in_sqr_cols = common.in_cols;
	common.in_sqr_elem = common.in_elem;

	d_common.in_sqr_rows = common.in_rows;
	d_common.in_sqr_cols = common.in_cols;
	d_common.in_sqr_elem = common.in_elem;
	
	common.in_sqr_mem = common.in_mem;

	// pointers
	unique[0].d_in_sqr = clCreateBuffer(cl.ctxt(),                                                // The OpenCL context.
	                                    CL_MEM_READ_WRITE,                                        // Memory flags. Enable read/write.
	                                    sizeof(cl_float) * common.in_sqr_elem * common.allPoints, // The device memory size.
	                                    NULL,                                                     // NULL - nothing to copy.
	                                    NULL);                                                    // NULL - no error code used.

	// Template mask create.

	// Common
	common.tMask_rows = common.in_rows + (common.sSize + 1 + common.sSize) - 1;
	common.tMask_cols = common.tMask_rows;
	common.tMask_elem = common.tMask_rows * common.tMask_cols;

	d_common.tMask_rows = common.in_rows + (common.sSize + 1 + common.sSize) - 1;
	d_common.tMask_cols = common.tMask_rows;
	d_common.tMask_elem = common.tMask_rows * common.tMask_cols;
	
	common.tMask_mem = sizeof(float) * common.tMask_elem;

	// Pointers
	unique[0].d_tMask = clCreateBuffer(cl.ctxt(),                                               // The OpenCL context.
	                                   CL_MEM_READ_WRITE,                                       // Memory flags. Enable read/write.
	                                   sizeof(cl_float) * common.tMask_elem * common.allPoints, // The device memory size.
	                                   NULL,                                                    // NULL - nothing to copy.
	                                   NULL);                                                   // NULL - no error code used.

	// Point mask initialize.

	// common
	common.mask_rows = common.maxMove;
	common.mask_cols = common.mask_rows;
	common.mask_elem = common.mask_rows * common.mask_cols;

	d_common.mask_rows = common.maxMove;
	d_common.mask_cols = common.mask_rows;
	d_common.mask_elem = common.mask_rows * common.mask_cols;
	
	common.mask_mem = sizeof(float) * common.mask_elem;

	// Mask convolution.

	// common
	common.mask_conv_rows = common.tMask_rows;                               // number of rows in I
	common.mask_conv_cols = common.tMask_cols;                               // number of columns in I
	common.mask_conv_elem = common.mask_conv_rows * common.mask_conv_cols;   // number of elements

	d_common.mask_conv_rows = common.tMask_rows;                             // number of rows in I
	d_common.mask_conv_cols = common.tMask_cols;                             // number of columns in I
	d_common.mask_conv_elem = common.mask_conv_rows * common.mask_conv_cols; // number of elements
	
	common.mask_conv_mem = sizeof(float) * common.mask_conv_elem;
	common.mask_conv_ioffset = (common.mask_rows - 1) / 2;
	if ((common.mask_rows - 1) % 2 > 0.5)
	{
		common.mask_conv_ioffset = common.mask_conv_ioffset + 1;
		d_common.mask_conv_ioffset = common.mask_conv_ioffset + 1;
	}
	common.mask_conv_joffset = (common.mask_cols - 1) / 2;
	if ((common.mask_cols - 1) % 2 > 0.5)
	{
		common.mask_conv_joffset = common.mask_conv_joffset + 1;
		d_common.mask_conv_joffset = common.mask_conv_joffset + 1;
	}

	// pointers
	unique[0].d_mask_conv = clCreateBuffer(cl.ctxt(),                                                   // The OpenCL context.
	                                       CL_MEM_READ_WRITE,                                           // Memory flags. Enable read/write.
	                                       sizeof(cl_float) * common.mask_conv_elem * common.allPoints, // The device memory size.
	                                       NULL,                                                        // NULL - nothing to copy.
	                                       NULL);                                                       // NULL - no error code used.


	// Kernel.
	// Thread block.

	// All kernels operations within kernel use same max size of threads.
	// Size of block size is set to the size appropriate for max size
	// operation (on padded matrix). Other use subsets of that.

	time7 = get_time();

	// Copy arguments.

	time8 = get_time();

	/* Print frame progress start. */
	// printf("frame progress: ");
	// fflush(NULL);

	// Some final memory buffer creation:
	cl_mem mem_common_change = clCreateBuffer(cl.ctxt(),                                // The OpenCL context.
	                                          CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,   // Memory flags. Enable read/write.
	                                          sizeof(struct_common_change),             // The device memory size.
	                                          &d_common_change,                         // Pointer to host memory to copy.
	                                          NULL);                                    // NULL - no error code used.

	cl_mem mem_common = clCreateBuffer(cl.ctxt(),                                // The OpenCL context.
	                                   CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,   // Memory flags. Enable read/write.
	                                   sizeof(struct_common),                    // The device memory size.
	                                   &d_common,                                // Pointer to host memory to copy.
	                                   NULL);                                    // NULL - no error code used.
		
	cl_mem mem_unique = clCreateBuffer(cl.ctxt(),                                // The OpenCL context.
	                                   CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,   // Memory flags. Enable read/write.
	                                   sizeof(struct_unique)*common.allPoints,   // The device memory size.
	                                   d_unique,                                 // Pointer to host memory to copy.
	                                   NULL);                                    // NULL - no error code used.

	int localWorksize = cl.localSize();
	
	/*********************
	 * LAUNCH THE KERNEL *
	 *********************/
	
	for (common_change.frame_no = 0; common_change.frame_no < frames_processed; common_change.frame_no++)
	{
		// Extract a cropped version of the first frame from the video file
		frame = get_frame(frames,                 // pointer to video file
		                  common_change.frame_no, // number of frame that needs to be returned
		                  0,                      // cropped?
		                  0,                      // scaled?
		                  1);                     // converted

		// TODO: Delete this. cudaMemcpy(common_change.d_frame, frame, common.frame_mem, cudaMemcpyHostToDevice);
		clEnqueueWriteBuffer(cl.q(),                                 // The queue.
		                     common_change.d_frame,                  // The frame memory on the device.
		                     CL_TRUE,                                // Blocking? Ie, wait until done copying to move past this line on the host?
		                     0,                                      // Offset.
		                     sizeof(cl_float)*common.frame_elem,     // Size to copy.
		                     frame,                                  // The frame memory on the host.
		                     0,                                      // Number of events in wait list. Not used.
		                     NULL,                                   // The wait list. Not used.
		                     NULL);                                  // Event. Not used.

		// launch GPU kernel
		// Set the kernel arguments.
		clSetKernelArg(cl.kernel(kn), 0,  sizeof(cl_mem),   (void*) &common_change.d_frame);
		clSetKernelArg(cl.kernel(kn), 1,  sizeof(cl_mem),   (void*) &common.d_endoRow);
		clSetKernelArg(cl.kernel(kn), 2,  sizeof(cl_mem),   (void*) &common.d_endoCol);
		clSetKernelArg(cl.kernel(kn), 3,  sizeof(cl_mem),   (void*) &common.d_tEndoRowLoc);
		clSetKernelArg(cl.kernel(kn), 4,  sizeof(cl_mem),   (void*) &common.d_tEndoColLoc);
		clSetKernelArg(cl.kernel(kn), 5,  sizeof(cl_mem),   (void*) &common.d_epiRow);
		clSetKernelArg(cl.kernel(kn), 6,  sizeof(cl_mem),   (void*) &common.d_epiCol);
		clSetKernelArg(cl.kernel(kn), 7,  sizeof(cl_mem),   (void*) &common.d_tEpiRowLoc);
		clSetKernelArg(cl.kernel(kn), 8,  sizeof(cl_mem),   (void*) &common.d_tEpiColLoc);
		clSetKernelArg(cl.kernel(kn), 9,  sizeof(cl_mem),   (void*) &common.d_endoT);
		clSetKernelArg(cl.kernel(kn), 10, sizeof(cl_mem),   (void*) &common.d_epiT);
		clSetKernelArg(cl.kernel(kn), 11, sizeof(cl_mem),   (void*) &unique[0].d_in2);
		clSetKernelArg(cl.kernel(kn), 12, sizeof(cl_mem),   (void*) &unique[0].d_conv);
		clSetKernelArg(cl.kernel(kn), 13, sizeof(cl_mem),   (void*) &unique[0].d_in2_pad_cumv);
		clSetKernelArg(cl.kernel(kn), 14, sizeof(cl_mem),   (void*) &unique[0].d_in2_pad_cumv_sel);
		clSetKernelArg(cl.kernel(kn), 15, sizeof(cl_mem),   (void*) &unique[0].d_in2_sub_cumh);
		clSetKernelArg(cl.kernel(kn), 16, sizeof(cl_mem),   (void*) &unique[0].d_in2_sub_cumh_sel);
		clSetKernelArg(cl.kernel(kn), 17, sizeof(cl_mem),   (void*) &unique[0].d_in2_sub2);
		clSetKernelArg(cl.kernel(kn), 18, sizeof(cl_mem),   (void*) &unique[0].d_in2_sqr);
		clSetKernelArg(cl.kernel(kn), 19, sizeof(cl_mem),   (void*) &unique[0].d_in2_sqr_sub2);
		clSetKernelArg(cl.kernel(kn), 20, sizeof(cl_mem),   (void*) &unique[0].d_in_sqr);
		clSetKernelArg(cl.kernel(kn), 21, sizeof(cl_mem),   (void*) &unique[0].d_tMask);
		clSetKernelArg(cl.kernel(kn), 22, sizeof(cl_mem),   (void*) &unique[0].d_mask_conv);
		clSetKernelArg(cl.kernel(kn), 23, sizeof(cl_int),   (void*) &common_change.frame_no);
		clSetKernelArg(cl.kernel(kn), 24, sizeof(cl_mem),   (void*) &d_in_pointer);
		clSetKernelArg(cl.kernel(kn), 25, sizeof(cl_int),   (void*) &common.endoPoints);
		clSetKernelArg(cl.kernel(kn), 26, sizeof(cl_mem),   (void*) &mem_common_change);
		clSetKernelArg(cl.kernel(kn), 27, sizeof(cl_mem),   (void*) &mem_common);
		clSetKernelArg(cl.kernel(kn), 28, sizeof(cl_mem),   (void*) &mem_unique);
		clSetKernelArg(cl.kernel(kn), 29, sizeof(cl_int),   (void*) &localWorksize);
		cl.launch(kn);

		// free frame after each loop iteration, since AVI library allocates memory for every frame fetched
		free(frame);

		// print frame progress
		// printf("%d ", common_change.frame_no);
		// fflush(NULL);
	}

	time9 = get_time();

	// Print frame progress end.
	// printf("\n");
	// fflush(NULL);

	// Output.
	// Copy results back to host.
	clEnqueueReadBuffer(cl.q(),                                // The command queue.
	                    common.d_tEndoRowLoc,                  // The result on the device.
	                    CL_TRUE,                               // Blocking? (ie. Wait at this line until read has finished?)
	                    0,                                     // Offset. None in this case.
	                    common.endo_mem * common.no_frames,    // Size to copy.
	                    common.tEndoRowLoc,                    // The pointer to the memory on the host.
	                    0,                                     // Number of events in wait list. Not used.
	                    NULL,                                  // Event wait list. Not used.
	                    NULL);                                 // Event object for determining status. Not used.

	clEnqueueReadBuffer(cl.q(),                                // The command queue.
	                    common.d_tEndoColLoc,                  // The result on the device.
	                    CL_TRUE,                               // Blocking? (ie. Wait at this line until read has finished?)
	                    0,                                     // Offset. None in this case.
	                    common.endo_mem * common.no_frames,    // Size to copy.
	                    common.tEndoColLoc,                    // The pointer to the memory on the host.
	                    0,                                     // Number of events in wait list. Not used.
	                    NULL,                                  // Event wait list. Not used.
	                    NULL);                                 // Event object for determining status. Not used.

	clEnqueueReadBuffer(cl.q(),                                // The command queue.
	                    common.d_tEpiRowLoc,                   // The result on the device.
	                    CL_TRUE,                               // Blocking? (ie. Wait at this line until read has finished?)
	                    0,                                     // Offset. None in this case.
	                    common.epi_mem * common.no_frames,     // Size to copy.
	                    common.tEpiRowLoc,                     // The pointer to the memory on the host.
	                    0,                                     // Number of events in wait list. Not used.
	                    NULL,                                  // Event wait list. Not used.
	                    NULL);                                 // Event object for determining status. Not used.

	clEnqueueReadBuffer(cl.q(),                                // The command queue.
	                    common.d_tEpiColLoc,                   // The result on the device.
	                    CL_TRUE,                               // Blocking? (ie. Wait at this line until read has finished?)
	                    0,                                     // Offset. None in this case.
	                    common.epi_mem * common.no_frames,     // Size to copy.
	                    common.tEpiColLoc,                     // The pointer to the memory on the host.
	                    0,                                     // Number of events in wait list. Not used.
	                    NULL,                                  // Event wait list. Not used.
	                    NULL);                                 // Event object for determining status. Not used.

	time10 = get_time();

	// DEALLOCATION.
	// frame
	clReleaseMemObject(common_change.d_frame);

	// endo points
	free(common.endoRow);
	free(common.endoCol);
	free(common.tEndoRowLoc);
	free(common.tEndoColLoc);

	clReleaseMemObject(common.d_endoRow);
	clReleaseMemObject(common.d_endoCol);
	clReleaseMemObject(common.d_tEndoRowLoc);
	clReleaseMemObject(common.d_tEndoColLoc);
	clReleaseMemObject(common.d_endoT);

	// epi points
	free(common.epiRow);
	free(common.epiCol);
	free(common.tEpiRowLoc);
	free(common.tEpiColLoc);

	// Clean up the device memory...
	clReleaseMemObject(common.d_epiRow);
	clReleaseMemObject(common.d_epiCol);
	clReleaseMemObject(common.d_tEpiRowLoc);
	clReleaseMemObject(common.d_tEpiColLoc);
	clReleaseMemObject(common.d_epiT);
	clReleaseMemObject(unique[0].d_in2);
	clReleaseMemObject(unique[0].d_conv);
	clReleaseMemObject(unique[0].d_in2_pad_cumv);
	clReleaseMemObject(unique[0].d_in2_pad_cumv_sel);
	clReleaseMemObject(unique[0].d_in2_sub_cumh);
	clReleaseMemObject(unique[0].d_in2_sub_cumh_sel);
	clReleaseMemObject(unique[0].d_in2_sub2);
	clReleaseMemObject(unique[0].d_in2_sqr);
	clReleaseMemObject(unique[0].d_in2_sqr_sub2);
	clReleaseMemObject(unique[0].d_in_sqr);
	clReleaseMemObject(unique[0].d_tMask);
	clReleaseMemObject(unique[0].d_mask_conv);

	time11 = get_time();

	// Display the timing results.
	printf("Time spent in different stages of the application:\n");
	printf("%.12f s, %.12f % : SETUP VARIABLES\n", (float) (time1-time0) / 1000000, (float) (time1-time0) / (float) (time11-time0) * 100);
	printf("%.12f s, %.12f % : READ INITIAL VIDEO FRAME\n", (float) (time2-time1) / 1000000, (float) (time2-time1) / (float) (time11-time0) * 100);
	printf("%.12f s, %.12f % : READ COMMAND LINE PARAMETERS\n", (float) (time3-time2) / 1000000, (float) (time3-time2) / (float) (time11-time0) * 100);
	printf("%.12f s, %.12f % : READ INPUTS FROM FILE\n", (float) (time4-time3) / 1000000, (float) (time4-time3) / (float) (time11-time0) * 100);
	printf("%.12f s, %.12f % : SETUP, ALLOCATE CPU/GPU MEMORY\n", (float) (time5-time4) / 1000000, (float) (time5-time4) / (float) (time11-time0) * 100);
	printf("%.12f s, %.12f % : READ INPUTS FROM FILE\n", (float) (time6-time5) / 1000000, (float) (time6-time5) / (float) (time11-time0) * 100);
	printf("%.12f s, %.12f % : SETUP, ALLOCATE CPU/ GPU MEMORY\n", (float) (time7-time6) / 1000000, (float) (time7-time6) / (float) (time11-time0) * 100);
	printf("%.12f s, %.12f % : COPY DATA CPU->GPU\n", (float) (time8-time7) / 1000000, (float) (time8-time7) / (float) (time11-time0) * 100);
	printf("%.12f s, %.12f % : COMPUTE\n", (float) (time9-time8) / 1000000, (float) (time9-time8) / (float) (time11-time0) * 100);
	printf("%.12f s, %.12f % : COPY DATA GPU->CPU\n", (float) (time10-time9) / 1000000, (float) (time10-time9) / (float) (time11-time0) * 100);
	printf("%.12f s, %.12f % : FREE MEMORY\n", (float) (time11-time10) / 1000000, (float) (time11-time10) / (float) (time11-time0) * 100);
	printf("Total time:\n");
	printf("%.12f s\n", (float) (time11-time0) / 1000000);

}

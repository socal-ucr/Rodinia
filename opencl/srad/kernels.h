const char* source_str = "\n" \
	"// statistical kernel \n" \
	"__kernel void extract(long d_Ne, __global float *d_I) \n" \
	"{ \n" \
	"	int ei = get_global_id(0);  // unique thread id, more threads than actual elements !!! \n" \
	" \n" \
	"	// copy input to output & log uncompress \n" \
	"	if(ei<d_Ne) \n" \
	"	{ \n" \
	"		// do only for the number of elements, omit extra threads \n" \
	"		d_I[ei] = exp(d_I[ei]/255);// exponentiate input IMAGE and copy to output image \n" \
	"	} \n" \
	"} \n" \
	" \n" \
	" \n" \
	"// statistical kernel \n" \
	"__kernel void prepare(long d_Ne, \n" \
	"	__global float *d_I,      // pointer to output image (DEVICE GLOBAL MEMORY) \n" \
	"	__global float *d_sums,   // pointer to input image (DEVICE GLOBAL MEMORY) \n" \
	"	__global float *d_sums2) \n" \
	"{ \n" \
	"	int ei = get_global_id(0);  // unique thread id, more threads than actual elements !!! \n" \
	" \n" \
	"	// copy input to output & log uncompress \n" \
	"	if(ei < d_Ne) \n" \
	"	{ \n" \
	"		// do only for the number of elements, omit extra threads \n" \
	"		d_sums[ei] = d_I[ei]; \n" \
	"		d_sums2[ei] = d_I[ei] * d_I[ei]; \n" \
	"	} \n" \
	"} \n" \
	" \n" \
	" \n" \
	"// statistical kernel \n" \
	"__kernel void reduce(long d_Ne,               // number of elements in array \n" \
	"	long d_no,               // number of sums to reduce \n" \
	"	int d_mul,               // increment \n" \
	"	__global float* d_sums,  // pointer to partial sums variable \n" \
	"	__global float* d_sums2, // pointer to partial sums variable \n" \
	"	__local float* d_psum,   // data for block calculations allocated by every block in its shared memory. \n" \
	"	__local float* d_psum2,  // data for block calculations allocated by every block in its shared memory. \n" \
	"	int blocksize) \n" \
	"{ \n" \
	"	int ei = get_global_id(0);        // unique thread id, more threads than actual elements !!! \n" \
	"	int bx = get_group_id(0);         // get current horizontal block index \n" \
	"	int tx = get_local_id(0);         // get current horizontal thread index \n" \
	"	int nf = (d_Ne%blocksize);        // number of elements assigned to last block \n" \
	"	int df = 0;                       // divisibility factor for the last block \n" \
	"	int i;                            // counters \n" \
	" \n" \
	"	// copy data to shared memory \n" \
	"	if(ei < d_no) \n" \
	"	{ \n" \
	"		// do only for the number of elements, omit extra threads \n" \
	"		d_psum[tx] = d_sums[ei*d_mul]; \n" \
	"		d_psum2[tx] = d_sums2[ei*d_mul]; \n" \
	"	} \n" \
	" \n" \
	"	// reduction of sums if all blocks are full (rare case) \n" \
	"	if(nf == blocksize) \n" \
	"	{ \n" \
	"		// sum of every 2, 4, ..., blocksize elements \n" \
	"		for(i=2; i<=blocksize; i=2*i) \n" \
	"		{ \n" \
	"			// sum of elements \n" \
	"			if((tx+1) % i == 0) \n" \
	"			{ \n" \
	"				// every ith \n" \
	"				d_psum[tx] = d_psum[tx] + d_psum[tx-i/2]; \n" \
	"				d_psum2[tx] = d_psum2[tx] + d_psum2[tx-i/2]; \n" \
	"			} \n" \
	"			// synchronization \n" \
	"			barrier(CLK_LOCAL_MEM_FENCE); \n" \
	"		} \n" \
	"		// final sumation by last thread in every block \n" \
	"		if(tx==blocksize-1) \n" \
	"		{ \n" \
	"			// block result stored in global memory \n" \
	"			d_sums[bx*d_mul*blocksize] = d_psum[tx]; \n" \
	"			d_sums2[bx*d_mul*blocksize] = d_psum2[tx]; \n" \
	"		} \n" \
	"	} \n" \
	"	// reduction of sums if last block is not full (common case) \n" \
	"	else{ \n" \
	"		// for full blocks \n" \
	"		if(bx != (get_num_groups(0) - 1)) \n" \
	"		{ \n" \
	"			// sum of every 2, 4, ..., blocksize elements \n" \
	"			for(i=2; i<=blocksize; i=2*i) \n" \
	"			{ \n" \
	"				// sum of elements \n" \
	"				if((tx+1) % i == 0) \n" \
	"				{ \n" \
	"					// every ith \n" \
	"					d_psum[tx] = d_psum[tx] + d_psum[tx-i/2]; \n" \
	"					d_psum2[tx] = d_psum2[tx] + d_psum2[tx-i/2]; \n" \
	"				} \n" \
	"				// synchronization \n" \
	"				barrier(CLK_LOCAL_MEM_FENCE); \n" \
	"			} \n" \
	"			// final sumation by last thread in every block \n" \
	"			if(tx==blocksize-1) \n" \
	"			{ \n" \
	"				// block result stored in global memory \n" \
	"				d_sums[bx*d_mul*blocksize] = d_psum[tx]; \n" \
	"				d_sums2[bx*d_mul*blocksize] = d_psum2[tx]; \n" \
	"			} \n" \
	"		} \n" \
	"		// for not full block (last block) \n" \
	"		else{ \n" \
	"			// figure out divisibility \n" \
	"			for(i=2; i<=blocksize; i=2*i) \n" \
	"			{ \n" \
	"				if(nf >= i) \n" \
	"				{ \n" \
	"					df = i; \n" \
	"				} \n" \
	"			} \n" \
	"			// sum of every 2, 4, ..., blocksize elements \n" \
	"			for(i=2; i<=df; i=2*i) \n" \
	"			{ \n" \
	"				// sum of elements (only busy threads) \n" \
	"				if((tx+1) % i == 0 || tx<df) \n" \
	"				{ \n" \
	"					// every ith \n" \
	"					d_psum[tx] = d_psum[tx] + d_psum[tx-i/2]; \n" \
	"					d_psum2[tx] = d_psum2[tx] + d_psum2[tx-i/2]; \n" \
	"				} \n" \
	"				// synchronization (all threads) \n" \
	"				barrier(CLK_LOCAL_MEM_FENCE); \n" \
	"			} \n" \
	"			// compute the remainder and final summation by last busy thread \n" \
	"			if(tx == df-1) \n" \
	"			{ \n" \
	"				for(i=(bx*blocksize)+df; i<(bx*blocksize)+nf; i++) \n" \
	"				{ \n" \
	"					d_psum[tx] = d_psum[tx] + d_sums[i]; \n" \
	"					d_psum2[tx] = d_psum2[tx] + d_sums2[i]; \n" \
	"				} \n" \
	"				d_sums[bx*d_mul*blocksize] = d_psum[tx]; \n" \
	"				d_sums2[bx*d_mul*blocksize] = d_psum2[tx]; \n" \
	"			} \n" \
	"		} \n" \
	"	} \n" \
	"} \n" \
	" \n" \
	" \n" \
	"// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n" \
	"// BUG IN SRAD APPLICATIONS SEEMS TO BE SOMEWHERE IN THIS CODE, MEMORY CORRUPTION \n" \
	"// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n" \
	" \n" \
	"/* SRAD kernel */ \n" \
	"__kernel void srad(float d_lambda, \n" \
	"	int d_Nr, \n" \
	"	int d_Nc, \n" \
	"	long d_Ne, \n" \
	"	__global int *d_iN, \n" \
	"	__global int *d_iS, \n" \
	"	__global int *d_jE, \n" \
	"	__global int *d_jW, \n" \
	"	__global float *d_dN, \n" \
	"	__global float *d_dS, \n" \
	"	__global float *d_dE, \n" \
	"	__global float *d_dW, \n" \
	"	float d_q0sqr, \n" \
	"	__global float *d_c, \n" \
	"	__global float *d_I, \n" \
	"	int blocksize) \n" \
	"{ \n" \
	"	// indexes \n" \
	"	int bx = get_group_id(0);   // get current horizontal block index \n" \
	"	int tx = get_local_id(0);   // get current horizontal thread index \n" \
	"	int ei = bx*blocksize+tx;   // more threads than actual elements !!! \n" \
	"	int row;                    // column, x position \n" \
	"	int col;                    // row, y position \n" \
	" \n" \
	"	// variables \n" \
	"	float d_Jc; \n" \
	"	float d_dN_loc, d_dS_loc, d_dW_loc, d_dE_loc; \n" \
	"	float d_c_loc; \n" \
	"	float d_G2,d_L,d_num,d_den,d_qsqr; \n" \
	" \n" \
	"	// figure out row/col location in new matrix \n" \
	"	row = (ei+1) % d_Nr - 1;        // (0-n) row \n" \
	"	col = (ei+1) / d_Nr + 1 - 1;    // (0-n) column \n" \
	"	if((ei+1) % d_Nr == 0) \n" \
	"	{ \n" \
	"		row = d_Nr - 1; \n" \
	"		col = col - 1; \n" \
	"	} \n" \
	" \n" \
	"	// make sure that only threads matching jobs run \n" \
	"	if(ei<d_Ne) \n" \
	"	{ \n" \
	"		// directional derivatives, ICOV, diffusion coefficent \n" \
	"		d_Jc = d_I[ei];    // get value of the current element \n" \
	" \n" \
	"		// directional derivates (every element of IMAGE)(try to copy to shared memory or temp files) \n" \
	"		d_dN_loc = d_I[d_iN[row] + d_Nr*col] - d_Jc;     // north direction derivative \n" \
	"		d_dS_loc = d_I[d_iS[row] + d_Nr*col] - d_Jc;     // south direction derivative \n" \
	"		d_dW_loc = d_I[row + d_Nr*d_jW[col]] - d_Jc;     // west direction derivative \n" \
	"		d_dE_loc = d_I[row + d_Nr*d_jE[col]] - d_Jc;     // east direction derivative \n" \
	" \n" \
	"		// normalized discrete gradient mag squared (equ 52,53) \n" \
	"		d_G2 = (d_dN_loc*d_dN_loc + d_dS_loc*d_dS_loc + d_dW_loc*d_dW_loc + d_dE_loc*d_dE_loc) / (d_Jc*d_Jc);    // gradient (based on derivatives) \n" \
	" \n" \
	"		// normalized discrete laplacian (equ 54) \n" \
	"		d_L = (d_dN_loc + d_dS_loc + d_dW_loc + d_dE_loc) / d_Jc;  // laplacian (based on derivatives) \n" \
	" \n" \
	"		// ICOV (equ 31/35) \n" \
	"		d_num  = (0.5f*d_G2) - ((1.0f/16.0f)*(d_L*d_L)) ;        // num (based on gradient and laplacian) \n" \
	"		d_den  = 1 + (0.25f*d_L);                                // den (based on laplacian) \n" \
	"		d_qsqr = d_num/(d_den*d_den);                            // qsqr (based on num and den) \n" \
	" \n" \
	"		// diffusion coefficent (equ 33) (every element of IMAGE) \n" \
	"		d_den = (d_qsqr-d_q0sqr) / (d_q0sqr * (1+d_q0sqr)) ;       // den (based on qsqr and q0sqr) \n" \
	"		d_c_loc = 1.0f / (1.0f+d_den) ;                            // diffusion coefficient (based on den) \n" \
	" \n" \
	"		// saturate diffusion coefficent to 0-1 range \n" \
	"		if (d_c_loc < 0) \n" \
	"		{                        // if diffusion coefficient < 0 \n" \
	"			d_c_loc = 0;         // ... set to 0 \n" \
	"		} \n" \
	"		else if (d_c_loc > 1) \n" \
	"		{                        // if diffusion coefficient > 1 \n" \
	"			d_c_loc = 1;         // ... set to 1 \n" \
	"		} \n" \
	" \n" \
	"		// save data to global memory \n" \
	"		d_dN[ei] = d_dN_loc; \n" \
	"		d_dS[ei] = d_dS_loc; \n" \
	"		d_dW[ei] = d_dW_loc; \n" \
	"		d_dE[ei] = d_dE_loc; \n" \
	"		d_c[ei] = d_c_loc; \n" \
	"	} \n" \
	" \n" \
	"} \n" \
	" \n" \
	" \n" \
	"// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n" \
	"// BUG IN SRAD APPLICATIONS SEEMS TO BE SOMEWHERE IN THIS CODE, MEMORY CORRUPTION \n" \
	"// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n" \
	" \n" \
	"/* SRAD2 kernel */ \n" \
	"__kernel void srad2(float d_lambda, \n" \
	"	int d_Nr, \n" \
	"	int d_Nc, \n" \
	"	long d_Ne, \n" \
	"	__global int *d_iN, \n" \
	"	__global int *d_iS, \n" \
	"	__global int *d_jE, \n" \
	"	__global int *d_jW, \n" \
	"	__global float *d_dN, \n" \
	"	__global float *d_dS, \n" \
	"	__global float *d_dE, \n" \
	"	__global float *d_dW, \n" \
	"	__global float *d_c, \n" \
	"	__global float *d_I, \n" \
	"	int blocksize) \n" \
	"{ \n" \
	"	// indexes \n" \
	"	int bx = get_group_id(0);   // get current horizontal block index \n" \
	"	int tx = get_local_id(0);   // get current horizontal thread index \n" \
	"	int ei = bx*blocksize+tx;   // more threads than actual elements !!! \n" \
	"	int row;                    // column, x position \n" \
	"	int col;                    // row, y position \n" \
	" \n" \
	"	// variables \n" \
	"	float d_cN,d_cS,d_cW,d_cE; \n" \
	"	float d_D; \n" \
	" \n" \
	"	// figure out row/col location in new matrix \n" \
	"	row = (ei+1) % d_Nr - 1;        // (0-n) row \n" \
	"	col = (ei+1) / d_Nr + 1 - 1;    // (0-n) column \n" \
	"	if((ei+1) % d_Nr == 0) \n" \
	"	{ \n" \
	"		row = d_Nr - 1; \n" \
	"		col = col - 1; \n" \
	"	} \n" \
	" \n" \
	"	// make sure that only threads matching jobs run \n" \
	"	if(ei<d_Ne) \n" \
	"	{ \n" \
	"		// diffusion coefficent \n" \
	"		d_cN = d_c[ei];                       // north diffusion coefficient \n" \
	"		d_cS = d_c[d_iS[row] + d_Nr*col];     // south diffusion coefficient \n" \
	"		d_cW = d_c[ei];                       // west diffusion coefficient \n" \
	"		d_cE = d_c[row + d_Nr * d_jE[col]];   // east diffusion coefficient \n" \
	" \n" \
	"		// divergence (equ 58) \n" \
	"		d_D = d_cN*d_dN[ei] + d_cS*d_dS[ei] + d_cW*d_dW[ei] + d_cE*d_dE[ei];  // divergence \n" \
	" \n" \
	"		// image update (equ 61) (every element of IMAGE) \n" \
	"		// (updates image based on input time step and divergence) \n" \
	"		d_I[ei] = d_I[ei] + 0.25f*d_lambda*d_D; \n" \
	"	} \n" \
	"} \n" \
	" \n" \
	" \n" \
	"// statistical kernel \n" \
	"__kernel void compress(long d_Ne, \n" \
	"	__global float *d_I, \n" \
	"	int blocksize) \n" \
	"{ \n" \
	"	// indexes \n" \
	"	int bx = get_group_id(0);       // get current horizontal block index \n" \
	"	int tx = get_local_id(0);       // get current horizontal thread index \n" \
	"	int ei = (bx*blocksize)+tx;     // unique thread id, more threads than actual elements !!! \n" \
	" \n" \
	"	// copy input to output & log uncompress \n" \
	"	// do only for the number of elements, omit extra threads \n" \
	"	if(ei<d_Ne) \n" \
	"	{ \n" \
	"		d_I[ei] = log(d_I[ei])*255;   // exponentiate input IMAGE and copy to output image \n" \
	"	} \n" \
	"} \n";
// statistical kernel
__kernel void extract(long d_Ne, __global float *d_I)
{  
	int ei = get_global_id(0);  // unique thread id, more threads than actual elements !!!
  
	// copy input to output & log uncompress
	if(ei<d_Ne)
	{
		// do only for the number of elements, omit extra threads
		d_I[ei] = exp(d_I[ei]/255);// exponentiate input IMAGE and copy to output image
	}
}


// statistical kernel
__kernel void prepare(long d_Ne,
                      __global float *d_I,      // pointer to output image (DEVICE GLOBAL MEMORY)
                      __global float *d_sums,   // pointer to input image (DEVICE GLOBAL MEMORY)
                      __global float *d_sums2)
{
	int ei = get_global_id(0);  // unique thread id, more threads than actual elements !!!

	// copy input to output & log uncompress
	if(ei < d_Ne)
	{
		// do only for the number of elements, omit extra threads
		d_sums[ei] = d_I[ei];
		d_sums2[ei] = d_I[ei] * d_I[ei];
	}
}


// statistical kernel
__kernel void reduce(long d_Ne,               // number of elements in array
                     long d_no,               // number of sums to reduce
                     int d_mul,               // increment
                     __global float* d_sums,  // pointer to partial sums variable
                     __global float* d_sums2, // pointer to partial sums variable
                     __local float* d_psum,   // data for block calculations allocated by every block in its shared memory.
                     __local float* d_psum2,  // data for block calculations allocated by every block in its shared memory.
                     int blocksize)
{
	int ei = get_global_id(0);        // unique thread id, more threads than actual elements !!!
	int bx = get_group_id(0);         // get current horizontal block index
	int tx = get_local_id(0);         // get current horizontal thread index
	int nf = (d_Ne%blocksize);        // number of elements assigned to last block
	int df = 0;                       // divisibility factor for the last block
	int i;                            // counters

	// copy data to shared memory
	if(ei < d_no)
	{   
		// do only for the number of elements, omit extra threads            
		d_psum[tx] = d_sums[ei*d_mul];
		d_psum2[tx] = d_sums2[ei*d_mul];
	}
        
	// reduction of sums if all blocks are full (rare case) 
	if(nf == blocksize)
	{
		// sum of every 2, 4, ..., blocksize elements
		for(i=2; i<=blocksize; i=2*i)
		{
			// sum of elements
			if((tx+1) % i == 0)
			{   
				// every ith
				d_psum[tx] = d_psum[tx] + d_psum[tx-i/2];
				d_psum2[tx] = d_psum2[tx] + d_psum2[tx-i/2];
			}
			// synchronization
			barrier(CLK_LOCAL_MEM_FENCE);
		}
		// final sumation by last thread in every block
		if(tx==blocksize-1)
		{
			// block result stored in global memory
			d_sums[bx*d_mul*blocksize] = d_psum[tx];
			d_sums2[bx*d_mul*blocksize] = d_psum2[tx];
		}
	}
	// reduction of sums if last block is not full (common case)
	else{ 
		// for full blocks
		if(bx != (get_num_groups(0) - 1))
		{
			// sum of every 2, 4, ..., blocksize elements
			for(i=2; i<=blocksize; i=2*i)
			{                                                             
				// sum of elements
				if((tx+1) % i == 0)
				{                                                                            
					// every ith
					d_psum[tx] = d_psum[tx] + d_psum[tx-i/2];
					d_psum2[tx] = d_psum2[tx] + d_psum2[tx-i/2];
				}
				// synchronization
				barrier(CLK_LOCAL_MEM_FENCE);                                                                                        
			}
			// final sumation by last thread in every block
			if(tx==blocksize-1)
			{                                                                               
				// block result stored in global memory
				d_sums[bx*d_mul*blocksize] = d_psum[tx];
				d_sums2[bx*d_mul*blocksize] = d_psum2[tx];
			}       
		}
		// for not full block (last block)
		else{                                                                                                                           
			// figure out divisibility
			for(i=2; i<=blocksize; i=2*i)
			{                                                             
				if(nf >= i)
				{
					df = i;
				}
			}
			// sum of every 2, 4, ..., blocksize elements
			for(i=2; i<=df; i=2*i)
			{                                                                                 
				// sum of elements (only busy threads)
				if((tx+1) % i == 0 || tx<df)
				{                                                           
					// every ith
					d_psum[tx] = d_psum[tx] + d_psum[tx-i/2];
					d_psum2[tx] = d_psum2[tx] + d_psum2[tx-i/2];
				}
				// synchronization (all threads)
				barrier(CLK_LOCAL_MEM_FENCE);                                                                                        
			}
			// compute the remainder and final summation by last busy thread
			if(tx == df-1)
			{                                                                           
				for(i=(bx*blocksize)+df; i<(bx*blocksize)+nf; i++)
				{                                           
					d_psum[tx] = d_psum[tx] + d_sums[i];
					d_psum2[tx] = d_psum2[tx] + d_sums2[i];
				}
				d_sums[bx*d_mul*blocksize] = d_psum[tx];
				d_sums2[bx*d_mul*blocksize] = d_psum2[tx];
			}
		}
	}
}


// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// BUG IN SRAD APPLICATIONS SEEMS TO BE SOMEWHERE IN THIS CODE, MEMORY CORRUPTION
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

/* SRAD kernel */
__kernel void srad(float d_lambda, 
                   int d_Nr, 
                   int d_Nc, 
                   long d_Ne, 
                   __global int *d_iN, 
                   __global int *d_iS, 
                   __global int *d_jE, 
                   __global int *d_jW, 
                   __global float *d_dN, 
                   __global float *d_dS, 
                   __global float *d_dE, 
                   __global float *d_dW, 
                   float d_q0sqr, 
                   __global float *d_c, 
                   __global float *d_I,
                   int blocksize)
{
	// indexes
	int bx = get_group_id(0);   // get current horizontal block index
	int tx = get_local_id(0);   // get current horizontal thread index
	int ei = bx*blocksize+tx;   // more threads than actual elements !!!
	int row;                    // column, x position
	int col;                    // row, y position

	// variables
	float d_Jc;
	float d_dN_loc, d_dS_loc, d_dW_loc, d_dE_loc;
	float d_c_loc;
	float d_G2,d_L,d_num,d_den,d_qsqr;
        
	// figure out row/col location in new matrix
	row = (ei+1) % d_Nr - 1;        // (0-n) row
	col = (ei+1) / d_Nr + 1 - 1;    // (0-n) column
	if((ei+1) % d_Nr == 0)
	{
		row = d_Nr - 1;
		col = col - 1;
	}

	// make sure that only threads matching jobs run        
	if(ei<d_Ne)
	{
		// directional derivatives, ICOV, diffusion coefficent
		d_Jc = d_I[ei];    // get value of the current element
                
		// directional derivates (every element of IMAGE)(try to copy to shared memory or temp files)
		d_dN_loc = d_I[d_iN[row] + d_Nr*col] - d_Jc;     // north direction derivative
		d_dS_loc = d_I[d_iS[row] + d_Nr*col] - d_Jc;     // south direction derivative
		d_dW_loc = d_I[row + d_Nr*d_jW[col]] - d_Jc;     // west direction derivative
		d_dE_loc = d_I[row + d_Nr*d_jE[col]] - d_Jc;     // east direction derivative
                 
		// normalized discrete gradient mag squared (equ 52,53)
		d_G2 = (d_dN_loc*d_dN_loc + d_dS_loc*d_dS_loc + d_dW_loc*d_dW_loc + d_dE_loc*d_dE_loc) / (d_Jc*d_Jc);    // gradient (based on derivatives)
                
		// normalized discrete laplacian (equ 54)
		d_L = (d_dN_loc + d_dS_loc + d_dW_loc + d_dE_loc) / d_Jc;  // laplacian (based on derivatives)

		// ICOV (equ 31/35)
		d_num  = (0.5f*d_G2) - ((1.0f/16.0f)*(d_L*d_L)) ;        // num (based on gradient and laplacian)
		d_den  = 1 + (0.25f*d_L);                                // den (based on laplacian)
		d_qsqr = d_num/(d_den*d_den);                            // qsqr (based on num and den)
         
		// diffusion coefficent (equ 33) (every element of IMAGE)
		d_den = (d_qsqr-d_q0sqr) / (d_q0sqr * (1+d_q0sqr)) ;       // den (based on qsqr and q0sqr)
		d_c_loc = 1.0f / (1.0f+d_den) ;                            // diffusion coefficient (based on den)
            
		// saturate diffusion coefficent to 0-1 range
		if (d_c_loc < 0)
		{                        // if diffusion coefficient < 0
			d_c_loc = 0;         // ... set to 0
		}
		else if (d_c_loc > 1)
		{                        // if diffusion coefficient > 1
			d_c_loc = 1;         // ... set to 1
		}

		// save data to global memory
		d_dN[ei] = d_dN_loc; 
		d_dS[ei] = d_dS_loc; 
		d_dW[ei] = d_dW_loc; 
		d_dE[ei] = d_dE_loc;
		d_c[ei] = d_c_loc;
	}
        
}


// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// BUG IN SRAD APPLICATIONS SEEMS TO BE SOMEWHERE IN THIS CODE, MEMORY CORRUPTION
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

/* SRAD2 kernel */
__kernel void srad2(float d_lambda,
                    int d_Nr, 
                    int d_Nc, 
                    long d_Ne, 
                    __global int *d_iN,
                    __global int *d_iS,
                    __global int *d_jE,
                    __global int *d_jW,
                    __global float *d_dN, 
                    __global float *d_dS, 
                    __global float *d_dE, 
                    __global float *d_dW, 
                    __global float *d_c, 
                    __global float *d_I,
                    int blocksize)
{
	// indexes
	int bx = get_group_id(0);   // get current horizontal block index
	int tx = get_local_id(0);   // get current horizontal thread index
	int ei = bx*blocksize+tx;   // more threads than actual elements !!!
	int row;                    // column, x position
	int col;                    // row, y position

	// variables
	float d_cN,d_cS,d_cW,d_cE;
	float d_D;

	// figure out row/col location in new matrix
	row = (ei+1) % d_Nr - 1;        // (0-n) row
	col = (ei+1) / d_Nr + 1 - 1;    // (0-n) column
	if((ei+1) % d_Nr == 0)
	{
		row = d_Nr - 1;
		col = col - 1;
	}

	// make sure that only threads matching jobs run
	if(ei<d_Ne)
	{
		// diffusion coefficent
		d_cN = d_c[ei];                       // north diffusion coefficient
		d_cS = d_c[d_iS[row] + d_Nr*col];     // south diffusion coefficient
		d_cW = d_c[ei];                       // west diffusion coefficient
		d_cE = d_c[row + d_Nr * d_jE[col]];   // east diffusion coefficient

		// divergence (equ 58)
		d_D = d_cN*d_dN[ei] + d_cS*d_dS[ei] + d_cW*d_dW[ei] + d_cE*d_dE[ei];  // divergence

		// image update (equ 61) (every element of IMAGE)
		// (updates image based on input time step and divergence)
		d_I[ei] = d_I[ei] + 0.25f*d_lambda*d_D;
	}
}


// statistical kernel
__kernel void compress(long d_Ne,
                       __global float *d_I,
                       int blocksize)
{
	// indexes
	int bx = get_group_id(0);       // get current horizontal block index
	int tx = get_local_id(0);       // get current horizontal thread index
	int ei = (bx*blocksize)+tx;     // unique thread id, more threads than actual elements !!!

	// copy input to output & log uncompress
	// do only for the number of elements, omit extra threads
	if(ei<d_Ne)
	{
		d_I[ei] = log(d_I[ei])*255;   // exponentiate input IMAGE and copy to output image
	}
}


/********************************
* CALC LIKELIHOOD SUM
* DETERMINES THE LIKELIHOOD SUM BASED ON THE FORMULA: SUM( (IK[IND] - 100)^2 - (IK[IND] - 228)^2)/ 100
* param 1 I 3D matrix
* param 2 current ind array
* param 3 length of ind array
* returns a float representing the sum
********************************/
float calcLikelihoodSum(__global unsigned char * I, __global int * ind, int numOnes, int index){
	float likelihoodSum = 0.0;
	int x;
	for(x = 0; x < numOnes; x++)
		likelihoodSum += (pow((float)(I[ind[index*numOnes + x]] - 100),2) - pow((float)(I[ind[index*numOnes + x]]-228),2))/50.0;
	return likelihoodSum;
}
/****************************
CDF CALCULATE
CALCULATES CDF
param1 CDF
param2 weights
param3 Nparticles
*****************************/
void cdfCalc(__global float * CDF, __global float * weights, int Nparticles){
	int x;
	CDF[0] = weights[0];
	for(x = 1; x < Nparticles; x++){
		CDF[x] = weights[x] + CDF[x-1];
	}
}
/*****************************
* RANDU
* GENERATES A UNIFORM DISTRIBUTION
* returns a float representing a randomily generated number from a uniform distribution with range [0, 1)
******************************/
float d_randu(__global int * seed, int index)
{
	
	int M = INT_MAX;
	int A = 1103515245;
	int C = 12345;
	int num = A*seed[index] + C;
	seed[index] = num % M;
	num = seed[index];
	return fabs(num/((float) M));
}

/**
* Generates a normally distributed random number using the Box-Muller transformation
* @note This function is thread-safe
* @param seed The seed array
* @param index The specific index of the seed to be advanced
* @return a double representing random number generated using the Box-Muller algorithm
* @see http://en.wikipedia.org/wiki/Normal_distribution, section computing value for normal random distribution
*/
float d_randn(__global int * seed, int index){
	//Box-Muller algortihm
	float pi = 3.14159265358979323846;
	float u = d_randu(seed, index);
	float v = d_randu(seed, index);
	float cosine = cos(2*pi*v);
	float rt = -2*log(u);
	return sqrt(rt)*cosine;
}

/****************************
UPDATE WEIGHTS
UPDATES WEIGHTS
param1 weights
param2 likelihood
param3 Nparticles
****************************/
float updateWeights(__global float * weights, __global float * likelihood, int Nparticles){
	int x;
	float sum = 0;
	for(x = 0; x < Nparticles; x++){
		weights[x] = weights[x] * exp(likelihood[x]);
		sum += weights[x];
	}		
	return sum;
}

int findIndexBin(__global float * CDF, int beginIndex, int endIndex, float value)
{
	if(endIndex < beginIndex)
		return -1;
	int middleIndex;
	while(endIndex > beginIndex)
	{
		middleIndex = beginIndex + ((endIndex-beginIndex)/2);
		if(CDF[middleIndex] >= value)
		{
			if(middleIndex == 0)
				return middleIndex;
			else if(CDF[middleIndex-1] < value)
				return middleIndex;
			else if(CDF[middleIndex-1] == value)
			{
				while(CDF[middleIndex] == value && middleIndex >= 0)
					middleIndex--;
				middleIndex++;
				return middleIndex;
			}
		}
		if(CDF[middleIndex] > value)
			endIndex = middleIndex-1;
		else
			beginIndex = middleIndex+1;
	}
	return -1;
}

/*******************************************
* OpenCL helper function to read a single element from a 2d image
* param1: img
* param2: index
*******************************************/
float tex1Dfetch(__read_only image2d_t img, int index){

   const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | //Natural coordinates
         CLK_ADDRESS_CLAMP | //Clamp to zeros
         CLK_FILTER_NEAREST; //Don't interpolate

   if (index < 0) return 0;
   //Divide desired position by 4 because there are 4 components per pixel
   int imgPos = index >> 2;

   int2 coords;
   coords.x = imgPos >> 13;
   coords.y = imgPos & 0x1fff;   //Reads the float4
   float4 temp = read_imagef(img, smp, coords);
 

   //Computes the remainder of imgPos / 4 to check if function should return x,y,z or w component.
   imgPos = index & 0x0003;

   if (imgPos < 2){
      if (imgPos == 0) return temp.x;
      else return temp.y;
   }
   else{
      if (imgPos == 2) return temp.z;
      else return temp.w;
   }
}

/*****************************
* CUDA Find Index Kernel Function to replace FindIndex
* param1: arrayX
* param2: arrayY
* param3: CDF
* param4: u
* param5: xj
* param6: yj
* param7: weights
* param8: Nparticles
*****************************/
__kernel void find_index_kernel(__global float * arrayX, __global float * arrayY, 
				__global float * CDF, __global float * u, __global float * xj, 
				__global float * yj, __global float * weights, int Nparticles,
				__read_only image2d_t tex_CDF ){
	int i = get_global_id(0);
	
	if(i < Nparticles){
	
		int index = -1;
		int x;

		for(x = 0; x < Nparticles; x++){
			if(tex1Dfetch(tex_CDF, x) >= u[i]){
				index = x;
				break;
			}
		}
		if(index == -1){
			index = Nparticles-1;
		}
		
		xj[i] = arrayX[index];
		yj[i] = arrayY[index];
		
		weights[i] = 1/((float)(Nparticles));
		
	}
	barrier(CLK_GLOBAL_MEM_FENCE);
}
__kernel void normalize_weights_kernel(__global float * weights, int Nparticles, __global float * partial_sums, __global float * CDF, __global float * u, __global int * seed)
{
	int i = get_global_id(0);
	__local float u1, sumWeights;
	sumWeights = partial_sums[0];
	if(i < Nparticles)
	{
		weights[i] = weights[i]/sumWeights;
	}
	if(i == 0)
	{
		cdfCalc(CDF, weights, Nparticles);
		u1 = (1/((float)(Nparticles)))*d_randu(seed, i);
	}
	if(i < Nparticles)
	{
		barrier(CLK_GLOBAL_MEM_FENCE);
		u[i] = u1 + i/((float)(Nparticles));
	}
}
__kernel void sum_kernel(__global float* partial_sums, int Nparticles, __read_only image2d_t tex_sums)
{
	int i = get_global_id(0);
	
	if(i == 0)
	{
		int x;
		float sum = 0;
		for(x = 0; x < Nparticles/512; x++)
		{
			sum += tex1Dfetch(tex_sums, x);
		}
		partial_sums[0] = sum;
	}
}
/*****************************
* OpenCL Likelihood Kernel Function to replace FindIndex
* param1: arrayX
* param2: arrayY
* param2.5: CDF
* param3: ind
* param4: objxy
* param5: likelihood
* param6: I
* param6.5: u
* param6.75: weights
* param7: Nparticles
* param8: countOnes
* param9: max_size
* param10: k
* param11: IszY
* param12: Nfr
*****************************/
__kernel void likelihood_kernel(__global float * arrayX, __global float * arrayY, __global float * CDF, __global int * ind, __global int * objxy, __global float * likelihood, __global unsigned char * I, __global float * u, __global float * weights, int Nparticles, int countOnes, int max_size, int k, int IszY, int Nfr, __global int *seed, __global float * partial_sums){
	int block_id = get_group_id(0);
        int thread_id = get_local_id(0);
	int i = get_global_id(0);
	int y;
	float indX, indY;
	__local float buffer[512];
	if(i < Nparticles){
		arrayX[i] = arrayX[i] + 1.0 + 5.0*d_randn(seed, i);
		arrayY[i] = arrayY[i] - 2.0 + 2.0*d_randn(seed, i);
		barrier(CLK_GLOBAL_MEM_FENCE);
	}
	if(i < Nparticles)
	{
		for(y = 0; y < countOnes; y++){
			indX = round(arrayX[i]) + objxy[y*2 + 1];
			indY = round(arrayY[i]) + objxy[y*2];
			ind[i*countOnes + y] = fabs(indX*IszY*Nfr + indY*Nfr + k);
			if(ind[i*countOnes + y] >= max_size)
				ind[i*countOnes + y] = 0;
		}
		likelihood[i] = calcLikelihoodSum(I, ind, countOnes, i);
		likelihood[i] = likelihood[i]/countOnes;
		
		barrier(CLK_GLOBAL_MEM_FENCE);
	}
	
	if(i < Nparticles)
	{
		weights[i] = weights[i]*likelihood[i];
		barrier(CLK_GLOBAL_MEM_FENCE);
		buffer[thread_id] = weights[i];
		barrier(CLK_GLOBAL_MEM_FENCE);
		
		for(unsigned int s=get_local_size(0)/2; s>0; s>>=1)
		{
			if(thread_id < s)
			{
				buffer[thread_id] += buffer[thread_id + s];
			}
			barrier(CLK_GLOBAL_MEM_FENCE);
		}
		if(thread_id == 0)
		{
			partial_sums[block_id] = buffer[0];
		}
		barrier(CLK_GLOBAL_MEM_FENCE);
	}//*/	
}//*/


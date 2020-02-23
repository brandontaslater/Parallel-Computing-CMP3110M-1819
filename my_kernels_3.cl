//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
//-------------------------- EXPECTED OUTPUTS ---------------------------
//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
/*
Filename: temp_lincolnshire.txt
Total of 1873106 temperatures processed.
AVG = 9.77
MIN = -25.00
MAX = 45.00
STD = 5.92
1QT = 5.30
3QT = 14.00
MED = 9.80

Filename: temp_lincolnshire_short.txt
Total of 18732 temperatures processed.
AVG = 9.73
MIN = -25.00
MAX = 31.50
STD = 5.91
1QT = 5.10
3QT = 14.00
MED = 9.80
*/

// Variable Description:
// input_data - the vector holding the input data
// output_data - the vector holding the output data
// local

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
//------------------------- REDUCTION INTEGERS --------------------------
//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
// REDUCTION OF THE VECTOR INTO A SINGLE LOCATION USING LOCAL MEMORY USING ATOMIC_ADD FOR INTEGERS
kernel void Local_Atomic_Reduction_Integer(global const int* input_data, global int* output_data, local int* local_memory) 
{
	int global_id = get_global_id(0); // id is 0 to n (size of A vector)
	int local_id = get_local_id(0); // local id is 0 to local_size (repeats for each work group)
	int local_size = get_local_size(0); // the size of the local memory, id uses this for knowing max
	local_memory[local_id] = (input_data[global_id]);
	//local_memory[local_id] = input_data[global_id]; // transfer of global memory to local memory
	barrier(CLK_LOCAL_MEM_FENCE); // Local memory barrier, haults and waits for workgroup to finish
	
	// iterate over the size of the local memory size
	for (int i = 1; i < local_size; i *= 2)
	{
		// local id doesnt return a remainder from modulous of i*2 AND local id + i is less than local size
		if (!(local_id % (i * 2)) && ((local_id + i) < local_size))
		{
			local_memory[local_id] += local_memory[local_id + i]; // adds all local memory into the first location
		}
		barrier(CLK_LOCAL_MEM_FENCE); // Local memory barrier, haults and waits for workgroup to finish
	}
	// checks if local id is 0
	if (local_id == 0)  
	{
		// uses atomic to go through N - number of workgroup sizes to add up all the sums of local memories
		atomic_add(&output_data[0], local_memory[local_id]);
	}
}

// REDUCTION OF THE VECTOR INTO A SINGLE LOCATION USING LOCAL MEMORY NOT USING ATOMIC_ADD FOR INTEGERS
kernel void Local_Reduction_NON_Atomic_Integer(global const int* input_data, global int* output_data, local int* local_memory) 
{
	int global_id = get_global_id(0); // id is 0 to n (size of A vector)
	int local_id = get_local_id(0); // local id is 0 to local_size (repeats for each work group)
	int local_size = get_local_size(0); // the size of the local memory, id uses this for knowing max
	int group_id = get_group_id(0); // gets the group id, each local memory is assigned a group id
	local_memory[local_id] = input_data[global_id]; // transfer of global memory to local memory
	barrier(CLK_LOCAL_MEM_FENCE); // Local memory barrier, haults and waits for workgroup to finish

	// iterate over the size of the local memory size
	for (int i = 1; i < local_size; i *= 2)
	{
		// local id doesnt return a remainder from modulous of i*2 AND local id + i is less than local size
		if (!(local_id % (i * 2)) && ((local_id + i) < local_size))
		{
			local_memory[local_id] += local_memory[local_id + i]; // adds all local memory into the first location 
		}
		barrier(CLK_LOCAL_MEM_FENCE); // Local memory barrier, haults and waits for workgroup to finish
	}
	// checks if local id is 0
	if (local_id == 0)
	{                                                                                                                                                                                                                                                                                                          
		output_data[group_id] = local_memory[local_id]; // transfer of local memory to output (global memory)
		barrier(CLK_LOCAL_MEM_FENCE); // Local memory barrier, haults and waits for workgroup to finish
		// checks if the global id is 0
		if (global_id == 0)
		{
			// loops through entirety of the number of work groups 
			for (int i = 1; i < get_num_groups(0); ++i)
			{
				output_data[global_id] += output_data[i]; // sums up all the values in ouput from 0 to number of workgroups 
			}
		}
	}
}


//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
//--------------------- MINIMUM VALUE INTEGERS --------------------------
//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
// FINDS THE MIN ELEMENT IN DATA SET USING ATOMIC_MIN FOR INTEGERS
kernel void Min_Kernel_atomic(__global const int* input_data, __global int* output_data, __local int* local_memory)
{
	int global_id = get_global_id(0); // id is 0 to n (size of A vector)
	int local_id = get_local_id(0); // local id is 0 to local_size (repeats for each work group)
	int local_size = get_local_size(0); // the size of the local memory, id uses this for knowing max
	local_memory[local_id] = input_data[global_id]; // transfer of global memory to local memory
	barrier(CLK_LOCAL_MEM_FENCE); // Local memory barrier, haults and waits for workgroup to finish

	// iterate over the size of the local memory size
	for (int i = 1; i < local_size; i *= 2)
	{
		// local id doesnt return a remainder from modulous of i*2 AND local id + i is less than local size
		if (!(local_id % (i * 2)) && ((local_id + i) < local_size))
		{
			// checks is greater than i to the right
			if (local_memory[local_id] > local_memory[local_id + i])
			{
				local_memory[local_id] = local_memory[local_id + i];
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE); // Local memory barrier, haults and waits for workgroup to finish
	}
	// checks if local id is 0
	if (local_id == 0)
	{
		// uses atomic to go through N - number of workgroup sizes to find the minimum of all the sums of local memories
		atomic_min(&output_data[0], local_memory[local_id]);
	}
}

// FINDS THE MIN ELEMENT IN DATA SET NOT USING ATOMIC_MIN FOR INTEGERS
kernel void Min_Kernel_non_atomic(__global const int* input_data, __global int* output_data, __local int* local_memory)
{
	int global_id = get_global_id(0); // id is 0 to n (size of A vector)
	int local_id = get_local_id(0); // local id is 0 to local_size (repeats for each work group)
	int local_size = get_local_size(0); // the size of the local memory, id uses this for knowing max
	int group_id = get_group_id(0); // gets the group id, each local memory is assigned a group id
	local_memory[local_id] = input_data[global_id]; // transfer of global memory to local memory
	barrier(CLK_LOCAL_MEM_FENCE); // Local memory barrier, haults and waits for workgroup to finish

	// iterate over the size of the local memory size
	for (int i = 1; i < local_size; i *= 2)
	{
		// local id doesnt return a remainder from modulous of i*2 AND local id + i is less than local size
		if (!(local_id % (i * 2)) && ((local_id + i) < local_size))
		{
			// checks is greater than i to the right
			if (local_memory[local_id] > local_memory[local_id + i])
			{
				local_memory[local_id] = local_memory[local_id + i]; // sets the new value
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE); // Local memory barrier, haults and waits for workgroup to finish
	}
	// checks if local id is 0
	if (local_id == 0)
	{
		output_data[group_id] = local_memory[local_id]; // transfer of local memory to output (global memory)
		barrier(CLK_LOCAL_MEM_FENCE); // Local memory barrier, haults and waits for workgroup to finish

		// checks if the global id is 0
		if (global_id == 0)
		{
			// loops through entirety of the number of work groups 
			for (int i = 1; i < get_num_groups(0); ++i)
			{
				// if true index global id == i
				if (output_data[global_id] > output_data[i])
				{
					output_data[global_id] = output_data[i]; // sets the new value 
				}
			}
		}
	}
}


//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
//--------------------- MAXIMUM VALUE INTEGERS --------------------------
//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
// FINDS THE MAX ELEMENT IN DATA SET USING ATOMIC_MAX FOR INTEGERS
kernel void Max_Kernel_atomic(__global const int* input_data, __global int* output_data, __local int* local_memory)
{
	int global_id = get_global_id(0); // id is 0 to n (size of A vector)
	int local_id = get_local_id(0); // local id is 0 to local_size (repeats for each work group)
	int local_size = get_local_size(0); // the size of the local memory, id uses this for knowing max
	local_memory[local_id] = input_data[global_id]; // transfer of global memory to local memory
	barrier(CLK_LOCAL_MEM_FENCE); // Local memory barrier, haults and waits for workgroup to finish

	// iterate over the size of the local memory size
	for (int i = 1; i < local_size; i *= 2)
	{
		// local id doesnt return a remainder from modulous of i*2 AND local id + i is less than local size
		if (!(local_id % (i * 2)) && ((local_id + i) < local_size))
		{
			// checks is less than i to the right
			if (local_memory[local_id] < local_memory[local_id + i])
			{
				local_memory[local_id] = local_memory[local_id + i]; // sets the new value 

			}
		}
		barrier(CLK_LOCAL_MEM_FENCE); // Local memory barrier, haults and waits for workgroup to finish
	}
	// checks if local id is 0
	if (local_id == 0)
	{
		// uses atomic to go through N - number of workgroup sizes to find the maximum of all the sums of local memories
		atomic_max(&output_data[0], local_memory[local_id]);
	}
}

// FINDS THE MAX ELEMENT IN DATA SET NOT USING ATOMIC_MAX FOR INTEGERS
kernel void Max_Kernel_non_atomic(__global const int* input_data, __global int* output_data, __local int* local_memory)
{
	int global_id = get_global_id(0); // id is 0 to n (size of A vector)
	int local_id = get_local_id(0); // local id is 0 to local_size (repeats for each work group)
	int local_size = get_local_size(0); // the size of the local memory, id uses this for knowing max
	int group_id = get_group_id(0); // gets the group id, each local memory is assigned a group id
	local_memory[local_id] = input_data[global_id]; // transfer of global memory to local memory
	barrier(CLK_LOCAL_MEM_FENCE); // Local memory barrier, haults and waits for workgroup to finish

	// iterate over the size of the local memory size
	for (int i = 1; i < local_size; i *= 2)
	{
		// local id doesnt return a remainder from modulous of i*2 AND local id + i is less than local size
		if (!(local_id % (i * 2)) && ((local_id + i) < local_size))
		{
			// checks is less than i to the right
			if (local_memory[local_id] < local_memory[local_id + i])
			{
				local_memory[local_id] = local_memory[local_id + i]; // sets the new value 
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE); // Local memory barrier, haults and waits for workgroup to finish
	}
	// checks if local id is 0
	if (local_id == 0)
	{
		output_data[group_id] = local_memory[local_id]; // transfer of local memory to output (global memory)
		barrier(CLK_LOCAL_MEM_FENCE); // Local memory barrier, haults and waits for workgroup to finish

		// checks if the global id is 0
		if (global_id == 0)
		{
			// loops through entirety of the number of work groups 
			for (int i = 1; i < get_num_groups(0); ++i)
			{
				// if true index global id == i
				if (output_data[global_id] < output_data[i])  
				{
					output_data[global_id] = output_data[i]; // sets the new value 
				}
			}
		}
	}
}


//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
//------------------------- REDUCTION FLOATS ----------------------------
//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
// REDUCTION OF THE VECTOR INTO A SINGLE LOCATION USING LOCAL MEMORY USING ATOMIC SORT FOR FLOATS 
kernel void float_Reduction_Kernel_mulitcall(global const float* input_data, global float* output_data, __local float *local_memory)
{
	int global_id = get_global_id(0); // id is 0 to n (size of A vector)
	int local_id = get_local_id(0); // local id is 0 to local_size (repeats for each work group)
	int group_id = get_group_id(0); // gets the group id, each local memory is assigned a group id
	int local_size = get_local_size(0); // the size of the local memory, id uses this for knowing max
	local_memory[local_id] = input_data[global_id]; // transfer of global memory to local memory
	barrier(CLK_LOCAL_MEM_FENCE); // Local memory barrier, haults and waits for workgroup to finish

	// iterate over the size of the local memory size
	for (int i = 1; i < local_size; i *= 2)
	{
		// local id doesnt return a remainder from modulous of i*2 AND local id + i is less than local size
		if (!(local_id % (i * 2)) && ((local_id + i) < local_size))
		{
			local_memory[local_id] += local_memory[local_id + i]; // adds all local memory into the first location
		}
		barrier(CLK_LOCAL_MEM_FENCE); // Local memory barrier, haults and waits for workgroup to finish
	}
	// checks if local id is 0
	if (local_id == 0)
	{                                                                     
		output_data[group_id] = local_memory[local_id]; // transfer of local memory to output (global memory)
		barrier(CLK_LOCAL_MEM_FENCE); // Local memory barrier, haults and waits for workgroup to finish
	}
}

// REDUCTION OF THE VECTOR INTO A SINGLE LOCATION USING LOCAL MEMORY USING ATOMIC SORT FOR FLOATS 
kernel void float_Reduction_Kernel(global const float* input_data, global float* output_data, __local float *local_memory)
{
	int global_id = get_global_id(0); // id is 0 to n (size of A vector)
	int local_id = get_local_id(0); // local id is 0 to local_size (repeats for each work group)
	int group_id = get_group_id(0); // gets the group id, each local memory is assigned a group id
	int local_size = get_local_size(0); // the size of the local memory, id uses this for knowing max
	local_memory[local_id] = input_data[global_id]; // transfer of global memory to local memory
	barrier(CLK_LOCAL_MEM_FENCE); // Local memory barrier, haults and waits for workgroup to finish

	// iterate over the size of the local memory size
	for (int i = 1; i < local_size; i *= 2)
	{
		// local id doesnt return a remainder from modulous of i*2 AND local id + i is less than local size
		if (!(local_id % (i * 2)) && ((local_id + i) < local_size))
		{
			local_memory[local_id] += local_memory[local_id + i]; // adds all local memory into the first location
		}
		barrier(CLK_LOCAL_MEM_FENCE); // Local memory barrier, haults and waits for workgroup to finish
	}
	// checks if local id is 0
	if (local_id == 0)
	{
		output_data[group_id] = local_memory[local_id]; // transfer of local memory to output (global memory)
		barrier(CLK_LOCAL_MEM_FENCE); // Local memory barrier, haults and waits for workgroup to finish

		// checks if the global id is 0
		if (global_id == 0)
		{
			// loops through entirety of the number of work groups 
			for (int i = 1; i < get_num_groups(0); ++i)
			{
				output_data[global_id] += output_data[i]; // sums up all the values in ouput from 0 to number of workgroups 
			}
		}
	}
}


//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
//----------------------- MINIMUM VALUE FLOATS --------------------------
//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
// FINDS THE MIN ELEMENT IN DATA SET NOT USING ATOMIC_MIN FOR FLOATS
kernel void Float_Reduction_Min(global const float* input_data, global float* output_data, __local float* local_memory)
{
	int global_id = get_global_id(0); // id is 0 to n (size of A vector)
	int local_id = get_local_id(0); // local id is 0 to local_size (repeats for each work group)
	int group_id = get_group_id(0); // gets the group id, each local memory is assigned a group id
	int local_size = get_local_size(0); // the size of the local memory, id uses this for knowing max
	local_memory[local_id] = input_data[global_id]; // transfer of global memory to local memory
	barrier(CLK_LOCAL_MEM_FENCE); // Local memory barrier, haults and waits for workgroup to finish
	
	// iterate over the size of the local memory size
	for (int i = 1; i < local_size; i *= 2) 
	{
		// local id doesnt return a remainder from modulous of i*2 AND local id + i is less than local size
		if (!(local_id % (i * 2)) && ((local_id + i) < local_size))
		{
			// chekcs is greater than i to the right
			if (local_memory[local_id] > local_memory[local_id + i])
			{
				local_memory[local_id] = local_memory[local_id + i]; // sets the new value
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE); // Local memory barrier, haults and waits for workgroup to finish
	}
	// checks if local id is 0
	if (local_id == 0)
	{
		output_data[group_id] = local_memory[local_id]; // transfer of local memory to output (global memory)
		barrier(CLK_LOCAL_MEM_FENCE); // Local memory barrier, haults and waits for workgroup to finish

		// checks if the global id is 0
		if (global_id == 0)
		{
			// loops through entirety of the number of work groups 
			for (int i = 1; i < get_num_groups(0); ++i)
			{
				// if true index global id == i
				if (output_data[global_id] > output_data[i])
				{
					output_data[global_id] = output_data[i]; // sets the new value 
				}
			}
		}
	}
}


//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
//----------------------- MAXIMUM VALUE FLOATS --------------------------
//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
// FINDS THE MAX ELEMENT IN DATA SET NOT USING ATOMIC_MAX FOR FLOATS
kernel void Float_Reduction_Max(global const float* input_data, global float* output_data, __local float* local_memory)
{
	int global_id = get_global_id(0); // id is 0 to n (size of A vector)
	int local_id = get_local_id(0); // local id is 0 to local_size (repeats for each work group)
	int group_id = get_group_id(0); // gets the group id, each local memory is assigned a group id
	int local_size = get_local_size(0); // the size of the local memory, id uses this for knowing max
	local_memory[local_id] = input_data[global_id]; // transfer of global memory to local memory
	barrier(CLK_LOCAL_MEM_FENCE); // Local memory barrier, haults and waits for workgroup to finish
	
	// iterate over the size of the local memory size
	for (int i = 1; i < local_size; i *= 2) 
	{
		// local id doesnt return a remainder from modulous of i*2 AND local id + i is less than local size
		if (!(local_id % (i * 2)) && ((local_id + i) < local_size))
		{
			// checks is less than i to the right
			if (local_memory[local_id] < local_memory[local_id + i]) 
			{
				local_memory[local_id] = local_memory[local_id + i]; // sets new value
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE); // Local memory barrier, haults and waits for workgroup to finish
	}
	// checks if local id is 0
	if (local_id == 0)
	{
		output_data[group_id] = local_memory[local_id]; // transfer of local memory to output (global memory)
		barrier(CLK_LOCAL_MEM_FENCE); // Local memory barrier, haults and waits for workgroup to finish

		// checks if the global id is 0
		if (global_id == 0)
		{
			// loops through entirety of the number of work groups 
			for (int i = 1; i < get_num_groups(0); ++i)
			{
				// if true index global id == i
				if (output_data[global_id] < output_data[i])  
				{
					output_data[global_id] = output_data[i]; // sets the new value 
				}
			}
		}
	}
}


//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
//----------------- STANDARD DEVIATION VALUE FLOATS ---------------------
//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
// FINDS THE STANDARD DEVIATION USING LOCAL MEMORY AND PUBLISHING TO A SINGLE LOCATION
kernel void float_Reduction_Dev_Kernel(global const float* input_data, global float* output_data, local float *local_memory, float average)
{
	int global_id = get_global_id(0); // id is 0 to n (size of A vector)
	int local_id = get_local_id(0); // local id is 0 to local_size (repeats for each work group)
	int group_id = get_group_id(0); // gets the group id, each local memory is assigned a group id
	int local_size = get_local_size(0); // the size of the local memory, id uses this for knowing max 
	float data_value = (input_data[global_id] - average);
	local_memory[local_id] = data_value * data_value;
	barrier(CLK_LOCAL_MEM_FENCE); // Local memory barrier, haults and waits for workgroup to finish

	// iterate over the size of the local memory size
	for (int i = 1; i < local_size; i *= 2)
	{
		// local id doesnt return a remainder from modulous of i*2 AND local id + i is less than local size
		if (!(local_id % (i * 2)) && ((local_id + i) < local_size))
		{
			local_memory[local_id] += local_memory[local_id + i]; // adds all local memory into the first location
		}
		barrier(CLK_LOCAL_MEM_FENCE); // Local memory barrier, haults and waits for workgroup to finish
	}
	// checks if local id is 0
	if (local_id == 0)
	{
		output_data[group_id] = local_memory[local_id]; // transfer of local memory to output (global memory)
		barrier(CLK_LOCAL_MEM_FENCE); // Local memory barrier, haults and waits for workgroup to finish

		// checks if the global id is 0
		if (global_id == 0)
		{
			// loops through entirety of the number of work groups 
			for (int i = 1; i < get_num_groups(0); ++i)
			{
				output_data[global_id] += output_data[i]; // sums up all the values in ouput from 0 to number of workgroups 
			}
		}
	}
}


//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
//-------------------------- SORTING FLOATS -----------------------------
//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
//-------------------- ODD EVEN TRANSPOSITION SORT ----------------------
//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
// ODD SORTING PARTION
kernel void bubble_odd_sort_kernel(global float* input_data)
{
	int global_id = get_global_id(0); // id is 0 to n (size of A vector)
	int global_size = get_global_size(0); // the size of the input_data (global memory size)

	// if greater && and the global id is odd && global id + 1 is less global size 
	if (input_data[global_id] > input_data[global_id + 1] && (global_id % 2 != 0) && ((global_id + 1) < global_size))
	{
		// swapping values
		float temp = input_data[global_id];  
		input_data[global_id] = input_data[global_id + 1];
		input_data[global_id + 1] = temp;
	}
}

// EVEN SORTING PARTION
kernel void bubble_even_sort_kernel(global float* input_data)
{
	int global_id = get_global_id(0); // id is 0 to n (size of A vector)
	int global_size = get_global_size(0); // the size of the input_data (global memory size)

	// if greater && and the global id is even && global id + 1 is less global size 
	if (input_data[global_id] > input_data[global_id + 1] && (global_id % 2 == 0) && ((global_id + 1) < global_size))
	{
		// swapping values
		float temp = input_data[global_id];  
		input_data[global_id] = input_data[global_id + 1];
		input_data[global_id + 1] = temp;
	}
}

// DOESNT WORK FOR BIG DATA NEEDED SLITTING 
kernel void bubble_odd_even_sort_kernel(global float* input_data)
{
	int global_id = get_global_id(0); // id is 0 to n (size of A vector)
	int global_size = get_global_size(0); // the size of the input_data (global memory size)

	// if greater && and the global id is even && global id + 1 is less global size 
	if (input_data[global_id] > input_data[global_id + 1] && (global_id % 2 == 0) && ((global_id + 1) < global_size))
	{
		// swapping values
		float temp = input_data[global_id];  
		input_data[global_id] = input_data[global_id + 1];
		input_data[global_id + 1] = temp;
	}
	barrier(CLK_LOCAL_MEM_FENCE); // Local memory barrier, haults and waits for workgroup to finish

	// if greater && and the global id is odd && global id + 1 is less global size 
	if (input_data[global_id] > input_data[global_id + 1] && (global_id % 2 != 0) && ((global_id + 1) < global_size))
	{
		// swapping values
		float temp = input_data[global_id]; 
		input_data[global_id] = input_data[global_id + 1];
		input_data[global_id + 1] = temp;
	}
	barrier(CLK_LOCAL_MEM_FENCE); // Local memory barrier, haults and waits for workgroup to finish
}

/*
ATTEMPTED FIRST PHASE OF SHELLSORT
kernel void flip_adjacent(global float* input_data, local float *local_memory)
{
	// Get the global id of the work item (thread)
	int id = get_global_id(0);
	// Get the local id of the work item
	int local_id = get_local_id(0); // local id is 0 to local_size (repeats for each work group)
	// Get the group id of the work item
	int group_id = get_group_id(0); // gets the group id, each local memory is assigned a group id
	// Get the amount of work items
	int local_size = get_local_size(0); // the size of the local memory, id uses this for knowing max

	int gs = get_global_size(0); 

	if (id < (gs / 2))
	{
		if ((local_id< (N / 2)))
		{
			local_memory[local_id] = input_data[(group_id * (N / 2) + local_id)];
			local_memory[local_id+ (N / 2)] = input_data[(gs - (group_id * ((N / 2) + local_id)))];
			//barrier(CLK_LOCAL_MEM_FENCE); // Local memory barrier, haults and waits for workgroup to finish
			if (local_memory[local_id] > local_memory[local_id+ (N / 2)])
			{
				float temp = local_memory[local_id];
				local_memory[local_id] = local_memory[local_id+ (N / 2)];
				local_memory[local_id+ (N / 2)] = temp;
			}

			input_data[(group_id * (N / 2) + local_id)] = local_memory[local_id];
			input_data[(gs - (group_id * ((N / 2) + local_id)))] = local_memory[local_id+ (N / 2)];
			//barrier(CLK_LOCAL_MEM_FENCE); // Local memory barrier, haults and waits for workgroup to finish
		}
	}
}
*/

/*
ATTEMPTED SLIP FOR ODD EVEN TRANSPOSITION OPTIMISATION
kernel void flip_adjacent(global float* input_data)
{
	int id = get_global_id(0);
	int gs = get_global_size(0);

	if (id < (gs / 2))
	{
		float temp = input_data[id];
		input_data[id] = input_data[(gs - id) - 1];
		input_data[(gs - id) - 1] = temp;
	}
}
*/


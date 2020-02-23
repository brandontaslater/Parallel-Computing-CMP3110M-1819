#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <vector>

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include <chrono>
#include "Utils.h"
#include <math.h>
#include <algorithm> 
#include <chrono> 
#include <iostream>


typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::high_resolution_clock::time_point TimePoint;

///
/// Function takes the filepath and uses ifstream to read in the contents of that file.
/// The contents are read as a string but then the final column of data (after the 5th space character) is parsed to a float
/// The float is * by 100 and saved as an int so it can be passed into OpenCL kernels and still retain the decimal place data
///

// read the file as an integer (done through multiplying the float by 100 to remove the decimals)
vector<int> readFile(string filename)
{
	// stores the temperatures for all data
	std::vector<int> input_data;
	// stream the data through the file path passed to function
	ifstream infile(filename);
	// Run through each line of the file and store it as a string. 
	// Then parse the string to get the 6th column of data and return those values
	string LOCATION, YEAR, DAY, MONTH, TIME;
	float TEMPERATURE;
	
	// iterate through each line and store the temperature
	while (infile >> LOCATION >> YEAR >> DAY >> MONTH >> TIME >> TEMPERATURE)
	{
		input_data.push_back(TEMPERATURE * 100); // push data to top of file
	}
	// close file
	infile.close();
	// return
	return input_data;
}
/*
// read the file as a float
vector<float> read_file_float(string filename)
{
	// stores the temperatures for all data
	vector<float> data;
	// stream the data through the file path passed to function
	ifstream infile(filename);
	// Run through each line of the file and store it as a string. 
	// Then parse the string to get the 6th column of data and return those values
	string LOCATION, YEAR, DAY, MONTH, TIME;
	float TEMPERATURE;
	// iterate through each line and store the temperature
	while (infile >> LOCATION >> YEAR >> DAY >> MONTH >> TIME >> TEMPERATURE)
	{
		data.push_back(TEMPERATURE); // push data to top of file
	}
	// close file
	infile.close();
	// return
	return data;
}
*/

// read the file as a float
void read_file_float(string filename, vector<float>* data_float1, vector<float>* data_float2, vector<int>* data_int)
{
	// stores the temperatures for all data
	// stream the data through the file path passed to function
	ifstream infile(filename);
	// Run through each line of the file and store it as a string. 
	// Then parse the string to get the 6th column of data and return those values
	string LOCATION, YEAR, DAY, MONTH, TIME;
	float TEMPERATURE;
	// iterate through each line and store the temperature
	while (infile >> LOCATION >> YEAR >> DAY >> MONTH >> TIME >> TEMPERATURE)
	{
		(*data_float1).push_back(TEMPERATURE); // push data to top of file
		(*data_float2).push_back(TEMPERATURE); // push data to top of file
		(*data_int).push_back(TEMPERATURE*100); // push data to top of file
	}
	// close file
	infile.close();
}


void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

// calculates the standard deviation SQUENTIALLY
void standard_deviation(vector<float> data_set, float _mean, float* _sum)
{
	float std = 0.0;
	float sum = 0.0;
	for (int i = 0; i < data_set.size(); i++)
	{
		std = (data_set[i] - _mean);
		std = std * std;
		sum += std;
	}
	*_sum = sum / data_set.size();
}

// calculates the mean SQUENTIALLY
void mean(vector<float> data_set, float* _mean)
{
	float sum = 0.0;
	for (int i = 0; i < data_set.size(); i++)
	{
		sum = sum + data_set[i];
	}
	*_mean = sum / data_set.size();
}

// calculates the min and max SQUENTIALLY
void min_max(vector<float> data_set, float* _min, float* _max)
{
	float min = data_set[0];
	float max = data_set[0];
	for (int i = 1; i < data_set.size(); i++)
	{
		if (min > data_set[i])
		{
			min = data_set[i];
		}
		if (max < data_set[i])
		{
			max = data_set[i];
		}

		*_min = min;
		*_max = max;
	}
}

// A function to sort the algorithm using Odd Even sort 
//REFERENCE: https://www.geeksforgeeks.org/odd-even-sort-brick-sort/
void oddEvenSort(vector<float> arr, int n)
{
	bool isSorted = false; // Initially array is unsorted 

	while (!isSorted)
	{
		isSorted = true;

		// Perform Bubble sort on odd indexed element 
		for (int i = 1; i <= n - 2; i = i + 2)
		{
			if (arr[i] > arr[i + 1])
			{
				swap(arr[i], arr[i + 1]);
				isSorted = false;
			}
		}

		// Perform Bubble sort on even indexed element 
		for (int i = 0; i <= n - 2; i = i + 2)
		{
			if (arr[i] > arr[i + 1])
			{
				swap(arr[i], arr[i + 1]);
				isSorted = false;
			}
		}
	}

	return;
}

int main(int argc, char **argv) 
{
	//=====================================================================================
	//=====================================================================================
	//============================== HOST OPERTATIONS =====================================
	//=====================================================================================
	//=====================================================================================

	// assign the file path for the short and long .txt file
	std::string fileName_long = "temp_lincolnshire.txt";
	std::string fileName_short = "temp_lincolnshire_short.txt";
	
	// Read in file	
	std::vector<int> data_int;
	std::vector<float> data_float;
	std::vector<float> data_float_Zero_Pad;
	read_file_float(fileName_long, &data_float, &data_float_Zero_Pad, &data_int);

	// stores the lengths of each of the int and float data sets
	int dat_int_original_size = data_int.size();
	int dat_float_original_size = data_float.size();

	//=====================================================================================
	//=====================================================================================
	//==================================Squential TESTING==================================
	//=====================================================================================
	//=====================================================================================

	std::cout << "===============================================" << std::endl;
	std::cout << "===============================================" << std::endl;
	std::cout << "============== SQUENTIAL TESTING ==============" << std::endl;
	std::cout << "===============================================" << std::endl;
	std::cout << "===============================================" << std::endl;

	// initialisation of vars
	float _mean, _min, _max, _sum;
	auto start = std::chrono::high_resolution_clock::now();
	auto finish = std::chrono::high_resolution_clock::now();
	chrono::duration<double> elapsed;

	// mean calculation sequential
	//=====================================================================================
	start = std::chrono::high_resolution_clock::now();// start timer
	mean(data_float, &_mean); 
	finish = std::chrono::high_resolution_clock::now();	// end timer
	elapsed = finish - start;

	// OUTPUTS 
	std::cout << "" << std::endl;
	std::cout << "MEAN FLOAT: " << _mean << std::endl;
	std::cout << "Elapsed time: " << elapsed.count() << " s\n";
	std::cout << "===============================================" << std::endl;
	std::cout << "" << std::endl;
	//=====================================================================================


	// min max calculation sequential
	//=====================================================================================
	// start timer
	start = std::chrono::high_resolution_clock::now();
	min_max(data_float, &_min, &_max); 
	finish = std::chrono::high_resolution_clock::now();
	elapsed = finish - start;
	// end timer

	// OUTPUTS 
	std::cout << "" << std::endl;
	std::cout << "MIN FLOAT: " << _min << std::endl;
	std::cout << "Elapsed time: " << elapsed.count() << " s\n";
	std::cout << "===============================================" << std::endl;
	std::cout << "" << std::endl;
	std::cout << "" << std::endl;
	std::cout << "MAX FLOAT: " << _max << std::endl;
	std::cout << "Elapsed time: " << elapsed.count() << " s\n";
	std::cout << "===============================================" << std::endl;
	std::cout << "" << std::endl;
	//=====================================================================================

	// standard deviation calculation sequential
	//=====================================================================================
	// start timer
	start = std::chrono::high_resolution_clock::now();
	standard_deviation(data_float, _mean, &_sum); 
	finish = std::chrono::high_resolution_clock::now();
	elapsed = finish - start;
	// end timer

	// OUTPUTS 
	std::cout << "" << std::endl;
	std::cout << "STD FLOAT: " << sqrt(_sum) << std::endl;
	std::cout << "Elapsed time: " << elapsed.count() << " s\n";
	std::cout << "===============================================" << std::endl;
	std::cout << "" << std::endl;	
	std::cout << "" << std::endl;
	//=====================================================================================

	// odd even sort sequential
	//=====================================================================================
	// start timer
	start = std::chrono::high_resolution_clock::now();
	//oddEvenSort(data_float, data_float.size());
	finish = std::chrono::high_resolution_clock::now();
	elapsed = finish - start;
	// end timer

	// OUTPUTS 
	std::cout << "" << std::endl;
	std::cout << "Sorted " << data_float.size() << "floats" <<std::endl;
	std::cout << "Elapsed time: " << elapsed.count() << " s\n";
	std::cout << "===============================================" << std::endl;
	std::cout << "" << std::endl;
	std::cout << "" << std::endl;
	//=====================================================================================

	// CPU GPU 
	int platform_id = 0;
	
	// device id on CPU/GPU 
	int device_id = 0;

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); }
	}

	// tries the code, catches exceptions
	try {
		// initialisation of the host open cl operations
		//=====================================================================================
		cl::Context context = GetContext(platform_id, device_id); // Select computing devices
		std::cout << "Runinng on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl; // display the selected device
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);	//create a queue to which we will push commands for the device
		// Load & build the device code
		cl::Program::Sources sources; 		
		AddSources(sources, "my_kernels_3.cl");
		cl::Program program(context, sources);

		// build and debug the kernel code
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		// initialise the local size for local memory
		size_t local_size = 256;

		//=====================================================================================
		//=====================================================================================
		//====================================INTEGERS=========================================
		//=====================================================================================
		//=====================================================================================
		std::cout << "" << std::endl;
		std::cout << "===============================================" << std::endl;
		std::cout << "===============================================" << std::endl;
		std::cout << "================ PARALLEL =====================" << std::endl;
		std::cout << "============ INTEGERS TESTING =================" << std::endl;
		std::cout << "===============================================" << std::endl;

		// initialisation of the padding size required
		size_t padding_size_int = data_int.size() % local_size;

		//if the input vector is not a multiple of the local_size
		//insert additional neutral elements (0 for addition) so that the total will not be affected
		if (padding_size_int) {
			//create an extra vector with neutral values
			std::vector<int> A_ext(local_size - padding_size_int, 0);
			//append that extra vector to our input
			data_int.insert(data_int.end(), A_ext.begin(), A_ext.end());
		}

		// initialises the the sizes, in bytes and outputs 
		size_t input_int_elements = data_int.size();
		size_t input_int_size = data_int.size() * sizeof(int);
		size_t nr_groups_int = input_int_elements / local_size;
		size_t output_size;
		size_t local_size_bytes_integer = local_size * sizeof(float);

		// the buffer used for storing the data set for parsing the memory block to kernel
		cl::Buffer buffer_int(context, CL_MEM_READ_ONLY, input_int_size);

		//=====================================================================================
		// SUM - MEAN - INTEGERS - Non ATOMIC
		//=====================================================================================
		std::cout << "" << std::endl;
		std::cout << "" << std::endl;
		std::cout << "===============================================" << std::endl;
		std::cout << "===============================================" << std::endl;
		std::cout << "SUM - MEAN - INTEGERS - NON - ATOMIC" << std::endl;
		std::cout << "" << std::endl;

		// used for profilling the Write, Kernel and Read queue
		cl::Event Integer_mean_sum_non_atomic_Write; 
		cl::Event Integer_mean_sum_non_atomic;
		cl::Event Integer_mean_sum_non_atomic_Read;

		// stores the number vector of 1 size
		std::vector<int> Int_Mean_Local_Non_Atomic(1);
		// the output size of the above vector in bytes
		output_size = Int_Mean_Local_Non_Atomic.size() * sizeof(int);

		// creates the buffer for the interface for converting data between host and device
		cl::Buffer Buffer_Int_Mean_Local_Non_Atomic(context, CL_MEM_READ_WRITE, output_size);

		// allows the buffer object to be written from host memory
		queue.enqueueWriteBuffer(buffer_int, CL_TRUE, 0, input_int_size, &data_int[0], NULL, &Integer_mean_sum_non_atomic_Write);

		// setups a kernel and initialises the arguments for parsing
		// parses: buffer used for the data read in, the new buffer for outputting the sum and local memroy 
		cl::Kernel Reduction_Int_Local_non_atomic = cl::Kernel(program, "Local_Reduction_NON_Atomic_Integer");
		Reduction_Int_Local_non_atomic.setArg(0, buffer_int);
		Reduction_Int_Local_non_atomic.setArg(1, Buffer_Int_Mean_Local_Non_Atomic);
		Reduction_Int_Local_non_atomic.setArg(2, cl::Local(local_size * sizeof(int)));

		// enques the kernel to be executes on the device
		queue.enqueueNDRangeKernel(Reduction_Int_Local_non_atomic, cl::NullRange, cl::NDRange(input_int_elements), cl::NDRange(local_size), NULL, &Integer_mean_sum_non_atomic);
		
		// reads from the buffer object into host memory after kernel processing complete
		queue.enqueueReadBuffer(Buffer_Int_Mean_Local_Non_Atomic, CL_TRUE, 0, output_size, &Int_Mean_Local_Non_Atomic[0], NULL, &Integer_mean_sum_non_atomic_Read);

		// displays the write data time, and memory transfer
		std::cout << "Write execution time [ns]:" <<
			Integer_mean_sum_non_atomic_Write.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			Integer_mean_sum_non_atomic_Write.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << GetFullProfilingInfo(Integer_mean_sum_non_atomic_Write, ProfilingResolution::PROF_US) << endl;
		std::cout << "" << std::endl;
		std::cout << "" << std::endl;

		// displays the Kernel data time, and memory transfer
		std::cout << "Kernel execution time [ns]:" <<
			Integer_mean_sum_non_atomic.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			Integer_mean_sum_non_atomic.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << GetFullProfilingInfo(Integer_mean_sum_non_atomic, ProfilingResolution::PROF_US) << endl;
		std::cout << "" << std::endl;
		std::cout << "" << std::endl;

		// displays the read data time, and memory transfer
		std::cout << "Read execution time [ns]:" <<
			Integer_mean_sum_non_atomic_Read.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			Integer_mean_sum_non_atomic_Read.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << GetFullProfilingInfo(Integer_mean_sum_non_atomic_Read, ProfilingResolution::PROF_US) << endl;
		std::cout << "" << std::endl;
		std::cout << "" << std::endl;

		std::cout << "Size in bytes for vectors: " << std::endl;
		std::cout << "data_int: " << input_int_size << std::endl;
		std::cout << "Int_Mean_Local_Non_Atomic: " << output_size << std::endl;
		std::cout << "Local memory: " << local_size_bytes_integer << std::endl;
		std::cout << "" << std::endl;
		std::cout << "" << std::endl;
		
		std::cout << "\nSum: " << ((Int_Mean_Local_Non_Atomic[0])) << std::endl;
		std::cout << "\nMean: " << ((Int_Mean_Local_Non_Atomic[0] / 100) / input_int_elements) << std::endl;


		//=====================================================================================
		// SUM - MEAN - INTEGERS - ATOMIC 
		//=====================================================================================
		std::cout << "" << std::endl;
		std::cout << "" << std::endl;
		std::cout << "===============================================" << std::endl;
		std::cout << "===============================================" << std::endl;
		std::cout << "SUM - MEAN - INTEGERS - ATOMIC" << std::endl;
		std::cout << "" << std::endl;

		// used for profilling the Write, Kernel and Read queue
		cl::Event Integer_mean_sum_Read;
		cl::Event Integer_mean_sum;
		cl::Event Integer_mean_sum_Write;

		// stores the number vector of 1 size
		std::vector<int> Int_Mean_Local(1);
		// the output size of the above vector in bytes
		output_size = Int_Mean_Local.size() * sizeof(int);

		// creates the buffer for the interface for converting data between host and device
		cl::Buffer Buffer_Int_Mean_Local(context, CL_MEM_READ_WRITE, output_size);

		// allows the buffer object to be written from host memory
		queue.enqueueWriteBuffer(buffer_int, CL_TRUE, 0, input_int_size, &data_int[0], NULL, &Integer_mean_sum_Write);

		// setups a kernel and initialises the arguments for parsing
		// parses: buffer used for the data read in, the new buffer for outputting the sum and local memroy 
		cl::Kernel Reduction_Int_Local = cl::Kernel(program, "Local_Atomic_Reduction_Integer");
		Reduction_Int_Local.setArg(0, buffer_int);
		Reduction_Int_Local.setArg(1, Buffer_Int_Mean_Local);
		Reduction_Int_Local.setArg(2, cl::Local(local_size * sizeof(int)));

		// enques the kernel to be executes on the device
		queue.enqueueNDRangeKernel(Reduction_Int_Local, cl::NullRange, cl::NDRange(input_int_elements), cl::NDRange(local_size), NULL, &Integer_mean_sum);

		// reads from the buffer object into host memory after kernel processing complete
		queue.enqueueReadBuffer(Buffer_Int_Mean_Local, CL_TRUE, 0, output_size, &Int_Mean_Local[0], NULL, &Integer_mean_sum_Read);

		// displays the write data time, and memory transfer
		std::cout << "Write execution time [ns]:" <<
			Integer_mean_sum_Write.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			Integer_mean_sum_Write.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << GetFullProfilingInfo(Integer_mean_sum_Write, ProfilingResolution::PROF_US) << endl;
		std::cout << "" << std::endl;
		std::cout << "" << std::endl;

		// displays the Kernel data time, and memory transfer
		std::cout << "Kernel execution time [ns]:" <<
			Integer_mean_sum.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			Integer_mean_sum.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << GetFullProfilingInfo(Integer_mean_sum, ProfilingResolution::PROF_US) << endl;
		std::cout << "" << std::endl;
		std::cout << "" << std::endl;

		// displays the read data time, and memory transfer
		std::cout << "Read execution time [ns]:" <<
			Integer_mean_sum_Read.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			Integer_mean_sum_Read.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << GetFullProfilingInfo(Integer_mean_sum_Read, ProfilingResolution::PROF_US) << endl;
		std::cout << "" << std::endl;
		std::cout << "" << std::endl;

		std::cout << "Size in bytes for vectors: " << std::endl;
		std::cout << "data_int vector (file converted to integers): " << input_int_size << std::endl;
		std::cout << "Int_Mean_Local vector (file converted to integers): " << output_size << std::endl;
		std::cout << "Local memory: " << local_size_bytes_integer << std::endl;
		std::cout << "" << std::endl;
		std::cout << "" << std::endl;

		std::cout << "\nSum: " << ((Int_Mean_Local[0])) << std::endl;
		std::cout << "\nMean: " << ((Int_Mean_Local[0] / 100) / input_int_elements) << std::endl;


		//=====================================================================================
		// MIN - INTEGERS - ATOMIC 
		//=====================================================================================
		std::cout << "" << std::endl;
		std::cout << "" << std::endl;
		std::cout << "===============================================" << std::endl;
		std::cout << "===============================================" << std::endl;
		std::cout << "MIN - INTEGERS - ATOMIC" << std::endl;
		std::cout << "" << std::endl;

		// used for profilling the Write, Kernel and Read queue
		cl::Event Integer_min_atomic_Read;
		cl::Event Integer_min_atomic;
		cl::Event Integer_min_atomic_Write;

		// stores the number vector of 1 size
		std::vector<int> Int_Min_Local(1);
		// the output size of the above vector in bytes
		output_size = Int_Min_Local.size() * sizeof(int);

		// creates the buffer for the interface for converting data between host and device
		cl::Buffer Buffer_Int_Min_Local_atomic(context, CL_MEM_READ_WRITE, output_size);

		// allows the buffer object to be written from host memory
		queue.enqueueWriteBuffer(buffer_int, CL_TRUE, 0, input_int_size, &data_int[0], NULL, &Integer_min_atomic_Write);

		// setups a kernel and initialises the arguments for parsing
		// parses: buffer used for the data read in, the new buffer for outputting the sum and local memroy
		cl::Kernel Minimum_Int_Local = cl::Kernel(program, "Min_Kernel_atomic");
		Minimum_Int_Local.setArg(0, buffer_int);
		Minimum_Int_Local.setArg(1, Buffer_Int_Min_Local_atomic);
		Minimum_Int_Local.setArg(2, cl::Local(local_size * sizeof(int)));

		// enques the kernel to be executes on the device
		queue.enqueueNDRangeKernel(Minimum_Int_Local, cl::NullRange, cl::NDRange(input_int_elements), cl::NDRange(local_size), NULL, &Integer_min_atomic);

		// enques the kernel to be executes on the device
		queue.enqueueReadBuffer(Buffer_Int_Min_Local_atomic, CL_TRUE, 0, output_size, &Int_Min_Local[0], NULL, &Integer_min_atomic_Read);

		// displays the write data time, and memory transfer
		std::cout << "Write execution time [ns]:" <<
			Integer_min_atomic_Write.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			Integer_min_atomic_Write.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << GetFullProfilingInfo(Integer_min_atomic_Write, ProfilingResolution::PROF_US) << endl;
		std::cout << "" << std::endl;
		std::cout << "" << std::endl;

		// displays the Kernel data time, and memory transfer
		std::cout << "Kernel execution time [ns]:" <<
			Integer_min_atomic.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			Integer_min_atomic.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << GetFullProfilingInfo(Integer_min_atomic, ProfilingResolution::PROF_US) << endl;
		std::cout << "" << std::endl;
		std::cout << "" << std::endl;

		// displays the read data time, and memory transfer
		std::cout << "Read execution time [ns]:" <<
			Integer_min_atomic_Read.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			Integer_min_atomic_Read.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << GetFullProfilingInfo(Integer_min_atomic_Read, ProfilingResolution::PROF_US) << endl;
		std::cout << "" << std::endl;
		std::cout << "" << std::endl;

		std::cout << "Size in bytes for vectors: " << std::endl;
		std::cout << "data_int vector: " << input_int_size << std::endl;
		std::cout << "Int_Min_Local vector: " << output_size << std::endl;
		std::cout << "Local memory: " << local_size_bytes_integer << std::endl;
		std::cout << "" << std::endl;
		std::cout << "" << std::endl;

		std::cout << "\nMinimum: " << ((Int_Min_Local[0] / 100)) << std::endl;


		//=====================================================================================
		// MIN - INTEGERS - NON - ATOMIC 
		//=====================================================================================
		std::cout << "" << std::endl;
		std::cout << "" << std::endl;
		std::cout << "===============================================" << std::endl;
		std::cout << "===============================================" << std::endl;
		std::cout << "MIN - INTEGERS - NON - ATOMIC" << std::endl;
		std::cout << "" << std::endl;

		// used for profilling the Write, Kernel and Read queue
		cl::Event Integer_min_non_atomic_Write;
		cl::Event Integer_min_non_atomic;
		cl::Event Integer_min_non_atomic_Read;

		// stores the number vector of 1 size
		std::vector<int> Int_Min_Local_non_atomic(1);
		// the output size of the above vector in bytes
		output_size = Int_Min_Local_non_atomic.size() * sizeof(int);

		// creates the buffer for the interface for converting data between host and devices
		cl::Buffer Buffer_Int_Min_Local_non_atomic(context, CL_MEM_READ_WRITE, output_size);

		// allows the buffer object to be written from host memory
		queue.enqueueWriteBuffer(buffer_int, CL_TRUE, 0, input_int_size, &data_int[0], NULL, &Integer_min_non_atomic_Write);
	
		// setups a kernel and initialises the arguments for parsing
		// parses: buffer used for the data read in, the new buffer for outputting the sum and local memroy
		cl::Kernel Minimum_Int_Local_non_atomic = cl::Kernel(program, "Min_Kernel_non_atomic");
		Minimum_Int_Local_non_atomic.setArg(0, buffer_int);
		Minimum_Int_Local_non_atomic.setArg(1, Buffer_Int_Min_Local_non_atomic);
		Minimum_Int_Local_non_atomic.setArg(2, cl::Local(local_size * sizeof(int)));

		// enques the kernel to be executes on the device
		queue.enqueueNDRangeKernel(Minimum_Int_Local_non_atomic, cl::NullRange, cl::NDRange(input_int_elements), cl::NDRange(local_size), NULL, &Integer_min_non_atomic);

		// enques the kernel to be executes on the device
		queue.enqueueReadBuffer(Buffer_Int_Min_Local_non_atomic, CL_TRUE, 0, output_size, &Int_Min_Local_non_atomic[0], NULL, &Integer_min_non_atomic_Read);

		// displays the write data time, and memory transfer
		std::cout << "Write execution time [ns]:" <<
			Integer_min_non_atomic_Write.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			Integer_min_non_atomic_Write.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << GetFullProfilingInfo(Integer_min_non_atomic_Write, ProfilingResolution::PROF_US) << endl;
		std::cout << "" << std::endl;
		std::cout << "" << std::endl;

		// displays the Kernel data time, and memory transfer
		std::cout << "Kernel execution time [ns]:" <<
			Integer_min_non_atomic.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			Integer_min_non_atomic.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << GetFullProfilingInfo(Integer_min_non_atomic, ProfilingResolution::PROF_US) << endl;
		std::cout << "" << std::endl;
		std::cout << "" << std::endl;

		// displays the read data time, and memory transfer
		std::cout << "Read execution time [ns]:" <<
			Integer_min_non_atomic_Read.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			Integer_min_non_atomic_Read.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << GetFullProfilingInfo(Integer_min_non_atomic_Read, ProfilingResolution::PROF_US) << endl;
		std::cout << "" << std::endl;
		std::cout << "" << std::endl;

		std::cout << "Size in bytes for vectors: " << std::endl;
		std::cout << "data_int vector: " << input_int_size << std::endl;
		std::cout << "Int_Min_Local_non_atomic vector: " << output_size << std::endl;
		std::cout << "Local memory: " << local_size_bytes_integer << std::endl;
		std::cout << "" << std::endl;
		std::cout << "" << std::endl;

		std::cout << "\nMinimum: " << ((Int_Min_Local_non_atomic[0] / 100)) << std::endl;


		//=====================================================================================
		// MAX - INTEGERS - ATOMIC
		//=====================================================================================
		std::cout << "" << std::endl;
		std::cout << "" << std::endl;
		std::cout << "===============================================" << std::endl;
		std::cout << "===============================================" << std::endl;
		std::cout << "MAX - INTEGERS - ATOMIC" << std::endl;
		std::cout << "" << std::endl;

		// used for profilling the Write, Kernel and Read queue
		cl::Event Integer_max_atomic_Write;
		cl::Event Integer_max_atomic;
		cl::Event Integer_max_atomic_Read;

		// stores the number vector of 1 size
		std::vector<int> Int_Max_Local_Atomic(1);
		// the output size of the above vector in bytes
		output_size = Int_Max_Local_Atomic.size() * sizeof(int);

		// creates the buffer for the interface for converting data between host and device
		cl::Buffer Buffer_Int_Max_Local_Atomic(context, CL_MEM_READ_WRITE, output_size);

		// allows the buffer object to be written from host memorys
		queue.enqueueWriteBuffer(buffer_int, CL_TRUE, 0, input_int_size, &data_int[0], NULL, &Integer_max_atomic_Write);

		// setups a kernel and initialises the arguments for parsing
		// parses: buffer used for the data read in, the new buffer for outputting the sum and local memroy 
		cl::Kernel Maximum_Int_Local_atomic = cl::Kernel(program, "Max_Kernel_atomic");
		Maximum_Int_Local_atomic.setArg(0, buffer_int);
		Maximum_Int_Local_atomic.setArg(1, Buffer_Int_Max_Local_Atomic);
		Maximum_Int_Local_atomic.setArg(2, cl::Local(local_size * sizeof(int)));

		// enques the kernel to be executes on the device
		queue.enqueueNDRangeKernel(Maximum_Int_Local_atomic, cl::NullRange, cl::NDRange(input_int_elements), cl::NDRange(local_size), NULL, &Integer_max_atomic);

		// enques the kernel to be executes on the device
		queue.enqueueReadBuffer(Buffer_Int_Max_Local_Atomic, CL_TRUE, 0, output_size, &Int_Max_Local_Atomic[0], NULL, &Integer_max_atomic_Read);

		// displays the write data time, and memory transfer
		std::cout << "Write execution time [ns]:" <<
			Integer_max_atomic_Write.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			Integer_max_atomic_Write.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << GetFullProfilingInfo(Integer_max_atomic_Write, ProfilingResolution::PROF_US) << endl;
		std::cout << "" << std::endl;
		std::cout << "" << std::endl;

		// displays the Kernel data time, and memory transfer
		std::cout << "Kernel execution time [ns]:" <<
			Integer_max_atomic.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			Integer_max_atomic.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << GetFullProfilingInfo(Integer_max_atomic, ProfilingResolution::PROF_US) << endl;
		std::cout << "" << std::endl;
		std::cout << "" << std::endl;

		// displays the read data time, and memory transfer
		std::cout << "Read execution time [ns]:" <<
			Integer_max_atomic_Read.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			Integer_max_atomic_Read.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << GetFullProfilingInfo(Integer_max_atomic_Read, ProfilingResolution::PROF_US) << endl;
		std::cout << "" << std::endl;
		std::cout << "" << std::endl;

		std::cout << "Size in bytes for vectors: " << std::endl;
		std::cout << "data_int vector: " << input_int_size << std::endl;
		std::cout << "Int_Max_Local_Atomic vector: " << output_size << std::endl;
		std::cout << "" << std::endl;
		std::cout << "" << std::endl;

		std::cout << "\nMaximum: " << ((Int_Max_Local_Atomic[0] / 100)) << std::endl;


		//=====================================================================================
		// MAX - INTEGERS - NON - ATOMIC
		//=====================================================================================
		std::cout << "" << std::endl;
		std::cout << "" << std::endl;
		std::cout << "===============================================" << std::endl;
		std::cout << "===============================================" << std::endl;
		std::cout << "MAX - INTEGERS - NON - ATOMIC" << std::endl;
		std::cout << "" << std::endl;

		// used for profilling the Write, Kernel and Read queue
		cl::Event Integer_max_non_atomic_Write;
		cl::Event Integer_max_non_atomic;
		cl::Event Integer_max_non_atomic_Read;

		// stores the number vector of 1 size
		std::vector<int> Int_Max_Local_non_atomic(1);
		// the output size of the above vector in bytes
		output_size = Int_Max_Local_non_atomic.size() * sizeof(int);

		// creates the buffer for the interface for converting data between host and device
		cl::Buffer Buffer_Int_Max_Local_non_atomic(context, CL_MEM_READ_WRITE, output_size);

		// allows the buffer object to be written from host memory
		queue.enqueueWriteBuffer(buffer_int, CL_TRUE, 0, input_int_size, &data_int[0], NULL, &Integer_max_non_atomic_Write);

		// setups a kernel and initialises the arguments for parsing
		// parses: buffer used for the data read in, the new buffer for outputting the sum and local memroy 
		cl::Kernel Maximum_Int_Local_non_atomic = cl::Kernel(program, "Max_Kernel_non_atomic");
		Maximum_Int_Local_non_atomic.setArg(0, buffer_int);
		Maximum_Int_Local_non_atomic.setArg(1, Buffer_Int_Max_Local_non_atomic);
		Maximum_Int_Local_non_atomic.setArg(2, cl::Local(local_size * sizeof(int)));

		// enques the kernel to be executes on the device
		queue.enqueueNDRangeKernel(Maximum_Int_Local_non_atomic, cl::NullRange, cl::NDRange(input_int_elements), cl::NDRange(local_size), NULL, &Integer_max_non_atomic);

		// enques the kernel to be executes on the device
		queue.enqueueReadBuffer(Buffer_Int_Max_Local_non_atomic, CL_TRUE, 0, output_size, &Int_Max_Local_non_atomic[0], NULL, &Integer_max_non_atomic_Read);

		// displays the write data time, and memory transfer
		std::cout << "Write execution time [ns]:" <<
			Integer_max_non_atomic_Write.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			Integer_max_non_atomic_Write.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << GetFullProfilingInfo(Integer_max_non_atomic_Write, ProfilingResolution::PROF_US) << endl;
		std::cout << "" << std::endl;
		std::cout << "" << std::endl;

		// displays the Kernel data time, and memory transfer
		std::cout << "Kernel execution time [ns]:" <<
			Integer_max_non_atomic.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			Integer_max_non_atomic.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << GetFullProfilingInfo(Integer_max_non_atomic, ProfilingResolution::PROF_US) << endl;
		std::cout << "" << std::endl;
		std::cout << "" << std::endl;

		// displays the read data time, and memory transfer
		std::cout << "Read execution time [ns]:" <<
			Integer_max_non_atomic_Read.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			Integer_max_non_atomic_Read.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << GetFullProfilingInfo(Integer_max_non_atomic_Read, ProfilingResolution::PROF_US) << endl;
		std::cout << "" << std::endl;
		std::cout << "" << std::endl;

		std::cout << "Size in bytes for vectors: " << std::endl;
		std::cout << "data_int vector: " << input_int_size << std::endl;
		std::cout << "Int_Max_Local_non_atomic vector: " << output_size << std::endl;
		std::cout << "Local memory: " << local_size_bytes_integer << std::endl;
		std::cout << "" << std::endl;
		std::cout << "" << std::endl;

		std::cout << "\nMaximum: " << ((Int_Max_Local_non_atomic[0] / 100)) << std::endl;

		//=====================================================================================
		//=====================================================================================
		//======================================FLOATS=========================================
		//=====================================================================================
		//=====================================================================================
		std::cout << "" << std::endl;
		std::cout << "===============================================" << std::endl;
		std::cout << "===============================================" << std::endl;
		std::cout << "================FLOATS TESTING=================" << std::endl;
		std::cout << "===============================================" << std::endl;
		std::cout << "===============================================" << std::endl;

		// initialisation of the padding size required
		size_t padding_size_float = data_float.size() % local_size;

		// if the input vector is not a multiple of the local_size
		// insert additional neutral elements (0 for addition) so that the total will not be affected
		if (padding_size_float) {
			// create an extra vector with neutral values
			std::vector<float> A_ext(local_size - padding_size_float, 0);
			// append that extra vector to our input
			data_float.insert(data_float.end(), A_ext.begin(), A_ext.end());
		}

		// initialises the the sizes, in bytes and outputs
		size_t input_float_elements = data_float.size();
		size_t input_float_size = data_float.size() * sizeof(float);
		size_t nr_groups_float = input_float_elements / local_size * sizeof(float);

		// initialises the the sizes, in bytes and outputs
		size_t input_float_zero_pad_elements = data_float_Zero_Pad.size();
		size_t input_float_zero_pad_size = data_float_Zero_Pad.size() * sizeof(float);
		size_t nr_groups_zero_pad_float = input_float_zero_pad_elements / local_size * sizeof(float);
		size_t local_size_bytes_float = local_size * sizeof(float);
		output_size = 0;

		// the buffer used for storing the data set for parsing the memory block to kernel
		cl::Buffer buffer_float_zero_pad(context, CL_MEM_READ_ONLY, input_float_zero_pad_size);
		cl::Buffer buffer_float(context, CL_MEM_READ_ONLY, input_float_size);


		//=====================================================================================
		// SUM - MEAN - FLOATS - SINGLE - CALL
		//=====================================================================================
		std::cout << "" << std::endl;
		std::cout << "" << std::endl;
		std::cout << "===============================================" << std::endl;
		std::cout << "===============================================" << std::endl;
		std::cout << "SUM - MEAN - FLOATS - SINGLE - CALL" << std::endl;
		std::cout << "" << std::endl;

		// used for profilling the Write, Kernel and Read queue
		cl::Event Float_mean_Write;
		cl::Event Float_mean;
		cl::Event Float_mean_Read;

		// stores the number vector of 1 size
		std::vector<float> Float_Mean_Local(nr_groups_float);
		// the output size of the above vector in bytes
		output_size = Float_Mean_Local.size() * sizeof(float);
		// local size of the local memory space


		// creates the buffer for the interface for converting data between host and device
		cl::Buffer Buffer_float_Mean_Local(context, CL_MEM_READ_WRITE, output_size);

		// allows the buffer object to be written from host memory
		queue.enqueueWriteBuffer(buffer_float, CL_TRUE, 0, input_float_size, &data_float[0], NULL, &Float_mean_Write);

		// setups a kernel and initialises the arguments for parsing
		// parses: buffer used for the data read in, the new buffer for outputting the sum and local memroy 
		cl::Kernel Reduction_Float_Local = cl::Kernel(program, "float_Reduction_Kernel");
		Reduction_Float_Local.setArg(0, buffer_float);
		Reduction_Float_Local.setArg(1, Buffer_float_Mean_Local);
		Reduction_Float_Local.setArg(2, cl::Local(local_size_bytes_float));

		// enques the kernel to be executes on the devices
		queue.enqueueNDRangeKernel(Reduction_Float_Local, cl::NullRange, cl::NDRange(input_float_elements), cl::NDRange(local_size), NULL, &Float_mean);

		// enques the kernel to be executes on the device
		queue.enqueueReadBuffer(Buffer_float_Mean_Local, CL_TRUE, 0, output_size, &Float_Mean_Local[0], NULL, &Float_mean_Read);

		// displays the write data time, and memory transfer
		std::cout << "Write execution time [ns]:" <<
			Float_mean_Write.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			Float_mean_Write.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << GetFullProfilingInfo(Float_mean_Write, ProfilingResolution::PROF_US) << endl;
		std::cout << "" << std::endl;
		std::cout << "" << std::endl;

		// displays the Kernel data time, and memory transfer
		std::cout << "Kernel execution time [ns]:" <<
			Float_mean.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			Float_mean.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << GetFullProfilingInfo(Float_mean, ProfilingResolution::PROF_US) << endl;
		std::cout << "" << std::endl;
		std::cout << "" << std::endl;

		// displays the read data time, and memory transfer
		std::cout << "Read execution time [ns]:" <<
			Float_mean_Read.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			Float_mean_Read.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << GetFullProfilingInfo(Float_mean_Read, ProfilingResolution::PROF_US) << endl;
		std::cout << "" << std::endl;
		std::cout << "" << std::endl;

		std::cout << "Size in bytes for vectors: " << std::endl;
		std::cout << "data_float vector: " << input_float_size << std::endl;
		std::cout << "Float_Mean_Local vector: " << output_size << std::endl;
		std::cout << "Local memory: " << local_size_bytes_float << std::endl;
		std::cout << "" << std::endl;
		std::cout << "" << std::endl;

		std::cout << "\nSum: " << ((Float_Mean_Local[0])) << std::endl;
		std::cout << "\nMean: " << ((Float_Mean_Local[0]) / dat_float_original_size) << std::endl;

		//=====================================================================================
		// SUM - MEAN - FLOATS - MULTI - CALL
		//=====================================================================================
		std::cout << "" << std::endl;
		std::cout << "" << std::endl;
		std::cout << "===============================================" << std::endl;
		std::cout << "===============================================" << std::endl;
		std::cout << "SUM - MEAN - FLOATS - MULTI - CALL" << std::endl;
		std::cout << "" << std::endl;

		// used for profilling the Write, Kernel and Read queue
		cl::Event Float_mean_multi_Write;
		cl::Event Float_mean_multi;
		cl::Event Float_mean_multi_Read;

		// stores the number vector of 1 size
		std::vector<float> Float_Mean_Local_Multi(nr_groups_float);
		// the output size of the above vector in bytes
		output_size = Float_Mean_Local_Multi.size() * sizeof(float);
		// local size of the local memory space


		// creates the buffer for the interface for converting data between host and device
		cl::Buffer Buffer_float_Mean_Local_Multi(context, CL_MEM_READ_WRITE, output_size);

		// allows the buffer object to be written from host memory
		queue.enqueueWriteBuffer(buffer_float, CL_TRUE, 0, input_float_size, &data_float[0], NULL, &Float_mean_multi_Write);

		// setups a kernel and initialises the arguments for parsing
		// parses: buffer used for the data read in, the new buffer for outputting the sum and local memroy 
		cl::Kernel Reduction_Float_Local_Multi = cl::Kernel(program, "float_Reduction_Kernel_mulitcall");
		Reduction_Float_Local_Multi.setArg(0, buffer_float);
		Reduction_Float_Local_Multi.setArg(1, Buffer_float_Mean_Local_Multi);
		Reduction_Float_Local_Multi.setArg(2, cl::Local(local_size_bytes_float));

		// enques the kernel to be executes on the devices
		queue.enqueueNDRangeKernel(Reduction_Float_Local_Multi, cl::NullRange, cl::NDRange(input_float_elements), cl::NDRange(local_size), NULL, &Float_mean_multi);
		// waits for the event to be ready for storing 
		Float_mean_multi.wait();

		// stores the nano seconds for each call the kernel for profiling 
		int nano_secs = 0;
		// counts the amount of nano seconds needs to complete for each call to the kernel
		nano_secs += Float_mean_multi.getProfilingInfo<CL_PROFILING_COMMAND_END>() - Float_mean_multi.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		// loops over the remaining workgroups to add those up 
		for (int i = 0; i < 2; i++)
		{
			// passes in the buffer output from the above call and assigns the ouput as the input 
			Reduction_Float_Local_Multi.setArg(0, Buffer_float_Mean_Local_Multi);
			Reduction_Float_Local_Multi.setArg(1, Buffer_float_Mean_Local_Multi);
			// enques the kernel to be executes on the devices
			queue.enqueueNDRangeKernel(Reduction_Float_Local_Multi, cl::NullRange, cl::NDRange(input_float_elements), cl::NDRange(local_size), NULL, &Float_mean_multi);
			// waits for the event to be ready for storing 
			Float_mean_multi.wait();
			// counts the amount of nano seconds needs to complete for each call to the kernel
			nano_secs += Float_mean_multi.getProfilingInfo<CL_PROFILING_COMMAND_END>() - Float_mean_multi.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		}

		// enques the kernel to be executes on the device
		queue.enqueueReadBuffer(Buffer_float_Mean_Local_Multi, CL_TRUE, 0, output_size, &Float_Mean_Local_Multi[0], NULL, &Float_mean_multi_Read);

		// displays the write data time, and memory transfer
		std::cout << "Write execution time [ns]:" <<
			Float_mean_multi_Write.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			Float_mean_multi_Write.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << GetFullProfilingInfo(Float_mean_multi_Write, ProfilingResolution::PROF_US) << endl;
		std::cout << "" << std::endl;
		std::cout << "" << std::endl;

		// displays the Kernel data time, and memory transfer
		std::cout << "Kernel execution time [ns]:" << nano_secs << std::endl;
		std::cout << GetFullProfilingInfo(Float_mean_multi, ProfilingResolution::PROF_US) << endl;
		std::cout << "" << std::endl;
		std::cout << "" << std::endl;

		// displays the read data time, and memory transfer
		std::cout << "Read execution time [ns]:" <<
			Float_mean_multi_Read.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			Float_mean_multi_Read.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << GetFullProfilingInfo(Float_mean_multi_Read, ProfilingResolution::PROF_US) << endl;
		std::cout << "" << std::endl;
		std::cout << "" << std::endl;

		std::cout << "Size in bytes for vectors: " << std::endl;
		std::cout << "data_float vector: " << input_float_size << std::endl;
		std::cout << "Float_Mean_Local vector: " << output_size << std::endl;
		std::cout << "Local memory: " << local_size_bytes_float << std::endl;
		std::cout << "" << std::endl;
		std::cout << "" << std::endl;

		std::cout << "\nSum: " << ((Float_Mean_Local_Multi[0])) << std::endl;
		std::cout << "\nMean: " << ((Float_Mean_Local_Multi[0]) / dat_float_original_size) << std::endl;


		//=====================================================================================
		// MIN - FLOATS
		//=====================================================================================
		std::cout << "" << std::endl;
		std::cout << "" << std::endl;
		std::cout << "===============================================" << std::endl;
		std::cout << "===============================================" << std::endl;
		std::cout << "MIN - FLOATS"  << std::endl;
		std::cout << "" << std::endl;

		// used for profilling the Write, Kernel and Read queues
		cl::Event Floats_min_Write;
		cl::Event Floats_min;
		cl::Event Floats_min_Read;

		// stores the number vector of nr_groups_float size
		std::vector<float> Float_Min_Local(nr_groups_float);
		// the output size of the above vector in bytes
		output_size = Float_Min_Local.size() * sizeof(float);

		// creates the buffer for the interface for converting data between host and device
		cl::Buffer Buffer_float_Min_Local(context, CL_MEM_READ_WRITE, output_size);

		// allows the buffer object to be written from host memory
		queue.enqueueWriteBuffer(buffer_float, CL_TRUE, 0, input_float_size, &data_float[0], NULL, &Floats_min_Write);
		
		// setups a kernel and initialises the arguments for parsing
		// parses: buffer used for the data read in, the new buffer for outputting the sum and local memroy 
		cl::Kernel Minimum_Float_Local = cl::Kernel(program, "Float_Reduction_Min");
		Minimum_Float_Local.setArg(0, buffer_float);
		Minimum_Float_Local.setArg(1, Buffer_float_Min_Local);
		Minimum_Float_Local.setArg(2, cl::Local(local_size * sizeof(float)));

		// enques the kernel to be executes on the device
		queue.enqueueNDRangeKernel(Minimum_Float_Local, cl::NullRange, cl::NDRange(input_float_elements), cl::NDRange(local_size), NULL, &Floats_min);

		// enques the kernel to be executes on the device
		queue.enqueueReadBuffer(Buffer_float_Min_Local, CL_TRUE, 0, output_size, &Float_Min_Local[0], NULL, &Floats_min_Read);

		// displays the write data time, and memory transfer
		std::cout << "Write execution time [ns]:" <<
			Floats_min_Write.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			Floats_min_Write.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << GetFullProfilingInfo(Floats_min_Write, ProfilingResolution::PROF_US) << endl;
		std::cout << "" << std::endl;
		std::cout << "" << std::endl;

		// displays the Kernel data time, and memory transfer
		std::cout << "Kernel execution time [ns]:" <<
			Floats_min.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			Floats_min.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << GetFullProfilingInfo(Floats_min, ProfilingResolution::PROF_US) << endl;
		std::cout << "" << std::endl;
		std::cout << "" << std::endl;

		// displays the read data time, and memory transfer
		std::cout << "Read execution time [ns]:" <<
			Floats_min_Read.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			Floats_min_Read.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << GetFullProfilingInfo(Floats_min_Read, ProfilingResolution::PROF_US) << endl;
		std::cout << "" << std::endl;
		std::cout << "" << std::endl;

		std::cout << "Size in bytes for vectors: " << std::endl;
		std::cout << "data_float vector: " << input_float_size << std::endl;
		std::cout << "Float_Min_Local vector: " << output_size << std::endl;
		std::cout << "Local memory: " << local_size_bytes_float << std::endl;
		std::cout << "" << std::endl;
		std::cout << "" << std::endl;

		std::cout << "\nMin: " << ((Float_Min_Local[0])) << std::endl;


		//=====================================================================================
		// MAX - FLOATS
		//=====================================================================================
		std::cout << "" << std::endl;
		std::cout << "" << std::endl;
		std::cout << "===============================================" << std::endl;
		std::cout << "===============================================" << std::endl;
		std::cout << "MAX - FLOATS" << std::endl;
		std::cout << "" << std::endl;

		// used for profilling the Write, Kernel and Read queue
		cl::Event Floats_max_Write;
		cl::Event Floats_max;
		cl::Event Floats_max_Read;

		// stores the number vector of 1 size
		std::vector<float> Float_Max_Local(nr_groups_float);
		// the output size of the above vector in bytes
		output_size = Float_Max_Local.size() * sizeof(float);

		// creates the buffer for the interface for converting data between host and device
		cl::Buffer Buffer_float_Max_Local(context, CL_MEM_READ_WRITE, output_size);

		// allows the buffer object to be written from host memory
		queue.enqueueWriteBuffer(buffer_float, CL_TRUE, 0, input_float_size, &data_float[0], NULL, &Floats_max_Write);

		// setups a kernel and initialises the arguments for parsing
		// parses: buffer used for the data read in, the new buffer for outputting the sum and local memroy 
		cl::Kernel Maximum_Float_Local = cl::Kernel(program, "Float_Reduction_Max");
		Maximum_Float_Local.setArg(0, buffer_float);
		Maximum_Float_Local.setArg(1, Buffer_float_Max_Local);
		Maximum_Float_Local.setArg(2, cl::Local(local_size * sizeof(float)));
		
		// enques the kernel to be executes on the device
		queue.enqueueNDRangeKernel(Maximum_Float_Local, cl::NullRange, cl::NDRange(input_float_elements), cl::NDRange(local_size), NULL, &Floats_max);

		// enques the kernel to be executes on the device
		queue.enqueueReadBuffer(Buffer_float_Max_Local, CL_TRUE, 0, output_size, &Float_Max_Local[0], NULL, &Floats_max_Read);

		// displays the write data time, and memory transfer
		std::cout << "Write execution time [ns]:" <<
			Floats_max_Write.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			Floats_max_Write.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << GetFullProfilingInfo(Floats_max_Write, ProfilingResolution::PROF_US) << endl;
		std::cout << "" << std::endl;
		std::cout << "" << std::endl;

		// displays the Kernel data time, and memory transfer
		std::cout << "Kernel execution time [ns]:" <<
			Floats_max.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			Floats_max.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << GetFullProfilingInfo(Floats_max, ProfilingResolution::PROF_US) << endl;
		std::cout << "" << std::endl;
		std::cout << "" << std::endl;
		
		// displays the read data time, and memory transfer
		std::cout << "Read execution time [ns]:" <<
			Floats_max_Read.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			Floats_max_Read.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << GetFullProfilingInfo(Floats_max_Read, ProfilingResolution::PROF_US) << endl;
		std::cout << "" << std::endl;
		std::cout << "" << std::endl;

		std::cout << "Size in bytes for vectors: " << std::endl;
		std::cout << "data_float vector: " << input_float_size << std::endl;
		std::cout << "Float_Max_Local vector: " << output_size << std::endl;
		std::cout << "Local memory: " << local_size_bytes_float << std::endl;
		std::cout << "" << std::endl;
		std::cout << "" << std::endl;

		std::cout << "\nMax: " << ((Float_Max_Local[0])) << std::endl;


		//=====================================================================================
		// DEVIATION - FLOATS
		//=====================================================================================
		std::cout << "" << std::endl;
		std::cout << "" << std::endl;
		std::cout << "===============================================" << std::endl;
		std::cout << "===============================================" << std::endl;
		std::cout << "DEVIATION - FLOATS" << std::endl;
		std::cout << "" << std::endl;

		// used for profilling the Write, Kernel and Read queue
		cl::Event Deviation_float_Write;
		cl::Event Deviation_float;
		cl::Event Deviation_float_Read;

		// stores the number vector of 1 size
		std::vector<float> Float_Dev_Local(nr_groups_float);
		// the output size of the above vector in bytes
		output_size = Float_Dev_Local.size() * sizeof(float);
		// float_averger stores the mean to be passed into std kernel
		float Float_average = ((Float_Mean_Local[0]) / input_float_elements);

		// creates the buffer for the interface for converting data between host and device
		cl::Buffer Buffer_float_Dev_Local(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_average(context, CL_MEM_READ_ONLY, sizeof(float));

		// allows the buffer object to be written from host memory
		queue.enqueueWriteBuffer(buffer_float, CL_TRUE, 0, input_float_size, &data_float[0], NULL, &Deviation_float_Write);

		// setups a kernel and initialises the arguments for parsing
		// parses: buffer used for the data read in, the new buffer for outputting the sum and local memroy 
		cl::Kernel Deviation_Float_Local = cl::Kernel(program, "float_Reduction_Dev_Kernel");
		Deviation_Float_Local.setArg(0, buffer_float);
		Deviation_Float_Local.setArg(1, Buffer_float_Dev_Local);
		Deviation_Float_Local.setArg(2, cl::Local(local_size * sizeof(float)));//local memory size
		Deviation_Float_Local.setArg(3, Float_average);

		// enques the kernel to be executes on the device
		queue.enqueueNDRangeKernel(Deviation_Float_Local, cl::NullRange, cl::NDRange(input_float_elements), cl::NDRange(local_size), NULL, &Deviation_float);

		// enques the kernel to be executes on the device
		queue.enqueueReadBuffer(Buffer_float_Dev_Local, CL_TRUE, 0, output_size, &Float_Dev_Local[0], NULL, &Deviation_float_Read);

		// displays the write data time, and memory transfer
		std::cout << "Write execution time [ns]:" <<
			Deviation_float_Write.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			Deviation_float_Write.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << GetFullProfilingInfo(Deviation_float_Write, ProfilingResolution::PROF_US) << endl;
		std::cout << "" << std::endl;
		std::cout << "" << std::endl;

		// displays the Kernel data time, and memory transfer
		std::cout << "Kernel execution time [ns]:" <<
			Deviation_float.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			Deviation_float.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << GetFullProfilingInfo(Deviation_float, ProfilingResolution::PROF_US) << endl;
		std::cout << "" << std::endl;
		std::cout << "" << std::endl;

		// displays the read data time, and memory transfer
		std::cout << "Read execution time [ns]:" <<
			Deviation_float_Read.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			Deviation_float_Read.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << GetFullProfilingInfo(Deviation_float_Read, ProfilingResolution::PROF_US) << endl;
		std::cout << "" << std::endl;
		std::cout << "" << std::endl;

		std::cout << "Size in bytes for vectors: " << std::endl;
		std::cout << "data_float vector: " << input_float_size << std::endl;
		std::cout << "Float_Dev_Local vector: " << output_size << std::endl;
		std::cout << "Local memory: " << local_size_bytes_float << std::endl;
		std::cout << "" << std::endl;
		std::cout << "" << std::endl;

		std::cout << "\nSTD: " << (sqrt(Float_Dev_Local[0] / dat_float_original_size)) << std::endl;


		//=====================================================================================
		// SORTING - FLOATS
		//=====================================================================================
		std::cout << "" << std::endl;
		std::cout << "" << std::endl;
		std::cout << "===============================================" << std::endl;
		std::cout << "===============================================" << std::endl;
		std::cout << "SORTING - FLOATS" << std::endl;
		std::cout << "" << std::endl;

		// used for profilling the Write, Kernel and Read queue
		cl::Event sort_float_event_Write;
		cl::Event sort_float_event;
		cl::Event sort_float_event_Read;

		// stores the number vector of data_float.size
		std::vector<float> Float_Sort_Global(data_float.size());
		// the output size of the above vector in bytes
		output_size = data_float.size() * sizeof(float);

		// allows the buffer object to be written from host memory
		queue.enqueueWriteBuffer(buffer_float, CL_TRUE, 0, input_float_size, &data_float[0], NULL, &sort_float_event_Write);

		// setups a kernel and initialises the arguments for parsing
		// parses: buffer used for the data read in, the new buffer for outputting the sum and local memroy 
		cl::Kernel odd_sort = cl::Kernel(program, "bubble_odd_sort_kernel");
		odd_sort.setArg(0, buffer_float);
		cl::Kernel even_sort = cl::Kernel(program, "bubble_even_sort_kernel");
		even_sort.setArg(0, buffer_float);

		// starts timer for below kernel calls to N/2
		auto start = std::chrono::high_resolution_clock::now();
		nano_secs = 0;
		// iterates through size/2 times worse case for odd even sort
		for (int i = 0; i < data_float.size() / 2; i++)
		{
			// enques the kernel to be executes on the device
			queue.enqueueNDRangeKernel(odd_sort, cl::NullRange, cl::NDRange(input_float_elements), cl::NDRange(local_size), NULL, &sort_float_event);
			sort_float_event.wait();
			nano_secs += sort_float_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - sort_float_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
			// enques the kernel to be executes on the device
			queue.enqueueNDRangeKernel(even_sort, cl::NullRange, cl::NDRange(input_float_elements), cl::NDRange(local_size), NULL, &sort_float_event);
			sort_float_event.wait();
			nano_secs += sort_float_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - sort_float_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		}
		// enques the kernel to be executes on the device
		queue.enqueueReadBuffer(buffer_float, CL_TRUE, 0, output_size, &Float_Sort_Global[0], NULL, &sort_float_event_Read);

		// ends timer for above kernel calls
		auto finish = std::chrono::high_resolution_clock::now();
		chrono::duration<double> elapsed = finish - start;

		int offset = (Float_Sort_Global.size()) - dat_float_original_size;
		int middle = ((Float_Sort_Global.size() / 2));
		std::cout << "FINISHED SORTING" << std::endl;
		std::cout << "Min Through Sorting: " << Float_Sort_Global[0] << std::endl;
		std::cout << "Max Through Sorting: " << Float_Sort_Global[Float_Sort_Global.size() - 1] << std::endl;
		std::cout << "Offset: " << offset << std::endl;
		std::cout << "Original Size before pad: " << dat_float_original_size << std::endl;
		std::cout << "New Size after pad: " << data_float.size() << std::endl;
		std::cout << "New Size after sort: " << Float_Sort_Global.size() << std::endl;
		std::cout << "Midddle + Offset: " << middle + padding_size_float << std::endl;
		std::cout << "Middle: " << middle << std::endl;
		std::cout << "Median Through Sorting: " << Float_Sort_Global[(Float_Sort_Global.size()/2) + padding_size_float] << std::endl;
		std::cout << "25% Quartile Through Sorting: " << Float_Sort_Global[(Float_Sort_Global.size()*0.25) + padding_size_float] << std::endl;
		std::cout << "75% Quartile Through Sorting: " << Float_Sort_Global[(Float_Sort_Global.size()*0.75) + padding_size_float] << std::endl;
		std::cout << "Kernel execution time [ns]:" << elapsed.count() << std::endl;
		std::cout << "\nSTD: " << (sqrt(Float_Dev_Local[0] / input_float_elements)) << std::endl;
		std::cout << "" << std::endl;

		// displays the write data time, and memory transfer
		std::cout << "Write execution time [ns]:" <<
			sort_float_event_Write.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			sort_float_event_Write.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << GetFullProfilingInfo(sort_float_event_Write, ProfilingResolution::PROF_US) << endl;
		std::cout << "" << std::endl;
		std::cout << "" << std::endl;

		// displays the Kernel data time, and memory transfer
		std::cout << "Kernel execution time [ns]:" << nano_secs << std::endl;
		std::cout << GetFullProfilingInfo(sort_float_event, ProfilingResolution::PROF_US) << endl;
		std::cout << "" << std::endl;
		std::cout << "" << std::endl;

		// displays the read data time, and memory transfer
		std::cout << "Read execution time [ns]:" <<
			sort_float_event_Read.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			sort_float_event_Read.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << GetFullProfilingInfo(sort_float_event_Read, ProfilingResolution::PROF_US) << endl;
		std::cout << "" << std::endl;
		std::cout << "" << std::endl;

		std::cout << "Size in bytes for vectors: " << std::endl;
		std::cout << "data_float vector: " << input_float_size << std::endl;
		std::cout << "Float_Sort_Global vector: " << output_size << std::endl;
		std::cout << "Local memory: " << local_size_bytes_float << std::endl;
		std::cout << "" << std::endl;
		std::cout << "" << std::endl;
		// check that vector is sorted
		for (int i = 0; i < Float_Sort_Global.size()-1; i++)
		{
			if (Float_Sort_Global[i] > Float_Sort_Global[i + 1])
			{
				std::cout << "FOUND: " << i << std::endl;
				std::cout << Float_Sort_Global[i] << std::endl;
				std::cout << Float_Sort_Global[i + 1] << std::endl;
				system("PAUSE");
			}
		}
	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	system("PAUSE");
	return 0;
}


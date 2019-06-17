#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

int main(int argc, char** argv) {

	if(argc < 1){
      printf("USAGE: ./pi_process NUM_PONTOS");
    }

    int num_points = atof(argv[1]);
	
	int  num_process;
	double pi;
	double sum=0.0;
	double sum_partial=0.0;
	double step = 1.0/num_points;
	int rank;

	MPI_Init (&argc, &argv);

	//Get process ID
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);

    //Get number of process
    MPI_Comm_size (MPI_COMM_WORLD, &num_process);

    printf("Rank %d, Process: %d\n",rank, num_process);
	for (int i=rank; i < num_points; i+=num_process) {
		double x = (i+0.5)*step;
		sum_partial += 4.0/(1.0+x*x);
	}

	//Reduce to Sum all partial calculations
    MPI_Reduce(&sum_partial,
    	&sum, //sendbuf - address of send buffer (choice)
    	1, //count - number of elements in send buffer (integer) 
    	MPI_DOUBLE, //datatype - data type of elements of send buffer (handle)
    	MPI_SUM, //op - reduce operation (handle) 
    	0, //root - rank of root process (integer)
    	MPI_COMM_WORLD //comm - communicator (handle)
    	);

    //Caculate and print PI
    if (rank==0)
    {
        pi=(sum * step);
    	printf("PI = %f\n",(pi));
    }

	MPI_Finalize();
    
    return 0;


}

	
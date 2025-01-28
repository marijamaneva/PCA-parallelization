#include <iomanip>
#include <iostream>       // input/output stream operations
#include <chrono>         // measurement of execution time
#include <vector>         // usage of vector container
#include <fstream>        // file handling
#include <Eigen/Dense>    // matrix & linear algebra operations --> eigen library
#include <mpi.h>          // mpi library

using namespace std;           // standard namespace
using namespace Eigen;         // eigen namespace 
using namespace std::chrono;   


// struct to store timing data and parameters
struct PerformanceInfo {
    int numProcesses;
    int numRows;
    int numCols;
    int numPC;
    long long timeMean;
    long long timeStdDev;
    long long timeNormalization;
    long long timeCovMatrix;
    long long timeEigDecomp;
    long long timeProjection;
    long long timeTotal;
};

// function to save timing details into a file
void writePerformanceInfo(const PerformanceInfo& info, const string& filename) {
    // open the file in append mode (no overwriting)
    ofstream outFile(filename, ios::app); 
    if (!outFile.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return;
    }
    outFile << info.numProcesses << ", " << info.numRows << ", " << info.numCols << ", "
        << info.timeMean << ", " << info.timeStdDev << ", " << info.timeNormalization << ", "
        << info.timeCovMatrix << ", " << info.timeEigDecomp << ", " << info.timeProjection << ", "
        << info.timeTotal << endl;
    outFile.close();
}

// function to compute the mean of each column
vector<double> computeMean(const MatrixXd& data) {
    int numRows = data.rows();
    int numCols = data.cols();
    vector<double> mean(numCols, 0.0);

    for (int col = 0; col < numCols; ++col) {
        double sum = 0.0;
        for (int row = 0; row < numRows; ++row) {
            sum += data(row, col);
        }
        mean[col] = sum / numRows;
    }

    return mean;
}

// function to compute the standard deviation for each column
vector<double> computeStdDev(const MatrixXd& data, const vector<double>& mean) {
    int numRows = data.rows();
    int numCols = data.cols();
    vector<double> StdDev(numCols);

    for (int col = 0; col < numCols; ++col) {
        double squaredSum = 0.0;
        for (int row = 0; row < numRows; ++row) {
            double diff = data(row, col) - mean[col];
            squaredSum += diff * diff;
        }
        StdDev[col] = sqrt(squaredSum / (numRows - 1));
    }

    return StdDev;
}

// function to standardize the data using z-scores
void normalizeMatrix(MatrixXd& data, const vector<double>& mean, const vector<double>& StdDev) {
    int numRows = data.rows();
    int numCols = data.cols();

    for (int row = 0; row < numRows; ++row) {
        for (int col = 0; col < numCols; ++col) {
            data(row, col) = (data(row, col) - mean[col]) / StdDev[col];
        }
    }
}

// function to compute the covariance matrix
MatrixXd computeCovarianceMatrix(const MatrixXd& locData, const MatrixXd& matrixT, int numRows) {
    // return (matrixT * locData) / (numRows - 1);
    int loCols = locData.cols();
    int RowsT = matrixT.rows();
    int ColsT = matrixT.cols();
    MatrixXd result(RowsT, loCols);

    for (int i = 0; i < RowsT; ++i) {
        for (int j = 0; j < loCols; ++j) {
            double sum = 0.0;
            for (int k = 0; k < ColsT; ++k) {
                sum += matrixT(i, k) * locData(k, j); // dot product between i and j
            }
            result(i, j) = sum / (numRows - 1);
        }
    }

    return result;
}

// function to project data onto selected components (matrix multiplication)
MatrixXd MatrixMult(const MatrixXd& locData, const MatrixXd& PrinComp, int numRows) {
    int loCols = locData.cols();
    int PrinCompRows = PrinComp.rows();
    int PrinCompCols = PrinComp.cols();
    MatrixXd result(PrinCompRows, loCols); 

    for (int i = 0; i < PrinCompRows; ++i) {
        for (int j = 0; j < loCols; ++j) {
            double sum = 0.0;
            for (int k = 0; k < PrinCompCols; ++k) {
                sum += PrinComp(i, k) * locData(k, j); 
            }
            result(i, j) = sum;
        }
    }

    return result;
}


int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); //get rank
    MPI_Comm_size(MPI_COMM_WORLD, &size); //get number of processors

    string filename;
    int k = 3; // Number Principal Components

    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <dataset_file_path>" << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    filename = argv[1]; // put dataset file path in command line arguments


    MatrixXd data;
    auto startTotal = high_resolution_clock::now(); // start timer
    if (rank == 0) {
        // open dataset file
        ifstream inputFile(filename);
        if (!inputFile.is_open()) {
            cerr << "Error opening file: " << filename << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        string line;
        if (getline(inputFile, line)) {
            stringstream ss(line);
            vector<double> tempValues;
            double value;
            while (ss >> value) {
                tempValues.push_back(value);
            }
            int numCols = tempValues.size();
            int numRows = 1;

            while (getline(inputFile, line)) {
                numRows++;
            }
            inputFile.clear();
            inputFile.seekg(0, ios::beg);

            data.resize(numRows, numCols);
            for (int i = 0; i < numRows; ++i) {
                for (int j = 0; j < numCols; ++j) {
                    inputFile >> data(i, j);
                }
            }
        }
        else {
            cerr << "Error reading file: " << filename << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // distribution of the matrix dimensions to all processors
    int numRows, numCols;
    if (rank == 0) {
        numRows = data.rows();
        numCols = data.cols();
    }
    MPI_Bcast(&numRows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&numCols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // distribution columns to all processors
    int colsPerProcessor = numCols / size;
    int moreCols = numCols % size;
    int sendCols = colsPerProcessor + (rank < moreCols ? 1 : 0);
    int startCol = rank * colsPerProcessor + min(rank, moreCols);
    int endCol = startCol + sendCols - 1;

    // debugging columns values
    std::cout << "Rank " << rank << ": startCol = " << startCol << ", endCol = " << endCol << ", sendCols = " << sendCols << std::endl;

    MatrixXd locData(numRows, sendCols);
    MPI_Status status;

    if (rank == 0) {
        MPI_Datatype columnType;
        MPI_Type_vector(numRows, sendCols, numCols, MPI_DOUBLE, &columnType);
        MPI_Type_commit(&columnType);

        for (int i = 1; i < size; ++i) {
            int loCols = colsPerProcessor + (i < moreCols ? 1 : 0);
            int sendStartCol = i * colsPerProcessor + std::min(i, moreCols); 
            MPI_Send(data.data() + sendStartCol * numRows, loCols * numRows, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);

        }
        // local data for rank 0
        locData = data.block(0, startCol, numRows, sendCols); 
        MPI_Type_free(&columnType);
    }
    else {
        MPI_Recv(locData.data(), numRows * sendCols, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
    }

     // computation mean recall function
    auto startMean = high_resolution_clock::now();
    vector<double> mean = computeMean(locData);
    auto endMean = high_resolution_clock::now();
    if (rank == 0) {
        cout << "Mean calculation time: " << duration_cast<microseconds>(endMean - startMean).count() << " microseconds" << endl;
    }

    // computation standard deviation recall function
    auto startStdDev = high_resolution_clock::now();
    vector<double> StdDev = computeStdDev(locData, mean);  
    auto endStdDev = high_resolution_clock::now();
    if (rank == 0) {
        cout << "Standard deviation calculation time: " << duration_cast<microseconds>(endStdDev - startStdDev).count() << " microseconds" << endl;
    }
   
   // normalization data recall function
    auto startNormalization = high_resolution_clock::now();
    normalizeMatrix(locData, mean, StdDev);
    auto endNormalization = high_resolution_clock::now();
    if (rank == 0) {
        cout << "Normalization time: " << duration_cast<microseconds>(endNormalization - startNormalization).count() << " microseconds" << endl;
    }
    std::cout << "Rank " << rank << "standardizedData= " << locData << std::endl;
    
    // put together z-score normalized data from all processesors to the main processor
    MatrixXd GatheredData;
    if (rank == 0) {
        GatheredData.resize(numRows, numCols);
    }

    int send_count = sendCols * numRows;

    // counts for each processor
    std::vector<int> recv_counts(size);
    MPI_Gather(&send_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    // displacement for each processor 
    std::vector<int> displacements(size);
    if (rank == 0) {
        displacements[0] = 0;
        for (int i = 1; i < size; ++i) {
            displacements[i] = displacements[i - 1] + recv_counts[i - 1];
        }
    }

    // put together data from all processesors to the main processor
    MPI_Gatherv(locData.data(), send_count, MPI_DOUBLE, GatheredData.data(), recv_counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        cout << "Normalized Data:" << endl;
        cout << GatheredData << endl;
    }

    MatrixXd matrixT(numCols, numRows);

    // transposition of the data in the main processors
    if (rank == 0) {
        matrixT = GatheredData.transpose();
    }
    if (rank == 0) {
        cout << "Transposed Normalized Data:" << endl;
        cout << matrixT << endl;
    }

    // distribution of the transposed matrix dimensions to all processesors
    MPI_Bcast(&numRows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&numCols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // distribution the transposed matrix data to all processesors
    MPI_Bcast(matrixT.data(), numCols * numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // std::cout << "Rank " << rank << "TransposedData= " << matrixT << std::endl;

    // computation covariance matrix recall function
    auto startCovMatrix = high_resolution_clock::now();
    MatrixXd PartialCovMatrix = computeCovarianceMatrix(locData, matrixT, numRows);
    // std::cout << "Rank " << rank << "PartialCovMatrix= " << PartialCovMatrix << std::endl;

    // gather partial covariance matrices to the main processor
    MatrixXd gatheredPartialCovarianceMatrix;
    if (rank == 0) {
        gatheredPartialCovarianceMatrix.resize(numCols, numCols);
    }
 
    // send counts for each processor
    std::vector<int> send_counts(size);
    send_counts[0] = sendCols * numCols;
    for (int i = 1; i < size; ++i) {
        int loCols = colsPerProcessor + (i < moreCols ? 1 : 0);
        send_counts[i] = loCols * numCols;
    }

    // displacement for each processor 
    if (rank == 0) {
        displacements[0] = 0;
        for (int i = 1; i < size; ++i) {
            displacements[i] = displacements[i - 1] + send_counts[i - 1];
        }
    }

    // put together data from all processors to the main processor
    MPI_Gatherv(PartialCovMatrix.data(), sendCols * numCols, MPI_DOUBLE,
        gatheredPartialCovarianceMatrix.data(), send_counts.data(), displacements.data(), MPI_DOUBLE,
        0, MPI_COMM_WORLD);

    if (rank == 0) {
        cout << "Gathered Partial Covariance Matrix:" << endl;
        cout << gatheredPartialCovarianceMatrix << endl;
    }

    auto endCovMatrix = high_resolution_clock::now();
    if (rank == 0) {
        cout << "Time taken for calculating covariance matrix: " << duration_cast<microseconds>(endCovMatrix - startCovMatrix).count() << " microseconds" << endl;
    }

    MatrixXd localPrincipalComponents(sendCols,k);
    MatrixXd SelectedPrinComp;
    auto startEigDecomp = high_resolution_clock::now();
    if (rank == 0) {
        // eigendecomposition of the covariance matrix with eigensolver
        SelfAdjointEigenSolver<MatrixXd> eigensolver(gatheredPartialCovarianceMatrix);
        if (eigensolver.info() != Success) {
            cerr << "Failed to compute eigendecomposition!" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // eigenvalues and eigenvectors
        VectorXd eigenvalues = eigensolver.eigenvalues();
        MatrixXd eigenvectors = eigensolver.eigenvectors();
 
        SelectedPrinComp = eigenvectors.rightCols(k);
        cout << SelectedPrinComp << endl;
        cout << "Memory address of SelectedPrinComp: " << endl;
        for (int i = 0; i < SelectedPrinComp.rows(); ++i) {
            for (int j = 0; j < SelectedPrinComp.cols(); ++j) {
                cout << "Value at (" << i << ", " << j << "): " << &SelectedPrinComp(i, j) << endl;
            }
        }
    }
    auto endEigDecomp = high_resolution_clock::now();
    if (rank == 0) {
        cout << "Eigendecomposition time: " << duration_cast<microseconds>(endEigDecomp - startEigDecomp).count() << " microseconds" << endl;
    }

    auto startProjection = high_resolution_clock::now();
    int num_rows, num_cols;
    if (rank == 0) {
        num_rows = SelectedPrinComp.rows();
        num_cols = SelectedPrinComp.cols();
    }

    // distribution of the number of rows and columns to all processors
    MPI_Bcast(&num_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    int chunk_size = num_rows / size;
    int remainder = num_rows % size;

    // computation start and end rows for the current process
    int start_row = rank * chunk_size + std::min(rank, remainder);
    int end_row = start_row + chunk_size + (rank < remainder ? 1 : 0);
    end_row = std::min(end_row, num_rows);
    
    // initilize memory for local chunk
    MatrixXd locChunk(end_row - start_row, num_cols);
    int current_offset = 0;
    for (int i = 0; i < size; ++i) {
        // chunk size for this process
        int rows_for_process = chunk_size + (i < remainder ? 1 : 0);
        send_counts[i] = rows_for_process; // Send the number of rows

        // displacement for this processor
        displacements[i] = current_offset;
        current_offset += rows_for_process;
    }

    // allocation memory for receiving data in each processor
    int recv_count = send_counts[rank]; 

    // execution send and receive operations for each column
    for (int col = 0; col < num_cols; ++col) {
        // compute the memory offset for the current column 
        double* send_ptr = SelectedPrinComp.data() + col * num_rows; //col * num_rows to start from each column
        double* recv_ptr = locChunk.data() + col * (end_row - start_row);
        
        // send and receive operation
        MPI_Scatterv(send_ptr, send_counts.data(),displacements.data(), MPI_DOUBLE, recv_ptr,
            recv_count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    cout << "Rank " << rank << " received data:" << endl;
    cout << locChunk << endl;
    cout << endl;
    
    MatrixXd locProjection = MatrixMult(locChunk, locData, numRows);

    // print the result in each processor
    if (rank == 0) {
        cout << "Size of locProjection in process 0: " << locProjection.rows() << " x " << locProjection.cols() << endl;
        cout << "locProjection on rank " << rank << ":" << endl;
        cout << locProjection << "  " << endl;
    }

    // debugging check the size of local projection
    if (rank == 0) {
        cout << "Size of locProjection in process 0: " << locProjection.rows() << " x " << locProjection.cols() << endl;
    }
    
    // initialize overall projection matrix for final result
    MatrixXd projection(numRows, k);

    // reduce local projections from all processors to the main processor
    MPI_Reduce(locProjection.data(), projection.data(), numRows * k, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // print reduced projection in the main process
    if (rank == 0) {
        cout << "Reduced Projection:" << endl;
        cout << projection << endl;
    }
    auto endProjection = high_resolution_clock::now();
    if (rank == 0) {
        cout << "Data projection time: " << duration_cast<microseconds>(endProjection - startProjection).count() << " microseconds" << endl;
    }
    
    auto endTotal = high_resolution_clock::now();

    if (rank == 0) {
        cout << "Total execution time: " << duration_cast<microseconds>(endTotal - startTotal).count() << " microseconds" << endl;
        /// Open the file in append mode
        ofstream file("results.txt", ios::app);
        if (file.is_open()) {
            // Write the numerical timing results separated by space
            file << size << " " << numRows << " " << numCols << " " << k << " ";
            file << duration_cast<microseconds>(endMean - startMean).count() << " ";
            file << duration_cast<microseconds>(endStdDev - startStdDev).count() << " ";
            file << duration_cast<microseconds>(endNormalization - startNormalization).count() << " ";
            file << duration_cast<microseconds>(endCovMatrix - startCovMatrix).count() << " ";
            file << duration_cast<microseconds>(endEigDecomp - startEigDecomp).count() << " ";
            file << duration_cast<microseconds>(endProjection - startProjection).count() << " ";
            file << duration_cast<microseconds>(endTotal - startTotal).count() << endl;
            file.close();
        }
        else {
            cerr << "Error in opening timing file." << endl;
        }
    }
    MPI_Finalize();

    return 0;
}

#include <iomanip>
#include <iostream>       // input/output stream operations
#include <chrono>         // measurement of execution time
#include <vector>         // usage of vector container
#include <fstream>        // file handling
#include <Eigen/Dense>    // matrix & linear algebra operations --> eigen library

using namespace std;           // standard namespace
using namespace Eigen;         // eigen namespace 
using namespace std::chrono;   

// struct to store timing data and parameters
struct TimingInfo {
    int numRows;
    int numCols;
    long long timeMean;
    long long timeStdDev;
    long long timeNormalization;
    long long timeCovMatrix;
    long long timeEigDecomp;
    long long timeProjection;
    long long timeTotal;
};

// function to save timing details into a file
void saveTimingData(const TimingInfo& info, const string& outputFile) {
    // open the file in append mode (no overwriting)
    ofstream outFile(outputFile, ios::app);  
    if (!outFile.is_open()) {
        cerr << "Error opening file: " << outputFile << endl;
        return;
    }
    outFile << info.numRows << ", " << info.numCols << ", "
            << info.timeMean << ", " << info.timeStdDev << ", " << info.timeNormalization << ", "
            << info.timeCovMatrix << ", " << info.timeEigDecomp << ", " << info.timeProjection << ", "
            << info.timeTotal << endl;
    outFile.close();
}

// helper to print a vector
void printVector(const vector<double>& vec, const string& label) {
    cout << label << ": ";
    for (const auto& val : vec) {
        cout << val << " ";
    }
    cout << endl;
}

// helper to print a matrix
void printMatrix(const MatrixXd& mat, const string& label) {
    cout << label << " (" << mat.rows() << "x" << mat.cols() << "):\n" << mat << endl;
}

// function to compute the mean of each column
vector<double> computeMean(const MatrixXd& dataset) {
    int numRows = dataset.rows();
    int numCols = dataset.cols();
    vector<double> mean(numCols, 0.0);

    for (int col = 0; col < numCols; ++col) {
        double sum = 0.0;
        for (int row = 0; row < numRows; ++row) {
            sum += dataset(row, col);
        }
        mean[col] = sum / numRows;
    }
    return mean;
}

// function to compute the standard deviation for each column
vector<double> computeStdDev(const MatrixXd& dataset, const vector<double>& mean) {
    int numRows = dataset.rows();
    int numCols = dataset.cols();
    vector<double> StdDev(numCols);

    for (int col = 0; col < numCols; ++col) {
        double squaredSum = 0.0;
        for (int row = 0; row < numRows; ++row) {
            double diff = dataset(row, col) - mean[col];
            squaredSum += diff * diff;
        }
        StdDev[col] = sqrt(squaredSum / (numRows - 1));
    }
    return StdDev;
}

// function to standardize the data using z-scores
void normalizeMatrix(MatrixXd& dataset, const vector<double>& mean, const vector<double>& StdDev) {
    int numRows = dataset.rows();
    int numCols = dataset.cols();

    for (int row = 0; row < numRows; ++row) {
        for (int col = 0; col < numCols; ++col) {
            dataset(row, col) = (dataset(row, col) - mean[col]) / StdDev[col];
        }
    }
}

// function to compute the covariance matrix
MatrixXd computeCovarianceMatrix(const MatrixXd& matrix, const MatrixXd& matrixT, int numRows) {
    int RowsT = matrixT.rows();
    int Cols = matrix.cols();
    int ColsT = matrixT.cols();
    MatrixXd covarianceMatrix(RowsT, Cols);

    for (int i = 0; i < RowsT; ++i) {
        for (int j = 0; j < Cols; ++j) {
            double sum = 0.0;
            for (int k = 0; k < ColsT; ++k) {
                sum += matrixT(i, k) * matrix(k, j); // dot product between i and j
            }
            covarianceMatrix(i, j) = sum / (numRows - 1);
        }
    }
    return covarianceMatrix;
}

// function to project data onto selected components (matrix multiplication)
MatrixXd MatrixMult(const MatrixXd& matrix, const MatrixXd& PrinComp, int numRows) {
    int Cols = matrix.cols();
    int PrinCompRows = PrinComp.rows();
    int PrinCompCols = PrinComp.cols();

    MatrixXd result(PrinCompRows, Cols);

    for (int i = 0; i < PrinCompRows; ++i) {
        for (int j = 0; j < Cols; ++j) {
            double sum = 0.0;
            for (int k = 0; k < PrinCompCols; ++k) {
                sum += PrinComp(i, k) * matrix(k, j);
            }
            result(i, j) = sum;
        }
    }
    return result;
}

int main(int argc, char* argv[]) {

    // check if the file path is provided as a command-line argument
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <dataset file path>" << endl;
        return 1;
    }

    string datasetPath = argv[1]; // file name from command-line argument
    //string datasetPath = "datasetL.txt"; // file name
    MatrixXd dataset;

    auto startTotal = high_resolution_clock::now(); // start timer

    // open dataset file
    ifstream inputFile(datasetPath);
    if (!inputFile.is_open()) {
        cerr << "Error opening file: " << datasetPath << endl;
        return 1;
    }

    // Read file and populate the matrix
    string line;
    if (getline(inputFile, line)) {
        stringstream ss(line);
        vector<double> tempValues;
        double val;
        while (ss >> val) {
            tempValues.push_back(val);
        }
        int numCols = tempValues.size();
        int numRows = 1;

        while (getline(inputFile, line)) {
            numRows++;
        }

        inputFile.clear();
        inputFile.seekg(0, ios::beg);
        dataset.resize(numRows, numCols);

        for (int i = 0; i < numRows; ++i) {
            for (int j = 0; j < numCols; ++j) {
                inputFile >> dataset(i, j);
            }
        }
    } else {
        cerr << "Error reading file: " << datasetPath << endl;
        return 1;
    }

    int numRows = dataset.rows();
    int numCols = dataset.cols();
    printMatrix(dataset, "Original Dataset");

    // computation mean recall function
    auto startMean = high_resolution_clock::now();
    vector<double> mean = computeMean(dataset);
    auto endMean = high_resolution_clock::now();
    cout << "Mean calculation time: " << duration_cast<microseconds>(endMean - startMean).count() << " us" << endl;
    printVector(mean, "Mean");

    // computation standard deviation recall function
    auto startStdDev = high_resolution_clock::now();
    vector<double> StdDev = computeStdDev(dataset, mean);
    auto endStdDev = high_resolution_clock::now();
    cout << "Standard deviation calculation time: " << duration_cast<microseconds>(endStdDev - startStdDev).count() << " us" << endl;
    printVector(StdDev, "Standard Deviation");

    // normalization data recall function
    auto startNormalization = high_resolution_clock::now();
    normalizeMatrix(dataset, mean, StdDev);
    auto endNormalization = high_resolution_clock::now();
    cout << "Normalization time: " << duration_cast<microseconds>(endNormalization - startNormalization).count() << " us" << endl;
    printMatrix(dataset, "Normalized Dataset");

    // computation covariance matrix recall function
    auto startCovMatrix = high_resolution_clock::now();
    MatrixXd covarianceMatrix = computeCovarianceMatrix(dataset, dataset.transpose(), numRows);
    auto endCovMatrix = high_resolution_clock::now();
    cout << "Covariance matrix computation time: " << duration_cast<microseconds>(endCovMatrix - startCovMatrix).count() << " us" << endl;
    printMatrix(covarianceMatrix, "Covariance Matrix");

    // perform eigendecomposition recall function
    auto startEigDecomp = high_resolution_clock::now();
    SelfAdjointEigenSolver<MatrixXd> eigensolver(covarianceMatrix);
    if (eigensolver.info() != Success) {
        cerr << "Eigendecomposition failed!" << endl;
        return 1;
    }
    VectorXd eigenvalues = eigensolver.eigenvalues();
    MatrixXd eigenvectors = eigensolver.eigenvectors();
    auto endEigDecomp = high_resolution_clock::now();
    cout << "Eigendecomposition time: " << duration_cast<microseconds>(endEigDecomp - startEigDecomp).count() << " us" << endl;
    printVector(vector<double>(eigenvalues.data(), eigenvalues.data() + eigenvalues.size()), "Eigenvalues");
    printMatrix(eigenvectors, "Eigenvectors");

    // top k PCA
    int k = 3;
    MatrixXd PrincipalComponents = eigenvectors.rightCols(k);
    printMatrix(PrincipalComponents, "Top 3 Principal Components");

    // projecttion data recall function
    auto startProjection = high_resolution_clock::now();
    MatrixXd reducedData = MatrixMult(PrincipalComponents, dataset, numRows);
    auto endProjection = high_resolution_clock::now();
    cout << "Data projection time: " << duration_cast<microseconds>(endProjection - startProjection).count() << " us" << endl;
    printMatrix(reducedData, "Reduced Projection");

    // total execution time
    auto endTotal = high_resolution_clock::now();
    cout << "Total execution time: " << duration_cast<microseconds>(endTotal - startTotal).count() << " us" << endl;
    
    // file opened in append mode
    ofstream file("results.txt", ios::app);
    if (file.is_open()) {
        // write the numerical timing results
        file << numRows << " " << numCols << " ";
        file << (endMean - startMean).count() << " ";
        file << (endStdDev - startStdDev).count() << " ";
        file << (endNormalization - startNormalization).count() << " ";
        file << (endCovMatrix - startCovMatrix).count() << " ";
        file << (endEigDecomp - startEigDecomp).count() << " ";
        file << (endProjection - startProjection).count() << " ";
        file << (endTotal - startTotal).count() << endl;
        file.close();
    }
    else {
        cerr << "Error in opening timing file." << endl;
    }
    return 0;
}





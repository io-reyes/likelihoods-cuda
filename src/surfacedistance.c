#include <mex.h>

extern __declspec(dllimport) float sumDistanceCuda(double*, int, float*, int);

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]){
    // initialize variables
    mxArray *pts, *objPts;
    int numPts, numObjPts;
    double *dataOut;


    // check the input parameters
    if(nrhs != 2)
        mexErrMsgTxt("ERROR: requires 2 3D point matrices as arguments");

    if(nlhs != 1)
        mexErrMsgTxt("ERROR: only 1 output");

    // read the inputs and check their dimensions
    pts    = prhs[0];
    objPts = prhs[1];

    if(mxGetN(pts) != 3 || mxGetN(objPts) != 3)
        mexErrMsgTxt("ERROR: inputs have to be N x 3 or K x 3 matrices of N or K 3D points");

    numPts    = mxGetM(pts);
    numObjPts = mxGetM(objPts);

    // initialize output and save value to it
    plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
    dataOut = mxGetPr(plhs[0]);

    dataOut[0] = sumDistanceCuda(mxGetPr(pts), numPts, (float*)mxGetPr(objPts), numObjPts);
}

#include <mex.h>

extern __declspec(dllimport) float* likelihoodsCuda(double*, int, float*, int, float*, float);

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]){
    // initialize variables
    mxArray *pts, *objPts, *objAreas, *noise;
    int numPts, numObjPts, i;
    double *dataOut, *noiseVal;
    float *likesOut;

    // check the input parameters
    if(nrhs != 4)
        mexErrMsgTxt("ERROR: requires 2 3D point matrices, a vector of areas, and noise standard deviation as arguments");

    if(nlhs != 1)
        mexErrMsgTxt("ERROR: only 1 output");

    // read the inputs and check their dimensions
    pts      = prhs[0];
    objPts   = prhs[1];
    objAreas = prhs[2];
    noise    = prhs[3];

    if(mxGetN(pts) != 3 || mxGetN(objPts) != 3)
        mexErrMsgTxt("ERROR: first 2 inputs have to be N x 3 or K x 3 matrices of N or K 3D points");

    if(mxGetN(objAreas) != 1)
        mexErrMsgTxt("ERROR: 3rd input has to be a K x 1 vector of areas");

    if(mxGetM(noise) != 1 || mxGetN(noise) != 1)
        mexErrMsgTxt("ERROR: 4th input has to be a singleton");

    numPts    = mxGetM(pts);
    numObjPts = mxGetM(objPts);

    noiseVal = mxGetPr(noise);

    // initialize output and save value to it
    plhs[0] = mxCreateDoubleMatrix(numPts, 1, mxREAL);
    dataOut = mxGetPr(plhs[0]);

    likesOut = likelihoodsCuda(mxGetPr(pts), numPts, (float*)mxGetPr(objPts), numObjPts, (float*)mxGetPr(objAreas), (float)noiseVal[0]);

    // copy the results
    for(i = 0; i < numPts; i++)
        dataOut[i] = likesOut[i];
}

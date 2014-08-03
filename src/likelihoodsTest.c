#include <math.h>
#include <float.h>
#include <dirent.h>
#include <string.h>

#include "mat.h"
#include "cudamodules.h"

#define TRUTH_EXT "truth.mat"
#define PTS_EXT "pts.mat"
#define NUM_DIMS 3

int main(int argc, char *argv[])
{
    // declare loop variables;
    int i, j;

    // check for the following 4 arguments:
    //      <pt cloud MAT folder> <truth MAT folder> <noise variance> <output folder>
    if(argc != (4 + 1)){        // +1 because argv[0] is the program name
        printf("Usage: ./likelihoods.exe <pt cloud MAT folder> <truth MAT folder> <noise variance> <output folder>");
        return -1;
    }

    char* ptFolder  = argv[1];
    char* trtFolder = argv[2];
    char* outFolder = argv[4];
    double noise    = sqrt(atof(argv[3]));

    // read in the truth files
    DIR* d = opendir(trtFolder);
    if(d == NULL){
        printf("Error: could not open folder %s", trtFolder);
        return -1;
    }

    // find the number of truth files
    struct dirent* file;
    int numTruth      = 0;
    int maxFileLength = 0;
    while((file = readdir(d)) != NULL){
        if(strstr(file->d_name, TRUTH_EXT)){
            maxFileLength = (strlen(file->d_name) > maxFileLength) ? strlen(file->d_name) : maxFileLength;
            numTruth++;
        }
    }

    // allocate memory for the truth files and read them in
    float** truthPts   = (float**)malloc(numTruth * sizeof(float*));
    float** truthAreas = (float**)malloc(numTruth * sizeof(float*));
    int* numObjPts     = (int*)malloc(numTruth * sizeof(int));
    char* pathBuffer   = (char*)malloc((strlen(trtFolder) + maxFileLength + 1) * sizeof(char)); 

    MATFile* mf;
    mxArray* readTrtPts(NULL);
    mxArray* readTrtArea(NULL);

    i = 0;
    rewinddir(d);
    while((file = readdir(d)) != NULL){
        if(!strstr(file->d_name, TRUTH_EXT))
            continue;

        sprintf(pathBuffer, "%s%s", trtFolder, file->d_name);
        printf("Loading truth file: %s\n", file->d_name);

        // open the MAT file and read in the variables "object_point" and "object_area"
        mf = matOpen(pathBuffer, "r");

        readTrtPts  = matGetVariable(mf, "object_point");
        readTrtArea = matGetVariable(mf, "object_area");

        // allocate and save to memory
        numObjPts[i]  = mxGetM(readTrtPts);
        truthPts[i]   = (float*)malloc(NUM_DIMS * numObjPts[i] * sizeof(float));
        truthAreas[i] = (float*)malloc(numObjPts[i] * sizeof(float));



        // close file and increment counter
        matClose(mf);
        i++;
    }
    

    closedir(d);

    // free allocated memory
    /*for(i = 0; i < numTruth; i++){
        free(truthPts[i]);
        free(truthAreas[i]);
    }
    free(truthPts);
    free(truthAreas);
    free(numObjPts);
    free(pathBuffer);*/
}

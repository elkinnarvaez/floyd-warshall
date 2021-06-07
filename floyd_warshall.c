#include <stdio.h>
#include <omp.h>
#include <stdlib.h>

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

int dis[1000][1000];
int n;

void printDis() {
    int i, j;
    printf("    ");
    for (i = 0; i < n; ++i)
        printf("%4c", 'A' + i);
    printf("\n");
    for (i = 0; i < n; ++i) {
        printf("%4c", 'A' + i);
        for (j = 0; j < n; ++j)
            printf("%4d", dis[i][j]);
        printf("\n");
    }
    printf("\n");
}

void floyd_warshall_sequential() {
    int i, j, k;
    for (k = 0; k < n; ++k)
        for (i = 0; i < n; ++i)
            for (j = 0; j < n; ++j)
                dis[i][j] = MIN(dis[i][j], dis[i][k] + dis[k][j]);
}


void floyd_warshall_parallel(){
    int i, j, k;
    for (k = 0; k < n; ++k)
        #pragma omp parallel for private(i,j)
        for (i = 0; i < n; ++i)
            for (j = 0; j < n; ++j)
                dis[i][j] = MIN(dis[i][j], dis[i][k] + dis[k][j]);
}

void floyd_warshall_parallel2(){
    int i, j, k;
    for (k = 0; k < n; ++k)
        #pragma omp parallel for private(i,j)
        for (i = 0; i < n; ++i)
            #pragma omp parallel for private(j)
            for (j = 0; j < n; ++j)
                dis[i][j] = MIN(dis[i][j], dis[i][k] + dis[k][j]);
}

void individual_example_running_time(){
    n = 10;
    double start,stop;
    int i, j;
    for (i = 0; i < n; i++){
        for (j = 0; j < n; j++){
            if (i == j){
                dis[i][j] = 0;
            }
            else{
                dis[i][j] = (int)(11.0 * rand() / ( RAND_MAX + 1.0));
            }
        }
    }

    start = omp_get_wtime();
    floyd_warshall_parallel();
    stop = omp_get_wtime();
    
    printf("Time: %f\n", stop - start);
}

void multiple_examples_running_time(){
    for(n = 3; n < 5000; n = n + 500){
        double sum_time = 0;
        int num_iter = 3;
        for(int i = 0; i < num_iter; i++){
            double start,stop;
            int i, j;
            for (i = 0; i < n; i++){
                for (j = 0; j < n; j++){
                    if (i == j){
                        dis[i][j] = 0;
                    }
                    else{
                        dis[i][j] = (int)(11.0 * rand() / ( RAND_MAX + 1.0));
                    }
                }
            }

            start = omp_get_wtime();
            floyd_warshall_parallel();
            stop = omp_get_wtime();
        
            // printf("time %f\n",stop-start);
            sum_time = sum_time + (stop - start);
        }
        // printf("Time for n = %d is %f\n", n, sum_time/num_iter);
        printf("%d %f\n", n, sum_time/num_iter);
    }
}

void correcteness_test(){
    n = 4;
    dis[0][0] = 0; dis[0][1] = 3; dis[0][2] = 3; dis[0][3] = 2;
    dis[1][0] = 10; dis[1][1] = 0; dis[1][2] = 3; dis[1][3] = 3;
    dis[2][0] = 3; dis[2][1] = 4; dis[2][2] = 0; dis[2][3] = 7;
    dis[3][0] = 7; dis[3][1] = 4; dis[3][2] = 8; dis[3][3] = 0;
    floyd_warshall_sequential();
    printDis();
}

int main(){
    correcteness_test();
    return 0;
}
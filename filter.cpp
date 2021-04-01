#include "common.h"
#include <immintrin.h>
#include <math.h>

void slow_performance1(double *x, double* h, double* y, int N, int M) {
  for (int i = 0; i < N - (M - 1); i++) {
    y[i] = 0.0;
    for (int k = 0; k < M; k++) {
      y[i] += h[k] * fabs(x[i + (M - 1) - k]);
    }
  }
}
void maxperformance(double *x, double* h, double* y, int N, int M) {
  /* This is the most optimized version. */
  
  __m256d signMask=_mm256_set1_pd(-0.0);
  // __m256d xVector,xVector2;
  // __m256d acc1;
  __m256d zeros=_mm256_setzero_pd();
  __m256d h0,h1,h2,h3,x0,x1,x2,x3;
  __m256d x0u,x1u,x2u,x3u;
  __m256d x0u2,x1u2,x2u2,x3u2;
  __m256d x0u3,x1u3,x2u3,x3u3;
  
  h0=_mm256_set1_pd(h[0]);
  h1=_mm256_set1_pd(h[1]);
  h2=_mm256_set1_pd(h[2]);
  h3=_mm256_set1_pd(h[3]);
  

  int end=N-N%16;
  int i=0;
  for(;i<end;i+=16)
  {
    // acc1=zeros;
    x0=_mm256_loadu_pd(x+i+3);
    x0=_mm256_andnot_pd(signMask,x0);
    x0=_mm256_mul_pd(h0,x0);
    
    x1=_mm256_loadu_pd(x+i+2);
    x1=_mm256_andnot_pd(signMask,x1);
    x0=_mm256_fmadd_pd(h1,x1,x0);
    
    x2=_mm256_loadu_pd(x+i+1);
    x2=_mm256_andnot_pd(signMask,x2);
    x0=_mm256_fmadd_pd(h2,x2,x0);
    
    x3=_mm256_loadu_pd(x+i);
    x3=_mm256_andnot_pd(signMask,x3);
    // x3=_mm256_fmadd_pd(h3,x3,zeros);
    x0=_mm256_fmadd_pd(h3,x3,x0);
    _mm256_storeu_pd(y+i,x0);
    
    // acc1=zeros;
    x0u=_mm256_loadu_pd(x+i+7);
    x0u=_mm256_andnot_pd(signMask,x0u);
    x0u=_mm256_fmadd_pd(h0,x0u,zeros);
    
    x1u=_mm256_loadu_pd(x+i+6);
    x1u=_mm256_andnot_pd(signMask,x1u);
    x0u=_mm256_fmadd_pd(h1,x1u,x0u);
    
    x2u=_mm256_loadu_pd(x+i+5);
    x2u=_mm256_andnot_pd(signMask,x2u);
    x0u=_mm256_fmadd_pd(h2,x2u,x0u);
    
    x3u=_mm256_loadu_pd(x+i+4);
    x3u=_mm256_andnot_pd(signMask,x3u);
    x0u=_mm256_fmadd_pd(h3,x3u,x0u);
    _mm256_storeu_pd(y+i+4,x0u);
    
    x0u2=_mm256_loadu_pd(x+i+11);
    x0u2=_mm256_andnot_pd(signMask,x0u2);
    x0u2=_mm256_fmadd_pd(h0,x0u2,zeros);
    
    x1u2=_mm256_loadu_pd(x+i+10);
    x1u2=_mm256_andnot_pd(signMask,x1u2);
    x0u2=_mm256_fmadd_pd(h1,x1u2,x0u2);
    
    x2u2=_mm256_loadu_pd(x+i+9);
    x2u2=_mm256_andnot_pd(signMask,x2u2);
    x0u2=_mm256_fmadd_pd(h2,x2u2,x0u2);
    
    x3u2=_mm256_loadu_pd(x+i+8);
    x3u2=_mm256_andnot_pd(signMask,x3u2);
    x0u2=_mm256_fmadd_pd(h3,x3u2,x0u2);
    _mm256_storeu_pd(y+i+8,x0u2);
    
    x0u3=_mm256_loadu_pd(x+i+15);
    x0u3=_mm256_andnot_pd(signMask,x0u3);
    x0u3=_mm256_fmadd_pd(h0,x0u3,zeros);
    
    x1u3=_mm256_loadu_pd(x+i+14);
    x1u3=_mm256_andnot_pd(signMask,x1u3);
    x0u3=_mm256_fmadd_pd(h1,x1u3,x0u3);
    
    x2u3=_mm256_loadu_pd(x+i+13);
    x2u3=_mm256_andnot_pd(signMask,x2u3);
    x0u3=_mm256_fmadd_pd(h2,x2u3,x0u3);
    
    x3u3=_mm256_loadu_pd(x+i+12);
    x3u3=_mm256_andnot_pd(signMask,x3u3);
    x0u3=_mm256_fmadd_pd(h3,x3u3,x0u3);
    
    
    // _mm256_store_pd(y+i,_mm256_add_pd(acc,acc2));
    _mm256_storeu_pd(y+i+12,x0u3);
    
  }
  

    //remaining elements
    //assume M=4
    
    for (; i < N - (M - 1); i++) 
    {
      y[i]=h[0]*fabs(x[i+3])+h[1]*fabs(x[i+2])+h[2]*fabs(x[i+1])+h[3]*fabs(x[i]);

    }
  

}

/*
* Called by the driver to register your functions
* Use add_function(func, description) to add your own functions
*/
void register_functions() 
{
  add_function(&slow_performance1, "slow_performance1",1);
  add_function(&maxperformance, "maxperformance",1);
}
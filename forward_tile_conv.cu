const int M          = 32; // the height of the output matrix
const int N          = 32; // the width of the output matrix
const int K          = 4;  // the length of the intermediate dimension in A*B
const int UNROLL     = 32 / K;
const int BLOCKDIM_Y = M / UNROLL;
const int BLOCKDIM_X = N;
const int BLOCKDIM_Z = 1;

__global__ void conv_forward_opt_kernel(const float *X, const shape xdims, const float *W, const shape wdims, float *Y, const shape ydims) {

  // X[b, c, h+p, w+q] = X[((b * xdims.depth + c) * xdims.height + (h + p)) * xdims.width + (w + q)]
  // W[m, c, p, q] = W[((m * wdims.depth + c) * wdims.height + p) * wdims.width + q]
  // Y[b, m, h, w] = Y[((b * ydims.depth + m) * ydims.height + h) * ydims.width + w]
  const int b     = ydims.num;
  const int wh    = wdims.height;
  const int ww    = wdims.width;
  const int h_in  = ydims.height + wh - 1;
  const int w_in  = ydims.width + ww - 1;
  const int c     = xdims.depth;
  const int m     = ydims.depth;
  const int h_out = h_in - wh + 1;
  const int w_out = w_in - ww + 1;

  int idxXBlockCol = blockIdx.x * m + threadIdx.x;
  int idxM         = blockIdx.y * N + threadIdx.y;

  int numXBlockRows = c * wh * ww;
  int numXBlockCols = h_out * w_out;

  __shared__ float blockMemA[M][K];
  __shared__ float blockMemB[K][N];

  int idxB    = blockIdx.z * BLOCKDIM_Z + threadIdx.z;
  int idxHOut = idxXBlockCol / w_out;
  int idxWOut = idxXBlockCol % w_out;

  if (idxB < b) {
    float PValue[UNROLL];
    for (int unroll = 0; unroll < UNROLL; unroll++) {
      PValue[unroll] = 0.0;
    }

    for (int idx = 0; idx < (numXBlockRows + K - 1) / K; idx++) {
// simplify the matrix indexing
#define Y4d(i3, i2, i1, i0) Y[(i3) * (m * h_out * w_out) + (i2) * (h_out * w_out) + (i1) * (w_out) + i0]
#define W4d(i3, i2, i1, i0) X[(i3) * (c * h_in * w_in) + (i2) * (h_in * w_in) + (i1) * (w_in) + i0]
#define X4d(i3, i2, i1, i0) W[(i3) * (c * wh * ww) + (i2) * (wh * ww) + (i1) * (ww) + i0]

      const int threadIdx1D      = threadIdx.y * BLOCKDIM_X + threadIdx.x;
      const int threadNumInBlock = BLOCKDIM_X * BLOCKDIM_Y;
      int max                    = (M * K + threadNumInBlock - 1) / threadNumInBlock;
      if (((K * N + threadNumInBlock - 1) / threadNumInBlock) > max)
        max = (K * N + threadNumInBlock - 1) / threadNumInBlock;
      for (int idxLoad = 0; idxLoad < max; idxLoad++) {
        int idxYBlockMemALoad = (threadIdx1D + idxLoad * threadNumInBlock) / K;
        int idxXBlockMemALoad = (threadIdx1D + idxLoad * threadNumInBlock) % K;
        int idxYBlockMemBLoad = (threadIdx1D + idxLoad * threadNumInBlock) / N;
        int idxXBlockMemBLoad = (threadIdx1D + idxLoad * threadNumInBlock) % N;

        int idxWColumnToLoad   = idx * K + idxXBlockMemALoad;
        int idxXBlockColToLoad = blockIdx.x * N + idxXBlockMemBLoad;
        int idxXBlockRowToLoad = (idxYBlockMemBLoad + idx * K);

        int idxHOutToLoad = idxXBlockColToLoad / w_out;
        int idxWOutToLoad = idxXBlockColToLoad % w_out;

        int idxCWColumn   = idxWColumnToLoad / (wh * ww);
        int idxMToLoad    = blockIdx.y * M + idxYBlockMemALoad;
        int idxCXBlockRow = idxXBlockRowToLoad / (wh * ww);
        if ((idxCWColumn < c) && (idxMToLoad < m)) {
          int idxKHeightWColumn                           = (idxWColumnToLoad % (wh * ww)) / ww;
          int idxKWidthWColumn                            = (idxWColumnToLoad % (wh * ww)) % ww;
          blockMemA[idxYBlockMemALoad][idxXBlockMemALoad] = X4d(idxMToLoad, idxCWColumn, idxKHeightWColumn, idxKWidthWColumn);
        } else {
          blockMemA[idxYBlockMemALoad][idxXBlockMemALoad] = 0.0;
        }
        if ((idxCXBlockRow < c) && (idxHOutToLoad < h_out)) {
          int idxKHeightXBlockRow = (idxXBlockRowToLoad % (wh * ww)) / ww;
          int idxKWidthXBlockRow  = (idxXBlockRowToLoad % (wh * ww)) % ww;
          blockMemB[idxYBlockMemBLoad][idxXBlockMemBLoad] =
              W4d(idxB, idxCXBlockRow, idxHOutToLoad + idxKHeightXBlockRow, idxWOutToLoad + idxKWidthXBlockRow);
        } else {
          blockMemB[idxYBlockMemBLoad][idxXBlockMemBLoad] = 0.0;
        }
      }
      __syncthreads();
      for (int idx2 = 0; idx2 < K; idx2++) {
        for (int unroll = 0; unroll < UNROLL; unroll++) {
          PValue[unroll] += blockMemA[threadIdx.y + unroll * BLOCKDIM_Y][idx2] * blockMemB[idx2][threadIdx.x];
        }
      }
      __syncthreads();
    }
    if (idxM < M && idxXBlockCol < numXBlockCols) {
      for (int unroll = 0; unroll < UNROLL; unroll++) {
        Y4d(idxB, idxM + unroll * BLOCKDIM_Y, idxHOut, idxWOut) = PValue[unroll];
      }
    }
  }
}

static void convlayer_gpu_opt(const float *X, const shape &xdims, const float *W, const shape &wdims, float *Y, const shape &ydims) {
  // X[b, c, h+p, w+q] = X[((b * xdims.depth + c) * xdims.height + (h + p)) * xdims.width + (w + q)]
  // W[m, c, p, q] = W[((m * wdims.depth + c) * wdims.height + p) * wdims.width + q]
  // Y[b, m, h, w] = Y[((b * ydims.depth + m) * ydims.height + h) * ydims.width + w]
  const int wh      = wdims.height;
  const int ww      = wdims.width;
  const int m       = ydims.depth;
  const int h_in    = ydims.height + wh - 1;
  const int w_in    = ydims.width + ww - 1;
  const int h_out   = h_in - wh + 1;
  const int w_out   = w_in - ww + 1;
  const int b       = ydims.num;
  int numWtiledRows = m;
  int numXBlockCols = h_out * w_out;
  dim3 blockUnrollDimGolden(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
  dim3 gridUnrollDimGolden((numXBlockCols + BLOCKDIM_X - 1) / BLOCKDIM_X,
                           (numWtiledRows + BLOCKDIM_Y * UNROLL - 1) / (BLOCKDIM_Y * UNROLL),
                           (b + BLOCKDIM_Z - 1) / BLOCKDIM_Z);
  THROW_IF_ERROR(cudaMemset(Y, 0, sizeof(float) * ydims.num * ydims.depth * ydims.height * ydims.width));
  conv_forward_opt_kernel<<<gridUnrollDimGolden, blockUnrollDimGolden>>>(X, xdims, W, wdims, Y, ydims);
  THROW_IF_ERROR(cudaGetLastError());
}


#include <mpi.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

int bitcount(long bits) {
  bits = (bits & 0x55555555) + (bits >> 1 & 0x55555555);
  bits = (bits & 0x33333333) + (bits >> 2 & 0x33333333);
  bits = (bits & 0x0f0f0f0f) + (bits >> 4 & 0x0f0f0f0f);
  bits = (bits & 0x00ff00ff) + (bits >> 8 & 0x00ff00ff);
  return (bits & 0x0000ffff) + (bits >> 16 & 0x0000ffff);
}

class SampleCommunication {
 public:
  static const size_t BUF_MAX = 10;
  static void send_data(int dest, MPI_Request *request);
  static void recv_data(int source, MPI_Request *request);
};

static int proc_i, proc_n;

#define world_print(fmt, ...)          \
  do {                                 \
    MPI_Barrier(MPI_COMM_WORLD);       \
    if (proc_i == 0) {                 \
      std::printf(fmt, ##__VA_ARGS__); \
    }                                  \
    MPI_Barrier(MPI_COMM_WORLD);       \
  } while (false)

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_i);
  MPI_Comm_size(MPI_COMM_WORLD, &proc_n);
  {
    assert(bitcount(proc_n) == 1);

    MPI_Request requests[1];
    MPI_Status statuses[1];

    static constexpr int BUF_MAX = 10;
    // std::vector<double> send_buf(BUF_MAX);
    // std::vector<double> recv_buf(BUF_MAX);
    double send_buf[BUF_MAX];
    double recv_buf[BUF_MAX];
    for (int i = 0; i < BUF_MAX; i++) send_buf[i] = (i + 1) * proc_i;

    std::cout << proc_i << ": [";
    for (int i = 0; i < BUF_MAX; i++) std::cout << " " << send_buf[i];
    std::cout << " ]" << std::endl;

    int n = proc_n;
    while (n > 0) {
      const int mask = n - 1;
      const int pair = proc_i ^ mask;

      if (pair > proc_i) {
        MPI_Irecv(&recv_buf[0], BUF_MAX, MPI_DOUBLE, pair, 0, MPI_COMM_WORLD,
                  &requests[0]);
      } else {
        MPI_Isend(&send_buf[0], BUF_MAX, MPI_DOUBLE, pair, 0, MPI_COMM_WORLD,
                  &requests[0]);
      }
      MPI_Waitall(1, requests, statuses);

      if (pair > proc_i) {
        for (int i = 0; i < BUF_MAX; i++) send_buf[i] += recv_buf[i];
      }

      n >>= 1;
    }

    if (proc_i == 0) {
      std::cout << proc_i << ": sum up result : [";
      for (int i = 0; i < BUF_MAX; i++) std::cout << " " << send_buf[i];
      std::cout << " ]" << std::endl;
    }
  }

  {
    world_print("----- MPI_Bcast: start\n");

    double buf[10];
    double *address = buf;

    if (proc_i == 0) {
      for (int i = 0; i < 10; i++) buf[i] = i;
    }

    MPI_Bcast(buf, 10, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::cout << proc_i << ": [";
    for (int i = 0; i < 10; i++) std::cout << " " << buf[i];
    std::cout << "] " << std::endl;
    world_print("----- MPI_Bcast: end\n");
  }
  {
    world_print("----- MPI_Scatter and MPI_Gather: start\n");

    const int master_size = 40;
    std::vector<double> master(master_size);
    for (int i = 0; i < master_size; i++) master[i] = i;

    assert(master_size % proc_n == 0);
    const int slice_size = master_size / proc_n;
    const int scatter_send_size = (proc_i == 0) ? slice_size : 0;
    std::vector<double> slice(slice_size);

    if (proc_i == 0) {
      std::cout << "rank: 0, master: [" << std::endl;
      for (int k = 0; k < proc_n; k++) {
        for (int i = 0; i < slice_size; i++) {
          std::cout << " " << master[k * slice_size + i];
        }
        std::cout << std::endl;
      }
      std::cout << "]" << std::endl;
    }

    // second arg is buf size to send *** for each proc ***
    MPI_Scatter(master.data(), (proc_i == 0) ? slice_size : 0, MPI_DOUBLE,
                slice.data(), slice_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // calculate in each proc
    for (int i = 0; i < slice_size; i++) slice[i] = slice[i] * proc_i;

    MPI_Gather(slice.data(), slice_size, MPI_DOUBLE, master.data(),
               (proc_i == 0) ? slice_size : 0, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    if (proc_i == 0) {
      std::cout << "rank: 0, master: [" << std::endl;
      for (int k = 0; k < proc_n; k++) {
        for (int i = 0; i < slice_size; i++) {
          std::cout << " " << master[k * slice_size + i];
        }
        std::cout << std::endl;
      }
      std::cout << "]" << std::endl;
    }

    world_print("----- MPI_Scatter and MPI_Gather: end\n");
  }

  MPI_Finalize();
  return 0;
}

void SampleCommunication::send_data(int dest, MPI_Request *request) {
  double buf[BUF_MAX];
  for (int i = 0; i < BUF_MAX; i++) buf[i] = (i + 1) * proc_i;

  const double *buf_address = buf;

  std::cout << proc_i << " > " << dest << ": ";
  std::cout << " [";
  for (int i = 0; i < BUF_MAX; i++) std::cout << " " << buf[i];
  std::cout << " ]" << std::endl;

  MPI_Isend(buf_address, BUF_MAX, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD, request);
}

void SampleCommunication::recv_data(int source, MPI_Request *request) {
  double buf[BUF_MAX];
  double *buf_address = buf;

  std::cout << proc_i << " < " << source << ": ";
  MPI_Irecv(buf_address, BUF_MAX, MPI_DOUBLE, source, 0, MPI_COMM_WORLD,
            request);
  std::cout << " [";
  for (int i = 0; i < BUF_MAX; i++) std::cout << " " << buf[i];
  std::cout << " ]" << std::endl;
}
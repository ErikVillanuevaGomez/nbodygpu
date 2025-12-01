#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <string>
#include <cuda_runtime.h>
#include <algorithm>
#include <cctype>

#define G 6.674e-11
#define SOFTENING 0.1

#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


struct simulation {
  size_t nbpart;
  
  std::vector<double> mass;

  std::vector<double> x;
  std::vector<double> y;
  std::vector<double> z;

  std::vector<double> vx;
  std::vector<double> vy;
  std::vector<double> vz;

  std::vector<double> fx;
  std::vector<double> fy;
  std::vector<double> fz;

  simulation(size_t nb)
    :nbpart(nb), mass(nb),
     x(nb), y(nb), z(nb),
     vx(nb), vy(nb), vz(nb),
     fx(nb), fy(nb), fz(nb)
  {}
};

void random_init(simulation& s) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dismass(0.9, 1.);
  std::normal_distribution<double> dispos(0., 1.);
  std::normal_distribution<double> disvel(0., 1.);

  for (size_t i = 0; i<s.nbpart; ++i) {
    s.mass[i] = dismass(gen);
    s.x[i] = dispos(gen);
    s.y[i] = dispos(gen);
    s.z[i] = dispos(gen);
    
    s.vx[i] = disvel(gen);
    s.vy[i] = disvel(gen);
    s.vz[i] = disvel(gen);
  }
}

void init_solar(simulation& s) {
  enum Planets {SUN, MERCURY, VENUS, EARTH, MARS, JUPITER, SATURN, URANUS, NEPTUNE, MOON};
  s = simulation(10);

  s.mass[SUN] = 1.9891 * std::pow(10, 30);
  s.mass[MERCURY] = 3.285 * std::pow(10, 23);
  s.mass[VENUS] = 4.867 * std::pow(10, 24);
  s.mass[EARTH] = 5.972 * std::pow(10, 24);
  s.mass[MARS] = 6.39 * std::pow(10, 23);
  s.mass[JUPITER] = 1.898 * std::pow(10, 27);
  s.mass[SATURN] = 5.683 * std::pow(10, 26);
  s.mass[URANUS] = 8.681 * std::pow(10, 25);
  s.mass[NEPTUNE] = 1.024 * std::pow(10, 26);
  s.mass[MOON] = 7.342 * std::pow(10, 22);

  double AU = 1.496 * std::pow(10, 11);

  s.x = {0, 0.39*AU, 0.72*AU, 1.0*AU, 1.52*AU, 5.20*AU, 9.58*AU, 19.22*AU, 30.05*AU, 1.0*AU + 3.844*std::pow(10, 8)};
  s.y = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  s.z = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  s.vx = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  s.vy = {0, 47870, 35020, 29780, 24130, 13070, 9680, 6800, 5430, 29780 + 1022};
  s.vz = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
}

void load_from_file(simulation& s, std::string filename) {
  std::ifstream in (filename);
  size_t nbpart;
  in >> nbpart;
  s = simulation(nbpart);
  for (size_t i=0; i<s.nbpart; ++i) {
    in >> s.mass[i];
    in >> s.x[i] >> s.y[i] >> s.z[i];
    in >> s.vx[i] >> s.vy[i] >> s.vz[i];
    in >> s.fx[i] >> s.fy[i] >> s.fz[i];
  }
}

void dump_state(simulation& s) {
  std::cout << s.nbpart << '\t';
  for (size_t i=0; i<s.nbpart; ++i) {
    std::cout << s.mass[i] << '\t';
    std::cout << s.x[i] << '\t' << s.y[i] << '\t' << s.z[i] << '\t';
    std::cout << s.vx[i] << '\t' << s.vy[i] << '\t' << s.vz[i] << '\t';
    std::cout << s.fx[i] << '\t' << s.fy[i] << '\t' << s.fz[i] << '\t';
  }
  std::cout << '\n';
}


__global__ void compute_forces_kernel(int n, double *mass,
                                      double *x, double *y, double *z,
                                      double *fx, double *fy, double *fz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        double net_fx = 0.0;
        double net_fy = 0.0;
        double net_fz = 0.0;

        double xi = x[i];
        double yi = y[i];
        double zi = z[i];
        double mi = mass[i];

        for (int j = 0; j < n; j++) {
            if (i != j) {
                double dx = xi - x[j];
                double dy = yi - y[j];
                double dz = zi - z[j];
                
                double dist_sq = dx*dx + dy*dy + dz*dz + (SOFTENING * SOFTENING);
                double dist = sqrt(dist_sq);
                
                double f = (G * mi * mass[j]) / dist_sq;

                
                net_fx -= f * (dx / dist);
                net_fy -= f * (dy / dist);
                net_fz -= f * (dz / dist);
            }
        }
        fx[i] = net_fx;
        fy[i] = net_fy;
        fz[i] = net_fz;
    }
}

__global__ void update_physics_kernel(int n, double dt, double *mass,
                                      double *x, double *y, double *z,
                                      double *vx, double *vy, double *vz,
                                      double *fx, double *fy, double *fz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        double ax = fx[i] / mass[i];
        double ay = fy[i] / mass[i];
        double az = fz[i] / mass[i];

        vx[i] += ax * dt;
        vy[i] += ay * dt;
        vz[i] += az * dt;

        x[i] += vx[i] * dt;
        y[i] += vy[i] * dt;
        z[i] += vz[i] * dt;
        
        fx[i] = 0.0;
        fy[i] = 0.0;
        fz[i] = 0.0;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 6) {
        std::cerr << "usage: " << argv[0] << " <input> <dt> <nbstep> <printevery> <blocksize>\n";
        return -1;
    }

    double dt = std::atof(argv[2]);
    size_t nbstep = std::atol(argv[3]);
    size_t printevery = std::atol(argv[4]);
    int blockSize = std::atoi(argv[5]);

    simulation s(1);

    {
        std::string inputparam = argv[1];
        bool isNumber = !inputparam.empty() && std::all_of(inputparam.begin(), inputparam.end(), ::isdigit);
        
        if (isNumber) {
            size_t nbpart = std::stol(inputparam);
            if (nbpart > 0) {
                s = simulation(nbpart);
                random_init(s);
            }
        } else if (inputparam == "planet") {
            init_solar(s);
        } else {
            load_from_file(s, inputparam);
        }
    }

    size_t n = s.nbpart;
    size_t bytes = n * sizeof(double);

    double *d_mass, *d_x, *d_y, *d_z, *d_vx, *d_vy, *d_vz, *d_fx, *d_fy, *d_fz;
    cudaCheckError(cudaMalloc(&d_mass, bytes));
    cudaCheckError(cudaMalloc(&d_x, bytes));
    cudaCheckError(cudaMalloc(&d_y, bytes));
    cudaCheckError(cudaMalloc(&d_z, bytes));
    cudaCheckError(cudaMalloc(&d_vx, bytes));
    cudaCheckError(cudaMalloc(&d_vy, bytes));
    cudaCheckError(cudaMalloc(&d_vz, bytes));
    cudaCheckError(cudaMalloc(&d_fx, bytes));
    cudaCheckError(cudaMalloc(&d_fy, bytes));
    cudaCheckError(cudaMalloc(&d_fz, bytes));

    cudaCheckError(cudaMemcpy(d_mass, s.mass.data(), bytes, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_x, s.x.data(), bytes, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_y, s.y.data(), bytes, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_z, s.z.data(), bytes, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_vx, s.vx.data(), bytes, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_vy, s.vy.data(), bytes, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_vz, s.vz.data(), bytes, cudaMemcpyHostToDevice));
    
    cudaCheckError(cudaMemset(d_fx, 0, bytes));
    cudaCheckError(cudaMemset(d_fy, 0, bytes));
    cudaCheckError(cudaMemset(d_fz, 0, bytes));

    int gridSize = (n + blockSize - 1) / blockSize;

    for (size_t step = 0; step < nbstep; step++) {
        
        if (step % printevery == 0) {
            cudaCheckError(cudaMemcpy(s.x.data(), d_x, bytes, cudaMemcpyDeviceToHost));
            cudaCheckError(cudaMemcpy(s.y.data(), d_y, bytes, cudaMemcpyDeviceToHost));
            cudaCheckError(cudaMemcpy(s.z.data(), d_z, bytes, cudaMemcpyDeviceToHost));
            cudaCheckError(cudaMemcpy(s.vx.data(), d_vx, bytes, cudaMemcpyDeviceToHost));
            cudaCheckError(cudaMemcpy(s.vy.data(), d_vy, bytes, cudaMemcpyDeviceToHost));
            cudaCheckError(cudaMemcpy(s.vz.data(), d_vz, bytes, cudaMemcpyDeviceToHost));
            cudaCheckError(cudaMemcpy(s.fx.data(), d_fx, bytes, cudaMemcpyDeviceToHost));
            cudaCheckError(cudaMemcpy(s.fy.data(), d_fy, bytes, cudaMemcpyDeviceToHost));
            cudaCheckError(cudaMemcpy(s.fz.data(), d_fz, bytes, cudaMemcpyDeviceToHost));
            dump_state(s);
        }

        compute_forces_kernel<<<gridSize, blockSize>>>(n, d_mass, d_x, d_y, d_z, d_fx, d_fy, d_fz);
        cudaCheckError(cudaGetLastError());

        update_physics_kernel<<<gridSize, blockSize>>>(n, dt, d_mass, d_x, d_y, d_z, d_vx, d_vy, d_vz, d_fx, d_fy, d_fz);
        cudaCheckError(cudaGetLastError());
    }

    cudaFree(d_mass); cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
    cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_vz);
    cudaFree(d_fx); cudaFree(d_fy); cudaFree(d_fz);

    return 0;
}

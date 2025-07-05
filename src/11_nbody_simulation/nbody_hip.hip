#define HIP_ENABLE_WARP_SYNC_BUILTINS
#include <hip/hip_runtime.h>
#include <cmath>
#include <iostream>

#define BLOCK_SIZE 256
#define TILE_SIZE 64
#define SOFTENING 1e-9f
#define G_CONSTANT 6.67430e-11f

// Particle structure optimized for memory coalescing
struct Particle {
    float4 position;  // x, y, z, mass
    float4 velocity;  // vx, vy, vz, padding
    float4 force;     // fx, fy, fz, potential
};

// Structure of Arrays for better memory access patterns
struct ParticleArrays {
    float4* positions;   // x, y, z, mass
    float4* velocities;  // vx, vy, vz, padding
    float4* forces;      // fx, fy, fz, potential
};

// Advanced N-body force calculation with shared memory optimization
__global__ void calculateForces_optimized(
    float4* positions,
    float4* forces,
    int n,
    float softening
) {
    extern __shared__ float4 shared_pos[];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    
    float4 my_pos = {0.0f, 0.0f, 0.0f, 0.0f};
    float4 total_force = {0.0f, 0.0f, 0.0f, 0.0f};
    
    if (gid < n) {
        my_pos = positions[gid];
    }
    
    // Tile-based computation to optimize memory bandwidth
    for (int tile = 0; tile < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; tile++) {
        int idx = tile * BLOCK_SIZE + tid;
        
        // Collaboratively load tile data into shared memory
        if (idx < n) {
            shared_pos[tid] = positions[idx];
        } else {
            shared_pos[tid] = {0.0f, 0.0f, 0.0f, 0.0f};
        }
        __syncthreads();
        
        // Calculate forces with all particles in current tile
        for (int j = 0; j < BLOCK_SIZE; j++) {
            float4 other_pos = shared_pos[j];
            
            // Calculate distance vector
            float dx = other_pos.x - my_pos.x;
            float dy = other_pos.y - my_pos.y;
            float dz = other_pos.z - my_pos.z;
            
            // Calculate distance squared with softening
            float dist_sq = dx*dx + dy*dy + dz*dz + softening*softening;
            float inv_dist = rsqrtf(dist_sq);
            float inv_dist3 = inv_dist * inv_dist * inv_dist;
            
            // Calculate force magnitude
            float force_mag = G_CONSTANT * my_pos.w * other_pos.w * inv_dist3;
            
            // Accumulate force components
            total_force.x += force_mag * dx;
            total_force.y += force_mag * dy;
            total_force.z += force_mag * dz;
            
            // Accumulate potential energy
            total_force.w -= G_CONSTANT * my_pos.w * other_pos.w * inv_dist;
        }
        __syncthreads();
    }
    
    // Store computed forces
    if (gid < n) {
        forces[gid] = total_force;
    }
}

// Warp-optimized force calculation using shuffle operations
__global__ void calculateForces_warp_optimized(
    float4* positions,
    float4* forces,
    int n
) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    extern __shared__ float4 shared_positions[];
    
    float4 my_pos = {0.0f, 0.0f, 0.0f, 0.0f};
    float4 total_force = {0.0f, 0.0f, 0.0f, 0.0f};
    
    if (gid < n) {
        my_pos = positions[gid];
    }
    
    // Process particles in groups of 32 (warp size)
    for (int i = 0; i < n; i += 32) {
        float4 other_pos;
        
        // Each thread loads one particle
        if (i + lane_id < n) {
            other_pos = positions[i + lane_id];
        } else {
            other_pos = {0.0f, 0.0f, 0.0f, 0.0f};
        }
        
        // Calculate forces with all 32 particles using warp shuffle
        for (int j = 0; j < 32; j++) {
            float4 target_pos;
            target_pos.x = __shfl_sync(0xffffffffffffffffULL, other_pos.x, j);
            target_pos.y = __shfl_sync(0xffffffffffffffffULL, other_pos.y, j);
            target_pos.z = __shfl_sync(0xffffffffffffffffULL, other_pos.z, j);
            target_pos.w = __shfl_sync(0xffffffffffffffffULL, other_pos.w, j);
            
            if (i + j < n && gid != i + j) {
                // Calculate distance
                float dx = target_pos.x - my_pos.x;
                float dy = target_pos.y - my_pos.y;
                float dz = target_pos.z - my_pos.z;
                
                float dist_sq = dx*dx + dy*dy + dz*dz + SOFTENING*SOFTENING;
                float inv_dist = rsqrtf(dist_sq);
                float inv_dist3 = inv_dist * inv_dist * inv_dist;
                
                float force_mag = G_CONSTANT * my_pos.w * target_pos.w * inv_dist3;
                
                total_force.x += force_mag * dx;
                total_force.y += force_mag * dy;
                total_force.z += force_mag * dz;
                total_force.w -= G_CONSTANT * my_pos.w * target_pos.w * inv_dist;
            }
        }
    }
    
    if (gid < n) {
        forces[gid] = total_force;
    }
}

// Barnes-Hut tree node structure
struct TreeNode {
    float4 center_of_mass;  // x, y, z, total_mass
    float4 bounds_min;      // min x, y, z, padding
    float4 bounds_max;      // max x, y, z, padding
    int children[8];        // octree children indices
    int particle_count;
    int is_leaf;
};

// Barnes-Hut tree construction kernel
__global__ void buildOctree(
    float4* positions,
    TreeNode* nodes,
    int* node_assignments,
    int n,
    int max_nodes,
    float4 root_bounds_min,
    float4 root_bounds_max
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= n) return;
    
    float4 pos = positions[tid];
    int current_node = 0;  // Start at root
    
    // Traverse tree to find appropriate leaf
    while (!nodes[current_node].is_leaf && current_node < max_nodes) {
        // Determine which octant this particle belongs to
        float4 center = nodes[current_node].center_of_mass;
        int octant = 0;
        
        if (pos.x > center.x) octant |= 1;
        if (pos.y > center.y) octant |= 2;
        if (pos.z > center.z) octant |= 4;
        
        int child_node = nodes[current_node].children[octant];
        if (child_node == -1) {
            // Create new child node
            child_node = atomicAdd(&nodes[0].particle_count, 1);  // Use as node counter
            if (child_node < max_nodes) {
                nodes[current_node].children[octant] = child_node;
                
                // Initialize new node bounds
                float4 parent_min = nodes[current_node].bounds_min;
                float4 parent_max = nodes[current_node].bounds_max;
                float4 parent_center = nodes[current_node].center_of_mass;
                
                nodes[child_node].bounds_min = {
                    (octant & 1) ? parent_center.x : parent_min.x,
                    (octant & 2) ? parent_center.y : parent_min.y,
                    (octant & 4) ? parent_center.z : parent_min.z,
                    0.0f
                };
                
                nodes[child_node].bounds_max = {
                    (octant & 1) ? parent_max.x : parent_center.x,
                    (octant & 2) ? parent_max.y : parent_center.y,
                    (octant & 4) ? parent_max.z : parent_center.z,
                    0.0f
                };
                
                nodes[child_node].is_leaf = 1;
                nodes[child_node].particle_count = 0;
                
                for (int i = 0; i < 8; i++) {
                    nodes[child_node].children[i] = -1;
                }
            }
        }
        current_node = child_node;
    }
    
    // Assign particle to leaf node
    if (current_node < max_nodes) {
        node_assignments[tid] = current_node;
        atomicAdd(&nodes[current_node].particle_count, 1);
        
        // Update center of mass (simplified - would need proper atomic operations)
        atomicAdd(&nodes[current_node].center_of_mass.x, pos.x * pos.w);
        atomicAdd(&nodes[current_node].center_of_mass.y, pos.y * pos.w);
        atomicAdd(&nodes[current_node].center_of_mass.z, pos.z * pos.w);
        atomicAdd(&nodes[current_node].center_of_mass.w, pos.w);
    }
}

// Barnes-Hut force calculation with tree traversal
__global__ void calculateForces_BarnesHut(
    float4* positions,
    float4* forces,
    TreeNode* tree_nodes,
    int n,
    float theta  // Opening angle parameter
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (gid >= n) return;
    
    float4 my_pos = positions[gid];
    float4 total_force = {0.0f, 0.0f, 0.0f, 0.0f};
    
    // Stack for tree traversal
    int stack[64];
    int stack_size = 1;
    stack[0] = 0;  // Start with root
    
    while (stack_size > 0) {
        int node_idx = stack[--stack_size];
        TreeNode node = tree_nodes[node_idx];
        
        if (node.particle_count == 0) continue;
        
        // Calculate distance to center of mass
        float dx = node.center_of_mass.x - my_pos.x;
        float dy = node.center_of_mass.y - my_pos.y;
        float dz = node.center_of_mass.z - my_pos.z;
        float dist = sqrtf(dx*dx + dy*dy + dz*dz);
        
        // Calculate node size
        float node_size = node.bounds_max.x - node.bounds_min.x;
        
        // Check opening criterion
        if (node.is_leaf || (node_size / dist) < theta) {
            // Use this node's center of mass
            if (dist > SOFTENING) {
                float inv_dist = 1.0f / dist;
                float inv_dist3 = inv_dist * inv_dist * inv_dist;
                float force_mag = G_CONSTANT * my_pos.w * node.center_of_mass.w * inv_dist3;
                
                total_force.x += force_mag * dx;
                total_force.y += force_mag * dy;
                total_force.z += force_mag * dz;
                total_force.w -= G_CONSTANT * my_pos.w * node.center_of_mass.w * inv_dist;
            }
        } else {
            // Add children to stack for further traversal
            for (int i = 0; i < 8 && stack_size < 63; i++) {
                if (node.children[i] != -1) {
                    stack[stack_size++] = node.children[i];
                }
            }
        }
    }
    
    forces[gid] = total_force;
}

// Leapfrog integration kernel
__global__ void updateParticles(
    float4* positions,
    float4* velocities,
    float4* forces,
    int n,
    float dt
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (gid >= n) return;
    
    float4 pos = positions[gid];
    float4 vel = velocities[gid];
    float4 force = forces[gid];
    
    // Calculate acceleration
    float inv_mass = 1.0f / pos.w;
    float4 accel = {
        force.x * inv_mass,
        force.y * inv_mass,
        force.z * inv_mass,
        0.0f
    };
    
    // Leapfrog integration
    vel.x += accel.x * dt;
    vel.y += accel.y * dt;
    vel.z += accel.z * dt;
    
    pos.x += vel.x * dt;
    pos.y += vel.y * dt;
    pos.z += vel.z * dt;
    
    positions[gid] = pos;
    velocities[gid] = vel;
}

// Advanced collision detection and response
__global__ void handleCollisions(
    float4* positions,
    float4* velocities,
    int n,
    float collision_radius,
    float restitution
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (gid >= n) return;
    
    float4 my_pos = positions[gid];
    float4 my_vel = velocities[gid];
    
    // Check collisions with other particles
    for (int i = 0; i < n; i++) {
        if (i == gid) continue;
        
        float4 other_pos = positions[i];
        float4 other_vel = velocities[i];
        
        float dx = other_pos.x - my_pos.x;
        float dy = other_pos.y - my_pos.y;
        float dz = other_pos.z - my_pos.z;
        float dist = sqrtf(dx*dx + dy*dy + dz*dz);
        
        if (dist < collision_radius && dist > 0.0f) {
            // Collision detected - elastic collision response
            float inv_dist = 1.0f / dist;
            float nx = dx * inv_dist;
            float ny = dy * inv_dist;
            float nz = dz * inv_dist;
            
            // Relative velocity
            float vrel_x = other_vel.x - my_vel.x;
            float vrel_y = other_vel.y - my_vel.y;
            float vrel_z = other_vel.z - my_vel.z;
            
            // Relative velocity in collision normal direction
            float vrel_normal = vrel_x * nx + vrel_y * ny + vrel_z * nz;
            
            if (vrel_normal > 0.0f) {  // Objects approaching
                // Calculate impulse
                float total_mass = my_pos.w + other_pos.w;
                float impulse = 2.0f * vrel_normal / total_mass * restitution;
                
                // Apply impulse
                my_vel.x += impulse * other_pos.w * nx;
                my_vel.y += impulse * other_pos.w * ny;
                my_vel.z += impulse * other_pos.w * nz;
                
                velocities[gid] = my_vel;
            }
        }
    }
}

// Wrapper functions for launching kernels from C++ files
extern "C" {
    void launchCalculateForcesOptimized(float4* positions, float4* forces, int n, float softening, int blockSize, int gridSize) {
        calculateForces_optimized<<<gridSize, blockSize, blockSize * sizeof(float4)>>>(positions, forces, n, softening);
    }
    
    void launchCalculateForcesWarpOptimized(float4* positions, float4* forces, int n, int blockSize, int gridSize) {
        calculateForces_warp_optimized<<<gridSize, blockSize, blockSize * sizeof(float4)>>>(positions, forces, n);
    }
    
    void launchBuildOctree(float4* positions, TreeNode* nodes, int* node_assignments, int n, int max_nodes, float4 root_bounds_min, float4 root_bounds_max, int blockSize, int gridSize) {
        buildOctree<<<gridSize, blockSize>>>(positions, nodes, node_assignments, n, max_nodes, root_bounds_min, root_bounds_max);
    }
    
    void launchCalculateForcesBarnesHut(float4* positions, float4* forces, TreeNode* tree_nodes, int n, float theta, int blockSize, int gridSize) {
        calculateForces_BarnesHut<<<gridSize, blockSize>>>(positions, forces, tree_nodes, n, theta);
    }
    
    void launchUpdateParticles(float4* positions, float4* velocities, float4* forces, int n, float dt, int blockSize, int gridSize) {
        updateParticles<<<gridSize, blockSize>>>(positions, velocities, forces, n, dt);
    }
    
    void launchHandleCollisions(float4* positions, float4* velocities, int n, float collision_radius, float restitution, int blockSize, int gridSize) {
        handleCollisions<<<gridSize, blockSize>>>(positions, velocities, n, collision_radius, restitution);
    }
}

// Simple test main for standalone execution
int main() {
    const int n = 1024;
    float4* d_positions;
    float4* d_forces;
    hipError_t err;
    
    // Allocate device memory
    err = hipMalloc(&d_positions, n * sizeof(float4));
    if (err != hipSuccess) {
        printf("Error: hipMalloc failed for d_positions: %s\n", hipGetErrorString(err));
        return 1;
    }
    
    err = hipMalloc(&d_forces, n * sizeof(float4));
    if (err != hipSuccess) {
        printf("Error: hipMalloc failed for d_forces: %s\n", hipGetErrorString(err));
        hipFree(d_positions);
        return 1;
    }
    
    // Fill positions with dummy data
    float4* h_positions = new float4[n];
    for (int i = 0; i < n; ++i) h_positions[i] = {float(i), float(i%100), float(i%10), 1.0f};
    
    err = hipMemcpy(d_positions, h_positions, n * sizeof(float4), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        printf("Error: hipMemcpy H2D failed: %s\n", hipGetErrorString(err));
        hipFree(d_positions);
        hipFree(d_forces);
        delete[] h_positions;
        return 1;
    }
    
    // Launch kernel
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    launchCalculateForcesOptimized(d_positions, d_forces, n, 0.1f, blockSize, gridSize);
    
    // Copy result back
    float4* h_forces = new float4[n];
    err = hipMemcpy(h_forces, d_forces, n * sizeof(float4), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        printf("Error: hipMemcpy D2H failed: %s\n", hipGetErrorString(err));
    }
    
    // Calculate checksum
    float checksum = 0.0f;
    for (int i = 0; i < n; ++i) {
        checksum += h_forces[i].x + h_forces[i].y + h_forces[i].z + h_forces[i].w;
    }
    printf("N-body simulation result checksum: %f\n", checksum);
    
    // Cleanup
    err = hipFree(d_positions);
    if (err != hipSuccess) {
        printf("Warning: hipFree d_positions failed: %s\n", hipGetErrorString(err));
    }
    
    err = hipFree(d_forces);
    if (err != hipSuccess) {
        printf("Warning: hipFree d_forces failed: %s\n", hipGetErrorString(err));
    }
    
    delete[] h_positions;
    delete[] h_forces;
    return 0;
}
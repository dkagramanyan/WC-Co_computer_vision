#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <queue>
#include <climits>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <sys/resource.h>  // For getrusage() to measure peak CPU RAM usage
#include <filesystem> // C++17 for std::filesystem::path

// Uncomment for debug prints
// #define DEBUG_PRINT

// For GPU error-checking convenience
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl;                  \
            exit(EXIT_FAILURE);                                                 \
        }                                                                        \
    } while(0)

// -----------------------------------------------------------------------------
// Global variables to track peak GPU VRAM usage
// -----------------------------------------------------------------------------
static size_t g_totalGPUMem = 0;         // total GPU memory (constant per device)
static size_t g_minFreeMem = SIZE_MAX;   // smallest "free memory" encountered
static bool   g_memInitialized = false;

// A small utility function to update the global minimum free GPU memory
// after each allocation/free. The difference (total - g_minFreeMem) is
// effectively the maximum usage we've seen.
void updateVRAMUsage() {
    // Ensure all operations have completed
    cudaDeviceSynchronize();
    // Query current free/total memory
    size_t freeMem = 0, totalMem = 0;
    cudaMemGetInfo(&freeMem, &totalMem);
    // Initialize our global total memory (once only)
    if (!g_memInitialized) {
        g_totalGPUMem = totalMem;
        g_memInitialized = true;
    }
    // Track the minimum free memory
    if (freeMem < g_minFreeMem) {
        g_minFreeMem = freeMem;
    }
}

// We define custom wrappers for cudaMalloc/cudaFree
// so that each time we allocate/free, we update usage info.
template <typename T>
inline void myCudaMalloc(T** ptr, size_t size) {
    CUDA_CHECK(cudaMalloc((void**)ptr, size));
    updateVRAMUsage();
}

inline void myCudaFree(void* ptr) {
    CUDA_CHECK(cudaFree(ptr));
    updateVRAMUsage();
}

// -----------------------------------------------------------------------------
// Graph Structure (adjacency list in CSR-like form)
// -----------------------------------------------------------------------------
struct Graph {
    int numVertices;
    int numEdges;
    std::vector<int> adjacencyList; // All edges flattened
    std::vector<int> edgesOffset;   // Where each vertex's adjacency starts
    std::vector<int> edgesSize;     // How many neighbors each vertex has
};

// -----------------------------------------------------------------------------
// Load the graph from an edge list file: each line "src dst"
// -----------------------------------------------------------------------------
void loadGraph(const std::string &filename, Graph &G)
{
    std::ifstream in(filename);
    if (!in.is_open()) {
        std::cerr << "Could not open file: " << filename << "\n";
        exit(EXIT_FAILURE);
    }

    std::vector<std::pair<int,int>> edges;
    int maxNodeId = -1;
    {
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            std::stringstream ss(line);
            int s, t;
            ss >> s >> t;
            edges.push_back({s, t});
            maxNodeId = std::max(maxNodeId, std::max(s, t));
        }
    }
    in.close();

    G.numVertices = maxNodeId + 1;
    G.numEdges    = (int)edges.size();

    G.edgesOffset.resize(G.numVertices, 0);
    G.edgesSize  .resize(G.numVertices, 0);

    // Count outdegree for each vertex
    for (auto &e : edges) {
        int s = e.first;
        G.edgesSize[s]++;
    }

    // Compute prefix sums for edgesOffset
    for (int i = 1; i < G.numVertices; i++) {
        G.edgesOffset[i] = G.edgesOffset[i-1] + G.edgesSize[i-1];
    }

    // Make a copy of offset as a "fill pointer"
    std::vector<int> fillPtr = G.edgesOffset;
    G.adjacencyList.resize(G.numEdges);

    // Fill adjacencyList
    for (auto &e : edges) {
        int s = e.first;
        int t = e.second;
        int pos = fillPtr[s]++;
        G.adjacencyList[pos] = t;
    }
}

// -----------------------------------------------------------------------------
// CUDA kernel: given a set of vertices in frontNodes, look up their neighbors
// in parallel. We'll collect all neighbors in a big array. We also store
// how many neighbors each vertex has in outDegrees[i], so that the CPU can know
// how to slice the neighbor array correctly later.
// -----------------------------------------------------------------------------
__global__
void kernel_expand_front(int nFront,
                         const int *frontNodes,
                         const int *adjList,
                         const int *offsets,
                         const int *sizes,
                         int *outDegrees,
                         int *neighbors)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nFront) return;

    int node = frontNodes[i];
    int start = offsets[node];
    int sz    = sizes[node];

    // Write the out-degree
    outDegrees[i] = sz;

    // Copy adjacency into the output neighbors array
    for (int j = 0; j < sz; j++) {
        neighbors[i + j * nFront] = adjList[start + j];
    }
}

// -----------------------------------------------------------------------------
// A "PathItem" holds:
//   - path: the sequence of visited nodes (so far)
//   - visited: a boolean array to mark which nodes were already visited
//
// We do BFS in "waves" where each wave is a collection of PathItem's.
// -----------------------------------------------------------------------------
struct PathItem {
    std::vector<int> path;
    std::vector<bool> visited;  // visited[v] = true if 'v' is already in path

    PathItem(int n) : visited(n, false) {}
};

// -----------------------------------------------------------------------------
// BFS-like enumeration of all simple paths from start->end using adjacency
// expansions on the GPU. We store all partial paths in a queue on the CPU.
//
// If your graph has cycles or is large, this can blow up in memory/time!!
// -----------------------------------------------------------------------------
void findAllPathsBFS_GPU(const Graph &G, int start, int end, 
                         std::vector<std::vector<int>> &allPaths)
{
    // Initial queue: one path containing [start]
    std::vector<PathItem> queue;
    {
        PathItem item(G.numVertices);
        item.path.push_back(start);
        item.visited[start] = true;
        queue.push_back(std::move(item));
    }

    // Allocate GPU adjacency
    int *d_adjList   = nullptr;
    int *d_offsets   = nullptr;
    int *d_sizes     = nullptr;

    // Use our custom wrappers to track memory usage
    myCudaMalloc(&d_adjList,   G.adjacencyList.size() * sizeof(int));
    myCudaMalloc(&d_offsets,   G.numVertices         * sizeof(int));
    myCudaMalloc(&d_sizes,     G.numVertices         * sizeof(int));

    CUDA_CHECK(cudaMemcpy(d_adjList, G.adjacencyList.data(),
                          G.adjacencyList.size()*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offsets, G.edgesOffset.data(),
                          G.numVertices*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sizes,   G.edgesSize.data(),
                          G.numVertices*sizeof(int), cudaMemcpyHostToDevice));

    // We'll expand BFS in "waves" until no more expansions are possible.
    // But we store *all* partial paths that haven't yet reached 'end'.
    while (true) {
        if (queue.empty()) {
            break; // no more partial paths to expand
        }

        // 1) Collect the last node of each path in queue[] into frontNodes[]
        const int nFront = (int)queue.size();
        std::vector<int> frontNodes(nFront);
        for (int i = 0; i < nFront; i++) {
            frontNodes[i] = queue[i].path.back(); 
        }

        // 2) Copy frontNodes[] to GPU
        int *d_frontNodes = nullptr;
        myCudaMalloc(&d_frontNodes, nFront * sizeof(int));
        CUDA_CHECK(cudaMemcpy(d_frontNodes, frontNodes.data(),
                              nFront*sizeof(int), cudaMemcpyHostToDevice));

        // 3) For each of the nFront nodes, we have edgesSize[node] neighbors.
        //    We want to store the total neighbors in outDegrees[] so we can
        //    figure out how to slice them on the CPU side.
        int *d_outDegrees = nullptr;
        myCudaMalloc(&d_outDegrees, nFront * sizeof(int));

        // 4) We'll store neighbors in a 2D layout [i + j*nFront].
        //    First find the maxOutDegree among these frontNodes.
        int maxOutDegree = 0;
        for (int fn : frontNodes) {
            maxOutDegree = std::max(maxOutDegree, G.edgesSize[fn]);
        }

        // 5) Allocate device array for neighbors
        int *d_neighbors = nullptr;
        if (maxOutDegree > 0) {
            myCudaMalloc(&d_neighbors, nFront * maxOutDegree * sizeof(int));
        }

        // 6) Launch kernel to fill outDegrees[] and neighbors[]
        {
            int blockSize = 128;
            int gridSize  = (nFront + blockSize - 1) / blockSize;
            kernel_expand_front<<<gridSize, blockSize>>>(
                nFront, d_frontNodes, d_adjList, d_offsets, d_sizes,
                d_outDegrees, d_neighbors
            );
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // 7) Copy outDegrees + neighbors back to CPU
        std::vector<int> outDegrees(nFront);
        if (maxOutDegree > 0) {
            std::vector<int> neighborsCPU(nFront * maxOutDegree);
            CUDA_CHECK(cudaMemcpy(outDegrees.data(), d_outDegrees, 
                                  nFront*sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(neighborsCPU.data(), d_neighbors,
                                  nFront*maxOutDegree*sizeof(int), cudaMemcpyDeviceToHost));

            // 8) Build the next wave of partial paths by expanding
            //    each path in the queue with all its neighbors
            std::vector<PathItem> nextQueue;
            nextQueue.reserve(nFront * 2); // rough guess

            for (int i = 0; i < nFront; i++) {
                const PathItem &pitem = queue[i];
                int deg = outDegrees[i];
                for (int j = 0; j < deg; j++) {
                    int nbr = neighborsCPU[i + j*nFront];
                    // skip if already visited
                    if (pitem.visited[nbr]) {
                        continue;
                    }
                    // create a new path
                    PathItem newItem = pitem; // copy
                    newItem.path.push_back(nbr);
                    newItem.visited[nbr] = true;

                    // if we reached 'end', store it
                    if (nbr == end) {
                        allPaths.push_back(newItem.path);
                    } else {
                        nextQueue.push_back(std::move(newItem));
                    }
                }
            }
            queue.swap(nextQueue);
        }
        else {
            queue.clear();
        }

        // Clean up this layer
        myCudaFree(d_frontNodes);
        myCudaFree(d_outDegrees);
        if (maxOutDegree > 0) {
            myCudaFree(d_neighbors);
        }
    }

    // Clean up adjacency on device
    myCudaFree(d_adjList);
    myCudaFree(d_offsets);
    myCudaFree(d_sizes);
}

// -----------------------------------------------------------------------------
// Save all paths to a text file
// -----------------------------------------------------------------------------
void saveAllPaths(const std::vector<std::vector<int>> &allPaths,
                  const std::string &outFilename)
{
    std::ofstream out(outFilename);
    if (!out.is_open()) {
        std::cerr << "Cannot open output file: " << outFilename << "\n";
        return;
    }
    for (size_t i = 0; i < allPaths.size(); i++) {
        for (size_t j = 0; j < allPaths[i].size(); j++) {
            out << allPaths[i][j];
            if (j+1 < allPaths[i].size()) {
                out << " -> ";
            }
        }
        out << "\n";
    }
    out.close();
    std::cout << "Saved " << allPaths.size() << " paths to " << outFilename << "\n";
}

// -----------------------------------------------------------------------------
// Main
// Usage: ./bfs_paths edges.txt start end
// -----------------------------------------------------------------------------
int main(int argc, char** argv)
{
    // -------------------------------------------------------------------------
    // 1) Start timing
    // -------------------------------------------------------------------------
    auto tStart = std::chrono::high_resolution_clock::now();

    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " edges.txt start end\n";
        return 1;
    }
    std::string filename = argv[1];
    int start = std::stoi(argv[2]);
    int end   = std::stoi(argv[3]);

    // -------------------------------------------------------------------------
    // 2) Load graph
    // -------------------------------------------------------------------------
    Graph G;
    loadGraph(filename, G);

    std::cout << "Graph loaded with " << G.numVertices 
              << " vertices and " << G.numEdges << " edges.\n";

    if (start < 0 || start >= G.numVertices ||
        end   < 0 || end   >= G.numVertices)
    {
        std::cerr << "Start or end node out of range [0.."
                  << (G.numVertices - 1) << "]\n";
        return 2;
    }

    // -------------------------------------------------------------------------
    // 3) Enumerate *all possible simple paths* from start -> end
    // -------------------------------------------------------------------------
    std::vector<std::vector<int>> allPaths;
    findAllPathsBFS_GPU(G, start, end, allPaths);

    // -------------------------------------------------------------------------
    // 4) Save results to a text file
    // -------------------------------------------------------------------------
    std::ostringstream outFile;
    std::string baseFilename = std::filesystem::path(filename).stem().string();
    outFile << "bfs_paths_" << start << "_" << end <<"_" << baseFilename << ".txt";

    saveAllPaths(allPaths, outFile.str());
    std::cout << "Total paths found: " << allPaths.size() << "\n";

    // -------------------------------------------------------------------------
    // 5) Stop timing + get CPU peak RAM usage
    // -------------------------------------------------------------------------
    auto tEnd = std::chrono::high_resolution_clock::now();
    double elapsedSec = std::chrono::duration<double>(tEnd - tStart).count();

    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    // Peak resident set size in kilobytes on Linux/Unix
    long peakRamKB = usage.ru_maxrss;

    // -------------------------------------------------------------------------
    // 6) Compute peak GPU usage
    // -------------------------------------------------------------------------
    // g_minFreeMem is the minimum "free memory" seen
    // g_totalGPUMem is the total GPU memory we recorded
    // So peak usage is (total - minFree).
    size_t peakGpuBytes = 0;
    if (g_memInitialized) {
        peakGpuBytes = g_totalGPUMem - g_minFreeMem;
    }

    // -------------------------------------------------------------------------
    // 7) Write statistics to a file
    // -------------------------------------------------------------------------
    {   
        std::ostringstream stats_filename;
        stats_filename << "bfs_stats_" << start << "_" << end << "_" << baseFilename << ".csv"; 
    
        std::ofstream statsOut(stats_filename.str());
        if (!statsOut.is_open()) {
            std::cerr << "Cannot open " << stats_filename.str() << " for writing.\n";
        } else {
            // Write CSV headers
            statsOut << "Execution Time (seconds),Peak CPU RAM (MB),Peak GPU VRAM (MB)\n";
            
            // Write data values
            statsOut << elapsedSec << "," 
                     << (peakRamKB / 1024) << "," 
                     << (peakGpuBytes / (1024 * 1024)) << "\n";
            
            statsOut.close();
            std::cout << "Saved execution stats to " << stats_filename.str() << "\n";
        }
    }


    return 0;
}

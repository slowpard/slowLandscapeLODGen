
import numpy as np
from numba import prange, njit, types
from numba.typed import Dict
from numba.typed import List


''''GEOMETRY FUNCTIONS'''
@njit(cache=True)
def calculate_derivative_maps(height_map, step = 128):
    f_x = np.zeros_like(height_map, dtype=np.float32)
    f_y = np.zeros_like(height_map, dtype=np.float32)
    f_x[:, 1:-1] = (height_map[:, 2:] - height_map[:, :-2]) / 2 / step
    f_y[1:-1, :] = (height_map[2:, :] - height_map[:-2, :]) / 2 / step
    f_x[:, 0] = (height_map[:, 1] - height_map[:, 0]) / step
    f_x[:, -1] = (height_map[:, -1] - height_map[:, -2]) / step
    f_y[0, :] = (height_map[1, :] - height_map[0, :]) / step
    f_y[-1, :] = (height_map[-1, :] - height_map[-2, :]) / step

    f_xx = np.zeros_like(height_map, dtype=np.float32)
    f_yy = np.zeros_like(height_map, dtype=np.float32)
    f_xy = np.zeros_like(height_map, dtype=np.float32)
    f_xx[:, 1:-1] = (height_map[:, 2:] - 2 * height_map[:, 1:-1] + height_map[:, :-2]) / step / step
    f_yy[1:-1, :] = (height_map[2:, :] - 2 * height_map[1:-1, :] + height_map[:-2, :]) / step / step
    f_xy[1:-1, 1:-1] = (height_map[2:, 2:] - height_map[2:, :-2] - height_map[:-2, 2:] + height_map[:-2, :-2]) / 4 / step / step

    return f_x, f_y, f_xx, f_yy, f_xy

@njit(cache=True)
def calculate_max_eigenvalue(f_xx, f_yy, f_xy):

    '''
    Math refresh
    det(A-lambda*I) = 0
    A - lambda * I = 
    | f_xx - lambda   f_xy        |
    | f_xy           f_yy - lambda |

    det(A-lambda*I) = (f_xx - lambda) * (f_yy - lambda) - f_xy ^ 2 = 0
    lambda^2 - (f_xx + f_yy) * lambda + (f_xx * f_yy - f_xy^2) = 0
    lamda = ((f_xx + f_yy) +/- sqrt((f_xx + f_yy)^2 - 4(f_xx * f_yy - f_xy^2)) / 2
    '''
    trace = f_xx + f_yy
    determinant =  np.sqrt(trace**2 - 4 * (f_xx * f_yy - f_xy**2))
    lambda_plus = 0.5 * (trace + determinant)
    lambda_minus = 0.5 * (trace - determinant)

    return np.maximum(np.abs(lambda_plus), np.abs(lambda_minus))

@njit(cache=True)
def calculate_gradient_magnitude(f_x, f_y):
    return np.sqrt(f_x**2 + f_y**2)

@njit(cache=True)
def apply_z_weighting(height_map):
    z_map = np.zeros_like(height_map, dtype=np.float32)
    z_map = z_map + 1

    '''
    z_map[height_map < -500] *=  0.02
    z_map[(height_map >= -300) & (height_map < 400)] *=  10
    z_map[(height_map >= -100) & (height_map < 100)] *=  20
   '''
    
    for i in range(height_map.shape[0]):
        for j in range(height_map.shape[1]):
            if height_map[i, j] < -500:
                z_map[i, j] *= 0.1
            elif height_map[i, j] >= -300 and height_map[i, j] < 400:
                z_map[i, j] *= 10
            if height_map[i, j] >= -100 and height_map[i, j] < 100:
                z_map[i, j] *= 20

    #z_map[32+1, : ] *= 10
    #z_map[-33+1, : ] *= 10
    #z_map[, 32+1] *= 10
    #z_map[, -33+1] *= 10

    z_map[::32, :] *= 10
    z_map[:, ::32] *= 10

    z_map = np.minimum(z_map, 500)

    return z_map

@njit(cache=True)
def compute_plane_quadric(v0, v1, v2):
    #returns a 4x4 matrix that can be used to calculate the quadric error metric
    #error = vT * Q * v, Q is sum of quadric matrices for each vertex

    u = v1 - v0
    v = v2 - v0
    normal = np.cross(u, v)
    normal_length = np.linalg.norm(normal)
    if normal_length < 1e-12:
        #print('zero length normal')
        return np.zeros((4, 4), dtype=np.float64)
    normal /= normal_length
    #normal * v0 + d =0
    d = -np.dot(normal, v0)
    plane = np.array([normal[0], normal[1], normal[2], d], dtype=np.float64)
    
    return np.outer(plane, plane)

@njit(cache=True)
def generate_vert_quadratics(vertices, triangles):

    n_tri = len(triangles)
    #quadrics = np.zeros((n_tri, 4, 4), dtype=np.float64)
    vert_quadrics = np.zeros((len(vertices), 4, 4), dtype=np.float64)

    for i in prange(n_tri):

       
            
        v0 = vertices[triangles[i,0]]
        v1 = vertices[triangles[i,1]]
        v2 = vertices[triangles[i,2]]
        quadric = compute_plane_quadric(v0, v1, v2)

        #quadrics[i] = quadric
        vert_quadrics[triangles[i,0]] += quadric
        vert_quadrics[triangles[i,1]] += quadric
        vert_quadrics[triangles[i,2]] += quadric

    return vert_quadrics

@njit(cache=True)
def create_vertices_from_heightmap(heightmap, vertice_weights_data):

    rows, cols = 32*32 + 1, 32*32 + 1

    vertices = np.zeros((rows * cols, 3), dtype=np.float64)
    vertice_weights = np.zeros((rows * cols), dtype=np.float64)
    triangles = np.zeros(((rows - 1) * (cols - 1) * 2, 3), dtype=np.uint32)
    #boundary = np.full((rows * cols), False, dtype=bool)
    #boundary_chain = np.zeros(2*(cols + rows) - 4, dtype=np.uint32)
    horizontal_border = np.full((rows * cols), False, dtype=bool)
    vertical_border = np.full((rows * cols), False, dtype=bool)
    #h_chains = np.zeros(((rows // 32 + 1) * (cols - 1), 2), dtype=np.uint32)
    #v_chains = np.zeros(((cols // 32 + 1) * (rows - 1), 2), dtype=np.uint32)
    

    for i in range(rows):
        for j in range(cols):
            x = np.float64(128 * j) 
            y = np.float64(128 * i)
            z = np.float64(heightmap[i + 33 - 1, j + 33 - 1])
            vertices[i * cols + j] = (x, y, z)
            vertice_weights[i * cols + j] = np.float64(vertice_weights_data[i + 33 - 1, j + 33 - 1])

    #boundary_idx = 0

    #h_idx = 0
    #v_idx = 0
    for i in range(rows):
        for j in range(cols):
            if i % 32 == 0:
                horizontal_border[i * cols + j] = True
                #if j < cols - 1:
                #    h_chains[h_idx, 0] = i * cols + j
                #    h_chains[h_idx, 1] = i * cols + j + 1
                #    h_idx += 1
            if j % 32 == 0:
                vertical_border[i * cols + j] = True
                #if i < rows - 1:
                #    v_chains[v_idx, 0] = i * cols + j
                #    v_chains[v_idx, 1] = (i + 1) * cols + j
                #    v_idx += 1

            vertical_border[i * cols + j] = (j % 32 == 0)

    '''
    for j in range(cols):
        i = 0
        boundary[i * cols + j] = True
        boundary_chain[boundary_idx] = i * cols + j
    
        boundary_idx += 1

    for i in range(1, rows):
        j = cols - 1
        boundary[i * cols + j] = True
        boundary_chain[boundary_idx] = i * cols + j
        boundary_idx += 1

    for j in range(cols - 2, -1, -1):
        i = rows - 1
        boundary[i * cols + j] = True
        boundary_chain[boundary_idx] = i * cols + j
        boundary_idx += 1
    
    for i in range(rows - 2, 0, -1):
        j = 0
        boundary[i * cols + j] = True
        boundary_chain[boundary_idx] = i * cols + j
        boundary_idx += 1

   '''
    
    for i in range(rows - 1):
        for j in range(cols - 1):
            triangles[i * (cols - 1) * 2 + j * 2, 0] =  i * cols + j
            triangles[i * (cols - 1) * 2 + j * 2, 1] = i * cols + j + 1
            triangles[i * (cols - 1) * 2 + j * 2, 2] = (i + 1) * cols + j
            triangles[i * (cols - 1) * 2 + j * 2 + 1, 0] = (i + 1) * cols + j
            triangles[i * (cols - 1) * 2 + j * 2 + 1, 1] = i * cols + j + 1
            triangles[i * (cols - 1) * 2 + j * 2 + 1, 2] = (i + 1) * cols + j + 1

    n_edges = rows*(cols - 1) + (rows - 1)*cols + (rows - 1)*(cols - 1)
    edges = np.zeros((n_edges, 2), dtype=np.uint32)
    idx = 0

    for i in range(rows): 
        for j in range(cols - 1):
            edges[idx, 0] = i * cols + j
            edges[idx, 1] = i * cols + (j + 1)
            idx += 1

    for i in range(rows - 1):
        for j in range(cols):
            edges[idx, 0] = i * cols + j
            edges[idx, 1] = (i + 1) * cols + j
            idx += 1

    for i in range(rows - 1):
        for j in range(cols - 1):
            edges[idx, 0] = i * cols + j
            edges[idx, 1] = (i + 1) * cols + (j + 1)
            idx += 1

    return vertices, triangles, edges, vertice_weights, horizontal_border, vertical_border

@njit(fastmath=True, cache=True)
def inverse_3x3(matrix, det):
    inv = np.zeros((3, 3), dtype=np.float64)
    inv_det = 1 / det
    inv[0, 0] = (matrix[1, 1] * matrix[2, 2] - matrix[1, 2] * matrix[2, 1]) * inv_det
    inv[0, 1] = (matrix[0, 2] * matrix[2, 1] - matrix[0, 1] * matrix[2, 2]) * inv_det
    inv[0, 2] = (matrix[0, 1] * matrix[1, 2] - matrix[0, 2] * matrix[1, 1]) * inv_det
    inv[1, 0] = (matrix[1, 2] * matrix[2, 0] - matrix[1, 0] * matrix[2, 2]) * inv_det
    inv[1, 1] = (matrix[0, 0] * matrix[2, 2] - matrix[0, 2] * matrix[2, 0]) * inv_det
    inv[1, 2] = (matrix[0, 2] * matrix[1, 0] - matrix[0, 0] * matrix[1, 2]) * inv_det
    inv[2, 0] = (matrix[1, 0] * matrix[2, 1] - matrix[1, 1] * matrix[2, 0]) * inv_det
    inv[2, 1] = (matrix[0, 1] * matrix[2, 0] - matrix[0, 0] * matrix[2, 1]) * inv_det
    inv[2, 2] = (matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]) * inv_det
    return inv



FLOAT64_ZERO_ONE = np.array([0, 0, 0, 1], dtype=np.float64)
    
@njit(fastmath=True, cache=True)
def edge_cost_evaluation(v0, v1, q1, q2, w1, w2, v0_is_boundary, v1_is_boundary, v0_h, v1_h, v0_v, v1_v):
    
    #det_val2 = np.linalg.det(Q_bar[:3, :3])

    #print(det_val, det_val2)
    
    optimum = np.zeros(3, dtype=np.float64)
    high_cost_flag = False
    if abs(v0[0] - v1[0]) > 4095.9999 or abs(v0[1] - v1[1]) > 4095.9999:
        high_cost_flag = True
        optimum = (w1 / (w1 + w2) * v0 + w2 / (w1 + w2) * v1)
    elif v0_h:
        if v0_v:
            if v1_h and v1_v:
                high_cost_flag = True
                optimum = (w1 / (w1 + w2) * v0 + w2 / (w1 + w2) * v1)
            else:
                optimum = v0
                #optimum[2] = (w1 / (w1 + w2) * v0[2] + w2 / (w1 + w2) * v1[2])
        else:
            if v1_h and v1_v:
                optimum = v1
                #optimum[2] = (w1 / (w1 + w2) * v0[2] + w2 / (w1 + w2) * v1[2])
            elif v1_v:
                high_cost_flag = True
                optimum = (w1 / (w1 + w2) * v0 + w2 / (w1 + w2) * v1)
            elif v1_h:
                optimum = (w1 / (w1 + w2) * v0 + w2 / (w1 + w2) * v1)
            else:
                optimum = v0
                #optimum[2] = (w1 / (w1 + w2) * v0[2] + w2 / (w1 + w2) * v1[2])  
    elif v0_v:
        if v1_h and v1_v:
            optimum = v1
            #optimum[2] = (w1 / (w1 + w2) * v0[2] + w2 / (w1 + w2) * v1[2])
        elif v1_h:
            high_cost_flag = True
            optimum = (w1 / (w1 + w2) * v0 + w2 / (w1 + w2) * v1)
        elif v1_v:
            optimum = (w1 / (w1 + w2) * v0 + w2 / (w1 + w2) * v1)
        else:
            optimum = v0
            #optimum[2] = (w1 / (w1 + w2) * v0[2] + w2 / (w1 + w2) * v1[2])
    elif v1_h or v1_v:
        optimum = v1
        #optimum[2] = (w1 / (w1 + w2) * v0[2] + w2 / (w1 + w2) * v1[2])

    else:
        
        #this solves dE/dx = dE/dy = dE/dz = 0

        Q_bar = np.zeros((4, 4), dtype=np.float64)
        for i in range(3):
            for j in range(4):
                Q_bar[i, j] = w1 * q1[i, j] + w2 * q2[i, j]

        Q_bar[3, 0] = np.float64(0.0)
        Q_bar[3, 1] = np.float64(0.0)
        Q_bar[3, 2] = np.float64(0.0)
        Q_bar[3, 3] = np.float64(1.0)

        det_val = Q_bar[0, 0] * ( Q_bar[1, 1] * Q_bar[2, 2] - Q_bar[1, 2] * Q_bar[2, 1]) - \
                Q_bar[0, 1] * ( Q_bar[1, 0] * Q_bar[2, 2] - Q_bar[1, 2] * Q_bar[2, 0]) + \
                Q_bar[0, 2] * ( Q_bar[1, 0] * Q_bar[2, 1] - Q_bar[1, 1] * Q_bar[2, 0])
        
        if det_val < np.float64(1e-6):
            #optimum = (w1/ (w1 + w2) * v0 + w2/ (w1 + w2) * v1) * np.float64(0.5) 
            #technically should optimize along the line
            optimum = (w1 / (w1 + w2) * v0 + w2 / (w1 + w2) * v1) #* np.float64(0.5)

        else:
            

            #Q_bar_inv = np.linalg.inv(Q_bar)
            #optimum = np.array((Q_bar_inv[0, 3], Q_bar_inv[1, 3], Q_bar_inv[2, 3]))

            Q_bar_inv = inverse_3x3(Q_bar, det_val)
            optimum = np.array(((-Q_bar_inv[0,0] * Q_bar[0, 3] - Q_bar_inv[0,1] * Q_bar[1, 3] - Q_bar_inv[0,2] * Q_bar[2, 3]), 
                                (-Q_bar_inv[1,0] * Q_bar[0, 3] - Q_bar_inv[1,1] * Q_bar[1, 3] - Q_bar_inv[1,2] * Q_bar[2, 3]),
                                (-Q_bar_inv[2,0] * Q_bar[0, 3] - Q_bar_inv[2,1] * Q_bar[1, 3] - Q_bar_inv[2,2] * Q_bar[2, 3])))
            #print(optimum, optimum2)

            #optimum = np.linalg.solve(Q_bar, FLOAT64_ZERO_ONE)[:3]
            #AX = Z -> X = A^-1 * [0 0 0 1]

            for i in range(3): # Avoiding bonkers z values (numerical instability? idk wtf happens)
                lower = v0[i] if v0[i] < v1[i] else v1[i]
                upper = v0[i] if v0[i] > v1[i] else v1[i]
                if optimum[i] < lower or optimum[i] > upper:
                    for j in range(3):
                        #optimum[j] = (w1 / (w1 + w2) * v0[j] + w2 / (w1 + w2) * v1[j]) * 0.5
                        optimum[j] = (w1  * v0[j] + w2  * v1[j]) / (w1 + w2) #* np.float64(0.5)
                    break   
        
    #v_opt = np.array((optimum[0], optimum[1], optimum[2], 1.0), dtype=np.float64)
    
    if high_cost_flag:
        cost = np.float64(1e+100)
    else:
        cost = np.float64(0.0)
        #cost = v_opt.T @ Q_u @ v_opt
        
        Q_u = np.zeros((4, 4), dtype=np.float64)
        for i in range(4):
            for j in range(4):
                Q_u[i, j] = q1[i, j] + q2[i, j]

        ox, oy, oz = optimum[0], optimum[1], optimum[2]
        cost += Q_u[0, 0] * ox * ox + Q_u[1, 1] * oy * oy + Q_u[2, 2] * oz * oz
        cost += 2.0 * (Q_u[0, 1] * ox * oy + Q_u[0, 2] * ox * oz + Q_u[1, 2] * oy * oz)
        cost += 2.0 * (Q_u[0, 3] * ox + Q_u[1, 3] * oy + Q_u[2, 3] * oz)
        cost += Q_u[3, 3]

    return cost, optimum

@njit(cache=True)
def calculate_edge_cost(vertices_array, edges_array, vertice_quadrics_array, weights_array, horizontal_border, vertical_border):
    n_edges = len(edges_array)
    cost = np.zeros(n_edges, dtype=np.float64)
    optimum = np.zeros((n_edges, 3), dtype=np.float64)
    for i in range(n_edges):
        edge = edges_array[i]
        v0 = vertices_array[edge[0]]
        v1 = vertices_array[edge[1]]
        q1 = vertice_quadrics_array[edge[0]]
        q2 = vertice_quadrics_array[edge[1]]
        w1 = weights_array[edge[0]]
        w2 = weights_array[edge[1]]
        v0_is_boundary = False #boundary[edge[0]]
        v1_is_boundary = False #boundary[edge[1]]
        v0_h = horizontal_border[edge[0]]
        v1_h = horizontal_border[edge[1]]
        v0_v = vertical_border[edge[0]]
        v1_v = vertical_border[edge[1]]
        cost[i], optimum[i] = edge_cost_evaluation(v0, v1, q1, q2, w1, w2, v0_is_boundary, v1_is_boundary, v0_h, v1_h, v0_v, v1_v)
        cost[i] = cost[i] * (w1 + w2) / 2

    return cost, optimum

def is_positive_semidefinite(M, tol=1e-14):
    
    if not np.allclose(M, M.T, atol=tol):
        return False
    
    
    w, _ = np.linalg.eig(M)
    
    
    return np.all(w > -tol)


@njit(fastmath=True, cache=True)
def simplify_mesh(vertices_array, triangles_array, edges_array,
                  vertex_quadrics_array, edge_cost, edges_collapse, weights_array, horizontal_border, vertical_border, 
                  target_verts=30000, min_error=500000, first_loop_min_error=500, vertices_batch=0.8, current_vert_abs_minimum=1089):
    #QEM decimation with some very not math-strict adjustments https://www.cs.cmu.edu/~./garland/Papers/quadrics.pdf
    

    num_vertices = len(vertices_array)
    num_edges = len(edges_array)
    vertex_valid = np.full(num_vertices, True, dtype=bool)
    edge_valid = np.full(num_edges, True, dtype=bool)

    vertices_map = np.arange(num_vertices, dtype=np.uint32)

    edge_cost_old = edge_cost.copy()
    sorted_index = np.argsort(edge_cost)
    
    current_edges = num_edges #init

    loop_idx = 0

    
    current_vertex = len(vertices_array)
    recalc_barrier = np.int32(current_vertex * vertices_batch)
    

    index_len = len(sorted_index)
    last_verts = current_vertex

    #that's a bit clumsy
    #we need to store edges connected to each vertex to update them quickly
    #normally in python it would be a dict of lists, but we use numba
    #and creating millions of numba lists just hangs the kernel when it tries to garbage collect later
    #because merging verts also merges edges, we can just daisy-chain them instead
    max_edges_per_vert = 6
    connected_edges = np.full((num_vertices, max_edges_per_vert), -1, np.int32)
    chained_edges = np.full((num_vertices), -1, np.int32)

    for i in range(num_edges):
        for v in range(2):
            vertex = edges_array[i, v]
            for j in range(max_edges_per_vert):
                if connected_edges[vertex, j] == -1:
                    connected_edges[vertex, j] = i
                    break

    while (current_vertex > target_verts or edge_cost_old[sorted_index[loop_idx]] < min_error) and current_vertex > current_vert_abs_minimum:
        
        if (current_vertex < recalc_barrier and edge_cost_old[sorted_index[loop_idx]] > first_loop_min_error) or loop_idx >= index_len: 
                if (last_verts - current_vertex) < 100:
                    break
                #print("current verts:", current_vertex)
                valid_indices = np.where(edge_valid)[0]
                sorted_index = valid_indices[np.argsort(edge_cost[valid_indices])]
                index_len = len(sorted_index)
                #sorted_index = np.argsort(edge_cost)
                edge_cost_old = edge_cost.copy()
                #boundary_old = boundary.copy()
                loop_idx = 0
                recalc_barrier = np.int32(current_vertex * vertices_batch)
                last_verts = current_vertex
        
        e_idx = sorted_index[loop_idx]

        loop_idx += 1
        #print(loop_idx)



        if not edge_valid[e_idx]:
            continue

        if edge_cost[e_idx] > edge_cost_old[e_idx] and edge_cost[e_idx] > min_error: #> np.float64(1e-6): #cost was updated and became higher, wait till full sort
            continue

        v0, v1 = edges_array[e_idx]
       
        #if not (vertex_valid[v0] and vertex_valid[v1]):
        #    print("invalid edge")
        #    edge_valid[e_idx] = False
        #    continue

        current_edges -= 1
        
        edge_valid[e_idx] = False
        vertex_valid[v1] = False
        
        current_vertex -= 1
        
        vertex_quadrics_array[v0] = (weights_array[v0] * vertex_quadrics_array[v0] + weights_array[v1] * vertex_quadrics_array[v1] ) / (weights_array[v0] + weights_array[v1])
        
        #yep, that's the original intention of the method author
        #the idea is to combine the errors of both verts
        weights_array[v0] = (weights_array[v0] + weights_array[v1]) 


        vertices_array[v0] = edges_collapse[e_idx]

        vertices_map[v1] = v0

        if horizontal_border[v1]:
            horizontal_border[v1] = False
            horizontal_border[v0] = True

        if vertical_border[v1]:
            vertical_border[v1] = False
            vertical_border[v0] = True



        #connecting all edges of v1 to v0

        v_i = v0
        while chained_edges[v_i] != -1:
            v_i = chained_edges[v_i]
        
        chained_edges[v_i] = v1

        #updating all edges connected to v1 and v0

        edges_loop = True

        pointer = 0
        i_vertex = v0

        verts = set()
        
        while edges_loop:

            if pointer >= max_edges_per_vert:
                i_vertex = chained_edges[i_vertex]
                if i_vertex == -1:
                    edges_loop = False
                    break
                pointer = 0
                continue
            
            edge_idx = connected_edges[i_vertex, pointer]
            pointer += 1

            if edge_idx == -1:
                continue

            if not edge_valid[edge_idx]:
                continue

            ev0 = edges_array[edge_idx, 0]
            ev1 = edges_array[edge_idx, 1]

            if ev0 == v1:
                edges_array[edge_idx, 0] = v0
                ev0 = v0
            elif ev1 == v1:
                edges_array[edge_idx, 1] = v0
                ev1 = v0

            #degenerate
            if ev1 == ev0:
                edge_valid[edge_idx] = False
                continue   

            if ev0 != v0:
                v_other = ev0
            else:
                v_other = ev1
            
            if not v_other in verts:
                verts.add(v_other)
            else:
                edge_valid[edge_idx] = False
                continue

    

            updated_cost, updated_optimum = edge_cost_evaluation(vertices_array[ev0], vertices_array[ev1], vertex_quadrics_array[ev0], 
                                                                 vertex_quadrics_array[ev1], weights_array[ev0], weights_array[ev1], 
                                                                 False, False, horizontal_border[ev0], horizontal_border[ev1],  
                                                                 vertical_border[ev0], vertical_border[ev1]) 
            edge_cost[edge_idx] = updated_cost * (weights_array[ev0] + weights_array[ev1]) / 2
            edges_collapse[edge_idx][0] = updated_optimum[0]
            edges_collapse[edge_idx][1] = updated_optimum[1]
            edges_collapse[edge_idx][2] = updated_optimum[2]
            
           
    #print("FINISHED", sum(vertex_valid), current_vertex, "cost: ", edge_cost[edge_idx])

    #vertices_map[v1] = v0
    #for i in range(len(vertices_map)):
    #    n = i
    #    if vertices_map[n] != n:
    #        while vertices_map[n] != n:
    #            n = vertices_map[n]
    #        vertices_map[i] = n
    #for i in range(len(h_chains)):
    #    h_chains[i, 0] = vertices_map[h_chains[i, 0]]
    #    h_chains[i, 1] = vertices_map[h_chains[i, 1]]
    #for i in range(len(v_chains)):
    #    v_chains[i, 0] = vertices_map[v_chains[i, 0]]
    #    v_chains[i, 1] = vertices_map[v_chains[i, 1]]
    
    return vertex_valid

  
def save_obj(vertices,triangles, filepath):

    with open(filepath, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]:.3f} {v[2]:.3f} {v[1]:.3f} 1.0\n")

        if triangles is not None:
            for tri in triangles:
                f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")

'''helper functions'''


#0.0 and 131072.0
def UpdateCellBordersWrapper(mesh_data):

            )   


    for x_quad in  mesh_data:
        for y_quad in mesh_data[x_quad]:

            for cycle in range(2):
                adj_cell_exists = False
                if cycle == 0:
                    if x_quad + 1 in mesh_data:
                        if y_quad in mesh_data[x_quad + 1]: #east
                            adj_cell_exists = True

                elif cycle == 1:
                    if y_quad + 1 in mesh_data[x_quad]:
                        adj_cell_exists = True
                    
                if adj_cell_exists:
                    q1 = mesh_data[x_quad][y_quad][0].copy()
                    q1_border = mesh_data[x_quad][y_quad][2].copy()

                    if cycle == 0:
                        q2 = mesh_data[x_quad + 1][y_quad][0].copy()                
                        q2_border = mesh_data[x_quad + 1][y_quad][2].copy()
                        border_c_axis = 0
                        other_axis = 1
                    else:
                        q2 = mesh_data[x_quad][y_quad + 1][0].copy()
                        q2_border = mesh_data[x_quad][y_quad + 1][2].copy()
                        border_c_axis = 1
                        other_axis = 0
                    #print(x_quad, y_quad, 'updating borders cycle ', cycle)
                    #print(q1.shape, q2.shape)
                    #print("q1 border coords:", q1[np.abs(q1[:, border_c_axis] - 131072.0) < 0.001][:, other_axis])
                    #print("q2 border coords:", q2[np.abs(q2[:, border_c_axis] - 0.0) < 0.001][:, other_axis])
                    q1, q2, q1_addins, q2_addins, q1_remove, q2_remove = UpdateCellBorders(q1, q2, border_c_axis, other_axis)
                    #print('Added verts:', q1_addins.shape, q2_addins.shape)

                    q1_border = q1_border[~q1_remove]
                    q2_border = q2_border[~q2_remove]

                    mesh_data[x_quad][y_quad] = (
                        np.concatenate((q1, q1_addins), axis=0),
                        None,
                        np.concatenate((q1_border, np.full(len(q1_addins), True, dtype=bool)))
                    )

                    if cycle == 0:
                        mesh_data[x_quad + 1][y_quad] = (
                            np.concatenate((q2, q2_addins), axis=0),
                            None,
                            np.concatenate((q2_border, np.full(len(q2_addins), True, dtype=bool)))
                        )
                    else:
                        mesh_data[x_quad][y_quad + 1] = (
                            np.concatenate((q2, q2_addins), axis=0),
                            None,
                            np.concatenate((q2_border, np.full(len(q2_addins), True, dtype=bool)))
                        )
     

def UpdateCellBorders(q1, q2, border_c_axis, other_axis, merging_sensitivity=50, snap_sensitivity=10):
    border_coord = (131072.0, 0.0)
    
    q1_border = np.abs(q1[:, border_c_axis] - border_coord[0]) < 0.001
    q2_border = np.abs(q2[:, border_c_axis] - border_coord[1]) < 0.001
    
    q1_collapsable = q1_border & (q1[:, other_axis] % 4096.0 > 0.001)
    q2_collapsable = q2_border & (q2[:, other_axis] % 4096.0 > 0.001)
    
    q1_non_collapsable = q1_border & ~q1_collapsable
    q2_non_collapsable = q2_border & ~q2_collapsable
    nc1_coords = q1[q1_non_collapsable, other_axis]
    nc2_coords = q2[q2_non_collapsable, other_axis]
    grid_coords = np.unique(np.concatenate([nc1_coords, nc2_coords]))
    
    q1_remove = np.zeros(len(q1), dtype=bool)
    q2_remove = np.zeros(len(q2), dtype=bool)
    
    #first pass: remove collapsable vertices too close to grid intersections
    for qi in np.where(q1_collapsable)[0]:
        dists = np.abs(grid_coords - q1[qi, other_axis])
        if dists.min() < snap_sensitivity:
            q1_remove[qi] = True
    
    for qj in np.where(q2_collapsable)[0]:
        dists = np.abs(grid_coords - q2[qj, other_axis])
        if dists.min() < snap_sensitivity:
            q2_remove[qj] = True
    
    #second pass: group remaining collapsable vertices within merging_sensitivity,
    #merge each group to weighted average, remove extras (keep one per side)
    q1_remaining = q1_collapsable & ~q1_remove
    q2_remaining = q2_collapsable & ~q2_remove
    q1_coll_idx = np.where(q1_remaining)[0]
    q2_coll_idx = np.where(q2_remaining)[0]
    
    combined_coords = np.concatenate([q1[q1_coll_idx, other_axis], q2[q2_coll_idx, other_axis]])
    combined_z = np.concatenate([q1[q1_coll_idx, 2], q2[q2_coll_idx, 2]])
    combined_idx = np.concatenate([q1_coll_idx, q2_coll_idx])
    combined_sources = np.concatenate([np.zeros(len(q1_coll_idx), dtype=int), np.ones(len(q2_coll_idx), dtype=int)])
    
    order = np.argsort(combined_coords)
    combined_coords = combined_coords[order]
    combined_z = combined_z[order]
    combined_idx = combined_idx[order]
    combined_sources = combined_sources[order]
    
    matched_q1 = np.zeros(len(q1), dtype=bool)
    matched_q2 = np.zeros(len(q2), dtype=bool)
    
    i = 0
    while i < len(combined_idx):
        #collect group: all consecutive vertices within merging_sensitivity of the first
        group_start = i
        group_end = i + 1
        while group_end < len(combined_idx) and combined_coords[group_end] - combined_coords[group_start] < merging_sensitivity:
            group_end += 1
        
        has_q1 = np.any(combined_sources[group_start:group_end] == 0)
        has_q2 = np.any(combined_sources[group_start:group_end] == 1)
        
        if has_q1 and has_q2:
            #merge: compute average position and Z across the group
            avg_coord = combined_coords[group_start:group_end].mean()
            avg_z = combined_z[group_start:group_end].mean()
            
            #keep first from each side, remove the rest
            kept_q1 = False
            kept_q2 = False
            for k in range(group_start, group_end):
                idx = combined_idx[k]
                src = combined_sources[k]
                if src == 0 and not kept_q1:
                    q1[idx, other_axis] = avg_coord
                    q1[idx, 2] = avg_z
                    matched_q1[idx] = True
                    kept_q1 = True
                elif src == 1 and not kept_q2:
                    q2[idx, other_axis] = avg_coord
                    q2[idx, 2] = avg_z
                    matched_q2[idx] = True
                    kept_q2 = True
                elif src == 0:
                    q1_remove[idx] = True
                elif src == 1:
                    q2_remove[idx] = True
        else:
            #single-side group: keep only one, remove extras
            kept = False
            for k in range(group_start, group_end):
                if not kept:
                    kept = True
                elif combined_sources[k] == 0:
                    q1_remove[combined_idx[k]] = True
                else:
                    q2_remove[combined_idx[k]] = True
        
        i = group_end
    
    # Third pass: unmatched collapsable vertices get copied to the other quad
    q1_addins = []
    q2_addins = []
    
    q1_border_coords = q1[q1_border & ~q1_remove, other_axis]
    q2_border_coords = q2[q2_border & ~q2_remove, other_axis]
    
    for qj in np.where(q2_remaining & ~matched_q2 & ~q2_remove)[0]:
        coord = q2[qj, other_axis]
        if np.abs(q1_border_coords - coord).min() < snap_sensitivity:
            continue
        addin = q2[qj].copy()
        addin[border_c_axis] = border_coord[0]
        q1_addins.append(addin)
    
    for qi in np.where(q1_remaining & ~matched_q1 & ~q1_remove)[0]:
        coord = q1[qi, other_axis]
        if np.abs(q2_border_coords - coord).min() < snap_sensitivity:
            continue
        addin = q1[qi].copy()
        addin[border_c_axis] = border_coord[1]
        q2_addins.append(addin)
    
    q1 = q1[~q1_remove]
    q2 = q2[~q2_remove]
    q1_addins = np.array(q1_addins).reshape(-1, 3) if q1_addins else np.zeros((0, 3), dtype=np.float64)
    q2_addins = np.array(q2_addins).reshape(-1, 3) if q2_addins else np.zeros((0, 3), dtype=np.float64)
    
    return q1, q2, q1_addins, q2_addins, q1_remove, q2_remove


@njit(cache=True)
def GenerateLODMeshData(height_map, target_verts, min_error, vertices_batch):
    #logging.info('Generating LOD mesh...')
    #print(height_map.shape)
    f_x, f_y, f_xx, f_yy, f_xy = calculate_derivative_maps(height_map)
    gradient_magnitude = calculate_gradient_magnitude(f_x, f_y)
    max_eigenvalue = calculate_max_eigenvalue(f_xx, f_yy, f_xy) 
    weighting = apply_z_weighting(height_map)
  
    vertice_weights = (np.full((34*32 + 1, 34*32 + 1), 1, dtype=np.float32) + 1 * gradient_magnitude + 500 * np.minimum(max_eigenvalue, 0.02)) 
    vertice_weights = vertice_weights * weighting
    #z_weighted = apply_z_weighting(height_map)
    #logging.info('Derivatives calculated....')

    vertices_array, triangles_array, edges_array, weights_array, horizontal_border, vertical_border = create_vertices_from_heightmap(height_map, vertice_weights)

    #logging.info('Arrays calculated....')

    vertice_quadrics_array = generate_vert_quadratics(vertices_array, triangles_array)
    #logging.info('Quadratics calculated....')

    edges_cost, edges_collapse = calculate_edge_cost(vertices_array, edges_array, vertice_quadrics_array, weights_array, horizontal_border, vertical_border)
    #edge_data = np.column_stack((edges_cost, edges_array, np.arange(len(edges_cost))))
    #logging.info('Starting mesh decimation...')

    p10 = np.percentile(edges_cost, 10)
    p25 = np.percentile(edges_cost, 25)
    p50 = np.percentile(edges_cost, 50)
    p90 = np.percentile(edges_cost, 90)
    p95 = np.percentile(edges_cost, 95)
    #print("cost percentiles:", p10, p25, p50, p90, p95)
    
   
    vertex_valid = simplify_mesh(vertices_array, triangles_array, edges_array,
                    vertice_quadrics_array, edges_cost, edges_collapse, weights_array, horizontal_border, vertical_border,
                    target_verts = target_verts, min_error = min_error, vertices_batch = vertices_batch, current_vert_abs_minimum=1089)
    
    
    #vertices_array, vertex_valid, h_chains, v_chains = simplify_mesh(vertices_array, triangles_array, edges_array,
    #                vertice_quadrics_array, edges_cost, edges_collapse, weights_array, horizontal_border, vertical_border, h_chains, v_chains, target_verts=30000)
    pairs = np.zeros((3,2), dtype = np.int32)
    '''
    orig_indices = np.nonzero(vertex_valid)[0]
    index_map = {orig: new for new, orig in enumerate(orig_indices)}
    #pairs = np.column_stack((final_boundary_sequence, np.roll(final_boundary_sequence, -1)))

    

    for i in range(len(h_chains)):
        if h_chains[i, 0] > h_chains[i, 1]:
            h_chains[i, 0], h_chains[i, 1] = h_chains[i, 1], h_chains[i, 0]
    
    for i in range(len(v_chains)):
        if v_chains[i, 0] > v_chains[i, 1]:
            v_chains[i, 0], v_chains[i, 1] = v_chains[i, 1], v_chains[i, 0]

    pairs = np.vstack((np.unique(h_chains, axis=0), np.unique(v_chains, axis=0)))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    for i in range(len(pairs)):
        pairs[i, 0] = index_map[pairs[i, 0]]
        pairs[i, 1] = index_map[pairs[i, 1]] '''

    return vertices_array[vertex_valid.astype(np.bool_)], pairs, np.logical_or(horizontal_border, vertical_border)[vertex_valid.astype(np.bool_)]

    #tris = triangle.triangulate({'vertices':vertices_array[vertex_valid.astype(bool)][:, :2], 'segments':pairs}, 'p')
    #tris = triangle.triangulate({'vertices':vertices_array[vertex_valid.astype(bool)][:, :2], 'segments':pairs})
    #SaveGeometryIntoNif(vertices_array, tris['triangles'], quad, folder, file_name, vertex_valid)


    
@njit(parallel=True, cache=True)
def GenerateLODMeshDataWrapper(worldspace_heightmap, quads, x_low, y_low, counter, target_verts, min_error, vertices_batch):

    dummy_verts = np.empty((0, 3), dtype=np.float64)
    dummy_pairs = np.empty((0, 2), dtype=np.int32)
    dummy_bound = np.empty((0,), dtype=np.bool_)
    initial_tuple = (dummy_verts, dummy_pairs, dummy_bound)
    results_list = List([initial_tuple] * counter)

    for thread in prange(counter):

        quad_x0 = quads[thread, 0] * 32
        quad_y0 = quads[thread, 1] * 32
        x_cell_offset = quad_x0 - x_low - 1
        y_cell_offset = quad_y0 - y_low - 1
        x_pos = x_cell_offset * 32
        y_pos = y_cell_offset * 32
        x_pos_end = x_pos + 34 * 32 + 1
        y_pos_end = y_pos + 34 * 32 + 1
        
        height_map = worldspace_heightmap[y_pos:y_pos_end, x_pos:x_pos_end]

        vertices_array, pairs, boundary = GenerateLODMeshData(height_map, target_verts=target_verts, min_error=min_error, vertices_batch=vertices_batch)
        results_list[thread] = (vertices_array, pairs, boundary)

    return results_list




@njit(cache=True)
def GenerateLODMeshDataWrapperST(worldspace_heightmap, quads, x_low, y_low, counter, target_verts, min_error, vertices_batch):

    dummy_verts = np.empty((0, 3), dtype=np.float64)
    dummy_pairs = np.empty((0, 2), dtype=np.int32)
    dummy_bound = np.empty((0,), dtype=np.bool_)
    initial_tuple = (dummy_verts, dummy_pairs, dummy_bound)
    results_list = List([initial_tuple] * counter)

    for thread in prange(counter):

        quad_x0 = quads[thread, 0] * 32
        quad_y0 = quads[thread, 1] * 32
        x_cell_offset = quad_x0 - x_low - 1
        y_cell_offset = quad_y0 - y_low - 1
        x_pos = x_cell_offset * 32
        y_pos = y_cell_offset * 32
        x_pos_end = x_pos + 34 * 32 + 1
        y_pos_end = y_pos + 34 * 32 + 1
        
        height_map = worldspace_heightmap[y_pos:y_pos_end, x_pos:x_pos_end]

        vertices_array, pairs, boundary = GenerateLODMeshData(height_map, target_verts=target_verts, min_error=min_error, vertices_batch=vertices_batch)
        results_list[thread] = (vertices_array, pairs, boundary)

    return results_list
    
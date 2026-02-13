from numba import prange, njit, types
import numpy as np
import logging
import math

'''''NORMAL MAPS FUNCTIONS'''

bicubic_convolution_kernel = np.array([[0, 2, 0, 0],
                                      [-1, 0, 1, 0],
                                      [2, -5, 4, -1],
                                        [-1, 3, -3, 1],])

@njit
def cubic_kernel(t, a=-0.5):

    abs_t = np.abs(t)
    abs_t2 = abs_t ** 2
    abs_t3 = abs_t ** 3

    result = np.where(
        abs_t <= 1,
        (a + 2) * abs_t3 - (a + 3) * abs_t2 + 1,
        np.where(
            abs_t < 2,
            a * abs_t3 - 5 * a * abs_t2 + 8 * a * abs_t - 4 * a,
            0
        )
    )
    return np.float64(result)


#decided not to go with the heightmap upscaling as it looks like shit
#and if i need noise, well, i can just add it later
@njit(parallel=True)
def build_resampled_heightmap(extended_grid, height_map, scale_factor):

    bicubic_weights = np.zeros((scale_factor, 4), dtype=np.float64)
    for i in range(scale_factor):
        t = np.float64(i / scale_factor)
        for j in range(4):
            bicubic_weights[i, j] = cubic_kernel(j - 1 - t)

    #print(bicubic_weights)
    resampled_heightmap = np.zeros((extended_grid, extended_grid), dtype=np.float64)

    

    for i in prange(17, extended_grid - 16):

        fx = np.int32(i % scale_factor)
        x = np.int32(i // scale_factor)
        wx = bicubic_weights[fx]
        x_indices = np.arange(x-1, x + 3)

        for j in range(17, extended_grid - 16):
            fy = np.int32(j % scale_factor)
            y = np.int32(j // scale_factor)
            wy = bicubic_weights[fy]
            y_indices = np.arange(y-1, y + 3)
            sample = np.empty((4, 4), dtype=np.float64)
            for xi in range(4):
                for yi in range(4):
                    sample[xi, yi] = height_map[x_indices[xi], y_indices[yi]]
                    
            resampled_heightmap[i, j] = np.dot(wx, np.dot(sample, wy))

    return resampled_heightmap

#simple derivative-based normal map generation
#not used
@njit(parallel=True)
def build_base_normal_map(normals_dx, normals_dy, normal_map_dimension, scale_factor, normal_boost):
    base_normal_map = np.zeros((normal_map_dimension, normal_map_dimension, 3), dtype=np.float64)
    for i in prange(normal_map_dimension):
        for j in range(normal_map_dimension):
            dx = - normal_boost * scale_factor * (normals_dx[i + 31 * scale_factor, j + 31 * scale_factor] + normals_dx[i + 31 * scale_factor, j + 1 + 31 * scale_factor])
            dy = - normal_boost * scale_factor * (normals_dy[i + 31 * scale_factor, j + 31 * scale_factor] + normals_dy[i + 1 + 31 * scale_factor, j + 31 * scale_factor])
            norm = 1 / np.sqrt(dx * dx + dy * dy + 1)
            
            base_normal_map[i, j, 0] = dx * norm * 127 + 128
            base_normal_map[i, j, 1] = dy * norm * 127 + 128
            base_normal_map[i, j, 2] = 1 * norm * 127 + 128
    return base_normal_map



@njit(parallel=True, fastmath=True)
def build_base_normal_map_vertexbaking(vertex_normals_a, normal_map_dimension, z_boost = 1.0):

    normal_map = np.zeros((normal_map_dimension, normal_map_dimension, 3), dtype=np.float64)

    scale_factor = np.int32(normal_map_dimension / ((vertex_normals_a.shape[0] - 1))) 
   
    #print(scale_factor)
    
    for j in prange(normal_map_dimension):
        y_int = np.int32(j // scale_factor)
        y_frac = (j + 0.5) / scale_factor - y_int

        for i in range(normal_map_dimension):

            x_int = np.int32(i // scale_factor)
            x_frac = (i + 0.5) / scale_factor - x_int
            
            ###
            #\#
            ###
            #barycentric coords for right triangle with side == 1

            upper = y_frac > (1.0 - x_frac)

            if upper:
                p0_x = x_int + 1
                p0_y = y_int + 1
                p1_x = x_int 
                p1_y = y_int + 1
                p2_x = x_int + 1
                p2_y = y_int 
                u, v, w = x_frac + y_frac - 1.0, 1.0 - x_frac, 1.0 - y_frac
            else:
                p0_x = x_int
                p0_y = y_int
                p1_x = x_int + 1
                p1_y = y_int
                p2_x = x_int
                p2_y = y_int + 1
                u, v, w = 1.0 - x_frac - y_frac, x_frac, y_frac
            
            #u, v, w = np.float64(1.0), np.float64(0.0), np.float64(0.0)

            nx = (vertex_normals_a[p0_y, p0_x, 0] * u +
                  vertex_normals_a[p1_y, p1_x, 0] * v +
                  vertex_normals_a[p2_y, p2_x, 0] * w)
            ny = (vertex_normals_a[p0_y, p0_x, 1] * u +
                  vertex_normals_a[p1_y, p1_x, 1] * v +
                  vertex_normals_a[p2_y, p2_x, 1] * w)
            nz = ((vertex_normals_a[p0_y, p0_x, 2] * u +
                  vertex_normals_a[p1_y, p1_x, 2] * v +
                  vertex_normals_a[p2_y, p2_x, 2] * w)) * z_boost
            

            inv_len = 1.0 / math.sqrt(nx*nx + ny*ny + nz*nz)

            normal_map[j, i, 0] = nx * inv_len
            normal_map[j, i, 1] = ny * inv_len
            normal_map[j, i, 2] = nz * inv_len

            

    return normal_map



@njit
def create_vertices_from_heightmap_simple(heightmap):

    rows, cols = heightmap.shape #8*32*32 + 1, 8*32*32 + 1

    vertices = np.zeros((rows * cols, 3), dtype=np.float64)

    triangles = np.zeros(((rows - 1) * (cols - 1) * 2, 3), dtype=np.uint32)
    scaling_factor = 1 / ((heightmap.shape[0] - 1 )/32/34)
    #print(scaling_factor)

    for i in range(rows):
        for j in range(cols):
            x = np.float64(128 * j) * scaling_factor 
            y = np.float64(128 * i) * scaling_factor
            #z = np.float64(heightmap[i + 32 - 1, j + 32 - 1])
            z = np.float64(heightmap[i, j])
            vertices[i * cols + j, 0] = x
            vertices[i * cols + j, 1] = y
            vertices[i * cols + j, 2] = z


       
    for i in range(rows - 1):
        for j in range(cols - 1):
            triangles[i * (cols - 1) * 2 + j * 2, 0] =  i * cols + j
            triangles[i * (cols - 1) * 2 + j * 2, 1] = i * cols + j + 1
            triangles[i * (cols - 1) * 2 + j * 2, 2] = (i + 1) * cols + j
            triangles[i * (cols - 1) * 2 + j * 2 + 1, 0] = (i + 1) * cols + j
            triangles[i * (cols - 1) * 2 + j * 2 + 1, 1] = i * cols + j + 1
            triangles[i * (cols - 1) * 2 + j * 2 + 1, 2] = (i + 1) * cols + j + 1



    return vertices, triangles

@njit(parallel=True, fastmath=True)
def calc_face_normals_non_normalized(tris, verts):
    normals = np.zeros((len(tris), 3), dtype=np.float64)
    for i in prange(len(tris)):
        v0 = verts[tris[i, 0]]
        v1 = verts[tris[i, 1]]
        v2 = verts[tris[i, 2]]
        edge1 = v1 - v0
        edge2 = v2 - v0
        normals[i] = np.cross(edge1, edge2)
    return normals

@njit(fastmath=True)
def calculate_angle(p_0, p_a, p_b):
    a = p_a - p_0
    b = p_b - p_0
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    cos_theta = max(min(dot_product / (norm_a * norm_b), 1.0), -1.0)
    return np.arccos(cos_theta)

@njit(fastmath=True, parallel=True)
def calc_vertex_normals_angle_weighted(tris, verts, normals):

    vertex_normals = np.zeros((len(verts), 3), dtype=np.float64)

    for i in prange(len(tris)):
        vertex_normals[tris[i, 0]] += normals[i] * calculate_angle(verts[tris[i, 0]], verts[tris[i, 1]], verts[tris[i, 2]])
        vertex_normals[tris[i, 1]] += normals[i] * calculate_angle(verts[tris[i, 1]], verts[tris[i, 0]], verts[tris[i, 2]])
        vertex_normals[tris[i, 2]] += normals[i] * calculate_angle(verts[tris[i, 2]], verts[tris[i, 0]], verts[tris[i, 1]])

    for i in prange(len(vertex_normals)):
        vertex_normals[i] = np.divide(vertex_normals[i], np.linalg.norm(vertex_normals[i]))

    return vertex_normals




@njit(parallel=True)
def compute_directional_slopes(
    height: np.ndarray,
    resolution: int,
    total_range: int,
):

    
    H, W = height.shape
    spacing_x, spacing_y = np.float32(4096/32), np.float32(4096/32)
    westslope  = np.zeros((H, W), dtype=np.float32)
    eastslope  = np.zeros((H, W), dtype=np.float32)
    northslope = np.zeros((H, W), dtype=np.float32)

    range1 = 32
    range2 = 64
    range3 = 128

    
    for j in prange(H):
        for i in range(W):
            h0 = height[j, i]

            # west direction (dx = –1, dy = 0) 
            maxw = 0.0
            dxpix = 0
            step = 1
            while True:
                dxpix -= step
                distpix = abs(dxpix)
                if distpix < range1:
                    step = 1
                elif distpix < range2:
                    step = 2
                elif distpix < range3:
                    step = 4
                else:
                    step = 8
                
                if distpix > total_range or i + dxpix < 0:
                    break
                slope = (height[j, i + dxpix] - h0) / (distpix * spacing_x)
                if slope > maxw:
                    maxw = slope
                    
            westslope[j, i] = maxw

            # east direction (dx = +1, dy = 0) 
            maxe = 0.0
            dxpix = 0
            step = 1
            while True:
                dxpix += step
                distpix = abs(dxpix)

                if distpix < range1:
                    step = 1
                elif distpix < range2:
                    step = 2
                elif distpix < range3:
                    step = 4
                else:
                    step = 8
                if distpix > total_range or i + dxpix >= W:
                    break
                slope = (height[j, i + dxpix] - h0) / (distpix * spacing_x)
                if slope > maxe:
                    maxe = slope

            eastslope[j, i] = maxe

            # north direction (dx = 0, dy = –1) 
            maxn = 0.0
            dypix = 0
            step = 1
            while True:
                dypix -= step
                distpix = abs(dypix)
                
                if distpix < range1:
                    step = 1
                elif distpix < range2:
                    step = 2
                elif distpix < range3:
                    step = 4
                else:
                    step = 8

                if distpix > total_range or j + dypix < 0:
                    break
                slope = (height[j + dypix, i] - h0) / (distpix * spacing_y)
                if slope > maxn:
                    maxn = slope

            northslope[j, i] = maxn

    return westslope, eastslope, northslope

@njit(parallel=True, fastmath=True)
def apply_shadows_to_vertex_normals(
    vertex_normals: np.ndarray,   # (H, W, 3) NORMALIZED vertex normals
    westslope: np.ndarray,
    eastslope: np.ndarray,
    northslope: np.ndarray,
    northboost: float = 1.0,
    shadow_strength: float = 1.0,
):
  

    H, W = vertex_normals.shape[0], vertex_normals.shape[1]
    result = np.zeros_like(vertex_normals)
    
    for j in prange(H):
        for i in range(W):
            nx = vertex_normals[j, i, 0]
            ny = vertex_normals[j, i, 1]
            #nz = vertex_normals[j, i, 2]
            
            sw = westslope[j, i]
            se = eastslope[j, i]
            sn = northslope[j, i]
            
            tilt = (se - sw) * shadow_strength
            north_factor = math.sqrt(max(sw, 0.0) * max(se, 0.0)) * northboost * shadow_strength
            
            nx_mod = nx - tilt
            ny_mod = ny + north_factor
            
            if ny_mod > sn * shadow_strength:
                ny_mod = sn * shadow_strength
            
            
            result[j, i, 0] = nx_mod
            result[j, i, 1] = ny_mod
            result[j, i, 2] = vertex_normals[j, i, 2] #NO normalization here - baking will do it
    
    return result


#numba-compatible rewrite of this: https://github.com/plottertools/vnoise/blob/main/vnoise/vnoise.py

GRAD3 = np.array([
    [1, 1], [-1, 1], [1, -1], [-1, -1],
    [1, 0], [-1, 0], [1, 0], [-1, 0],
    [0, 1], [0, -1], [0, 1], [0, -1],
    [1, 0], [-1, 0], [0, -1], [0, 1],
], dtype=np.float64)

PERM = np.array([
    151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225,
    140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148,
    247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32,
    57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175,
    74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122,
    60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54,
    65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169,
    200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64,
    52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212,
    207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213,
    119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9,
    129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104,
    218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241,
    81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157,
    184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93,
    222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180,
    151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225,
    140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148,
    247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32,
    57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175,
    74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122,
    60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54,
    65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169,
    200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64,
    52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212,
    207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213,
    119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9,
    129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104,
    218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241,
    81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157,
    184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93,
    222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180,
], dtype=np.int32)


@njit(cache=True)
def _grad2(hash_val, x, y):
    h = hash_val & 15
    return x * GRAD3[h, 0] + y * GRAD3[h, 1]


@njit(cache=True)
def _fade(t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


@njit(cache=True)
def _lerp(t, a, b):
    return a + t * (b - a)


@njit(cache=True)
def _noise2_single(x, y, repeat_x, repeat_y,):

    # Grid cell coordinates with tiling
    i = int(np.floor(x % repeat_x))
    j = int(np.floor(y % repeat_y))
    ii = int((i + 1) % repeat_x)
    jj = int((j + 1) % repeat_y)
    
    i = (i & 255) 
    j = (j & 255) 
    ii = (ii & 255) 
    jj = (jj & 255) 
    
    xf = x - np.floor(x)
    yf = y - np.floor(y)
    
    # Fade curves
    u = _fade(xf)
    v = _fade(yf)
    
    # Hash coordinates of the 4 corners
    A = PERM[i]
    B = PERM[ii]
    AA = PERM[A + j]
    AB = PERM[A + jj]
    BA = PERM[B + j]
    BB = PERM[B + jj]
    
    # Gradient dot products at corners, then interpolate
    return _lerp(v,
        _lerp(u, _grad2(PERM[AA], xf, yf), _grad2(PERM[BA], xf - 1.0, yf)),
        _lerp(u, _grad2(PERM[AB], xf, yf - 1.0), _grad2(PERM[BB], xf - 1.0, yf - 1.0))
    )


@njit(parallel=True, cache=True)
def noise2_grid(x, y, octaves=1, persistence=0.5, lacunarity=2.0,
                repeat_x=1024, repeat_y=1024):

    nx = x.shape[0]
    ny = y.shape[0]
    result = np.empty((nx, ny), dtype=np.float64)
    
    for i in prange(nx):
        for j in range(ny):

            if octaves == 1:
                result[i, j] = _noise2_single(x[i], y[j], repeat_x, repeat_y)
            else:
                    
                total = 0.0
                freq = 1.0
                ampl = 1.0
                max_ampl = 0.0

                for _ in range(octaves):
                    total += _noise2_single(
                        x[i] * freq, y[j] * freq,
                        int(repeat_x * freq), int(repeat_y * freq)
                        
                    ) * ampl
                    max_ampl += ampl
                    freq *= lacunarity
                    ampl *= persistence
                result[i, j] = total / max_ampl
                
    return result


def noise_generate_height_map_vectorized(width, height, scale=100.0, octaves=6, 
                                   persistence=0.5, lacunarity=2.0):
    scale = scale * (width / 8192)
    x = np.arange(width, dtype=np.float32) / scale 
    y = np.arange(height, dtype=np.float32) / scale 
        
    values = noise2_grid(x, y, octaves=octaves, 
                             persistence=persistence, 
                             lacunarity=lacunarity)
    
    height_map = values.reshape(height, width)
    height_map = (height_map - height_map.min()) / (height_map.max() - height_map.min())
    
    return height_map

@njit(parallel=True, fastmath=True)
def blend_normal_maps(base_normal, detail_normal):
    
    #Blend two normal maps using Reoriented Normal Mapping (RNM)   
    #Note z-coordinate multiplication
    #normals are in [-1, 1]

    h, w = base_normal.shape[:2]
    result = np.zeros((h, w, 3), dtype=np.float32)
    
    for i in prange(h):
        for j in prange(w):
            n1_x = base_normal[i, j, 0]
            n1_y = base_normal[i, j, 1]
            n1_z = base_normal[i, j, 2]
            
            n2_x = detail_normal[i, j, 0]
            n2_y = detail_normal[i, j, 1]
            n2_z = detail_normal[i, j, 2]
                        
            # Reoriented Normal Mapping blend
            result_x = n1_x + n2_x
            result_y = n1_y + n2_y
            result_z = n1_z * n2_z
            
            norm = np.sqrt(result_x*result_x + result_y*result_y + result_z*result_z)
            if norm > 0.0:
                result_x /= norm
                result_y /= norm
                result_z /= norm
            
            result[i, j, 0] = result_x
            result[i, j, 1] = result_y
            result[i, j, 2] = result_z
    
    return result


@njit(fastmath=True)
def compute_gradients_vectorized(height_map, strength):
    h, w = height_map.shape
    
    size_mult = h / 8192.0

    dx = np.zeros((h, w), dtype=np.float32)
    dy = np.zeros((h, w), dtype=np.float32)
    
    #central differences for interior
    dx[:, 1:-1] = (height_map[:, 2:] - height_map[:, :-2]) * 0.5 * strength * size_mult
    dy[1:-1, :] = (height_map[2:, :] - height_map[:-2, :]) * 0.5 * strength * size_mult
    
    #forward/backward differences for edges
    dx[:, 0] = (height_map[:, 1] - height_map[:, 0]) * strength * size_mult
    dx[:, -1] = (height_map[:, -1] - height_map[:, -2]) * strength * size_mult
    dy[0, :] = (height_map[1, :] - height_map[0, :]) * strength * size_mult
    dy[-1, :] = (height_map[-1, :] - height_map[-2, :]) * strength * size_mult
    
    return dx, dy

@njit(parallel=True, fastmath=True)
def gradients_to_normals(dx, dy):
    h, w = dx.shape
    normal_map = np.zeros((h, w, 3), dtype=np.float32)
    
    for i in prange(h):
        for j in prange(w):
            nx = -dx[i, j]
            ny = -dy[i, j]
            nz = 1.0
            
            norm = np.sqrt(nx*nx + ny*ny + nz*nz)
            
            normal_map[i, j, 0] = (nx / norm)  
            normal_map[i, j, 1] = (ny / norm) 
            normal_map[i, j, 2] = (nz / norm) 
    
    return normal_map

def noise_height_to_normal_map_fast(height_map, strength=1.0):
    dx, dy = compute_gradients_vectorized(height_map.astype(np.float32), strength)
    return gradients_to_normals(dx, dy)
import math
import logging
from numba import njit, prange
import numpy as np



''''COLOR MAP FUNCTIONS'''
HASH_SCALE_FACTOR = 43758.5453 
@njit(cache=True)
def hash2(p): #taken from Deliot 2019 paper         
    return (
        math.sin(p[0] * 127.1 + p[1] * 311.7) * HASH_SCALE_FACTOR % 1.0,
        math.sin(p[0] * 269.5 + p[1] * 183.3) * HASH_SCALE_FACTOR % 1.0
    )

def build_vert_grid(dimension, side):

    tri_height = math.sqrt(3) / 2 * side

    max_rows = math.ceil(dimension / tri_height) + 1
    max_cols = math.ceil(dimension / side) + 1 + 1 #parallelogram

    vertices = [ #this is stupid tbh
         [
            
                #v * side / 2 is a row by row offset
                #every row is shifted by 1/2side
                (u * side - (v % 2) * (side/2), v * tri_height) 
                #v * max_cols + u,
                #hash2((u * side + v * (side/2), v * tri_height))
            
            for u in range(int(1.2 * max_cols))
        ]
        for v in range(int(1.2 * max_rows))
    ]

    hashes = np.array([[hash2((u * side + v * (side/2), v * tri_height)) for u in range(int(1.2 * max_cols))] for v in range(int(1.2 * max_rows))])

    parameters = (dimension, side, tri_height, max_rows, max_cols)

    return np.array(vertices), hashes, parameters

SQRT3 = math.sqrt(3)
@njit(cache=True)
def get_triangle(x, y, tri_height, tri_side, tri_side_half):
    

    row = int(y / tri_height) #never negative
    column = int((x + (row % 2) * (tri_side_half)) / tri_side)

    o_y = y - row * tri_height
    o_x = x - column * tri_side + (row % 2) * (tri_side_half)

    #line 1 is y = sqrt(3) * x
    #line 2 is y = 2*sqrt(3)/2-sqrt(3) * x

    if o_x <= tri_side_half:
        if o_y > SQRT3 * o_x:
            r0, c0 = row, column
            r1, c1 = row + 1, column - (row % 2)
            r2, c2 = row + 1, column + 1 - (row % 2)
        else:
            r0, c0 = row, column
            r1, c1 = row, column + 1
            r2, c2 = row + 1, column + 1 - (row % 2)

    else:
        if o_y > 2 * tri_height - SQRT3 * o_x: 
            r0, c0 = row, column + 1
            r1, c1 = row + 1, column + 1 - (row % 2)
            r2, c2 = row + 1, column + 2 - (row % 2)
        else:
            r0, c0 = row, column
            r1, c1 = row, column + 1
            r2, c2 = row + 1, column + 1 - (row % 2)

    return r0, c0, r1, c1, r2, c2

'''
def generate_point_texture_OLD(x, y, quad_x, quad_y, worldspace):
    y = 32 * 32 - y #because of the way the texture is read
    x_cell, y_cell, x_point, y_point = get_matching_land_coords(x, y, quad_x, quad_y)
    #print(x_cell, y_cell)
    land = get_land_from_cell_coords(x_cell, y_cell, worldspace)
    if not land:
        return None
    texture_id_points = land.texture_id_points[y_point][x_point] #don't fuck up the order, VTXT is read row by row
    # 0 full image, 1 top left, 2 top right, 3 bottom left, 4 bottom right
    if x_point % 2 == 0:
        if y_point % 2 == 0:
            tex_quad = 2
        else:
            tex_quad = 4
    else:
        if y_point % 2 == 0:
            tex_quad = 1
        else:
            tex_quad = 3

    tex_quad = random.randint(1,4)

    opacity_points = land.opacity_points
    texture = Image.new("RGB", (int(CHUNK_DIM), int(CHUNK_DIM)), (0, 0, 0))
    
    
    opacities = opacity_points[y_point][x_point]
    for i in range(len(texture_id_points)):

        if texture_id_points[i] == 0:
            continue
        if not texture_id_points[i] in ltex_to_texture:
            continue
        #print(texture_id_points[i])
        angle = 0
        angle = random.choice([0, 90, 180, 270])

        ltex = ltex_to_texture[texture_id_points[i]][tex_quad].rotate(angle)

        if random.randint(0,1) == 1:
            ltex.transpose(Image.FLIP_LEFT_RIGHT)
        if random.randint(0,1) == 1:
            ltex.transpose(Image.FLIP_TOP_BOTTOM)
        
        texture = Image.blend(texture, ltex, opacities[i])
        #print(opacities[i])


    color_layer = Image.new("RGB", (int(tex_point_dim), int(tex_point_dim)), tuple(land.vertex_colors[y_point][x_point]))

    texture = ImageChops.multiply(texture, color_layer)


    return texture


@njit
def fill_chunk_JIT_OLD(offset_y, offset_x, grid, grid_hashes, ltex, average_color, chunk_dim):
    
    result = np.zeros((int(tex_point_dim), int(tex_point_dim), 3))
    for j in range(int(tex_point_dim)): #i horizontal j vertical
        y_coord = offset_y + j
        for i in range(int(tex_point_dim)):
            x_coord = offset_x + i
            r0, c0, r1, c1, r2, c2 = get_triangle(x_coord, y_coord)
            #print(r0, c0, r1, c1, r2, c2)
            color0 = ltex[round(y_coord - grid[r0][c0][1] + grid_hashes[r0][c0][1]*chunk_dim) % chunk_dim][round(x_coord - grid[r0][r0][0] + grid_hashes[r0][c0][0]*chunk_dim) % chunk_dim]
            color1 = ltex[round(y_coord - grid[r1][c1][1] + grid_hashes[r1][c1][1]*chunk_dim) % chunk_dim][round(x_coord - grid[r1][c1][0] + grid_hashes[r1][c1][0]*chunk_dim) % chunk_dim]
            color2 = ltex[round(y_coord - grid[r2][c2][1] + grid_hashes[r2][c2][1]*chunk_dim) % chunk_dim][round(x_coord  - grid[r2][c2][0] + grid_hashes[r2][c2][0]*chunk_dim) % chunk_dim]
            
            l1, l2, l3 = get_barycentric_coordinates(x_coord, y_coord, grid[r0, c0, 0], grid[r0, c0, 1], grid[r1, c1, 0], grid[r1, c1, 1], grid[r2, c2, 0], grid[r2, c2, 1])

            
            result[j, i] = (color0 * l1 + color1 * l2 + color2 * l3 - average_color) / math.sqrt(l1 * l1 + l2 * l2 + l3 * l3) + average_color
    
    return result
'''

@njit(parallel=True, cache=True)
def triangle_weight_precalc(grid, grid_hashes, internal_dimension):
    texture_sample_res = int(internal_dimension / 32 / 16)
    tri_side = texture_sample_res / 2
    tri_side_half = tri_side / 2
    tri_height = tri_side * math.sqrt(3) / 2
    precalc_data = np.zeros((int(internal_dimension), int(internal_dimension), 6), dtype=np.uint8)
    precalc_bar_weights = np.zeros((int(internal_dimension), int(internal_dimension), 4), dtype=np.float32)
    for j in prange(int(internal_dimension)):
        for i in range(int(internal_dimension)):
            r0, c0, r1, c1, r2, c2 = get_triangle(i, j, tri_height, tri_side, tri_side_half)
            col0_x = round(j - grid[r0, c0, 1] + grid_hashes[r0, c0, 1] * texture_sample_res) % texture_sample_res 
            col0_y = round(i - grid[r0, c0, 0] + grid_hashes[r0, c0, 0] * texture_sample_res) % texture_sample_res
            col1_x = round(j - grid[r1, c1, 1] + grid_hashes[r1, c1, 1] * texture_sample_res) % texture_sample_res
            col1_y = round(i - grid[r1, c1, 0] + grid_hashes[r1, c1, 0] * texture_sample_res) % texture_sample_res
            col2_x = round(j - grid[r2, c2, 1] + grid_hashes[r2, c2, 1] * texture_sample_res) % texture_sample_res
            col2_y = round(i - grid[r2, c2, 0] + grid_hashes[r2, c2, 0] * texture_sample_res) % texture_sample_res
            l1, l2, l3 = get_barycentric_coordinates(i, j, grid[r0, c0, 0], grid[r0, c0, 1], grid[r1, c1, 0], grid[r1, c1, 1], grid[r2, c2, 0], grid[r2, c2, 1])
            precalc_data[j, i] = np.array([col0_x, col0_y, col1_x, col1_y, col2_x, col2_y])
            precalc_bar_weights[j, i] = np.array([l1, l2, l3, math.sqrt(l1*l1 + l2*l2 + l3*l3)])
    return precalc_data, precalc_bar_weights



'''
@profile
def generate_point_texture_OLD2(x, y, quad_x, quad_y, worldspace, tiling_offset):
    
    offset =  (int((y - quad_y - tiling_offset) * tex_point_dim), int((x - quad_x - tiling_offset) * tex_point_dim)) #pillow [vertical, horizontal]
    
    y = 32 * 32 - y #because of the way the texture is read
    x_cell, y_cell, x_point, y_point = get_matching_land_coords(x, y, quad_x, quad_y)
    #print(x_cell, y_cell)
    land = get_land_from_cell_coords(x_cell, y_cell, worldspace)
    if not land:
        return None
    texture_id_points = land.texture_id_points[y_point][x_point] #don't fuck up the order, VTXT is read row by row
    # 0 full image, 1 top left, 2 top right, 3 bottom left, 4 bottom right
    

    opacity_points = land.opacity_points
    texture = Image.new("RGB", (int(tex_point_dim), int(tex_point_dim)), (0, 0, 0))
    
    
    opacities = opacity_points[y_point][x_point]
    for o in range(1): #range(len(texture_id_points)):

        if texture_id_points[o] == 0:
            continue
        if not texture_id_points[o] in ltex_to_texture:
            continue

        
        #ltex = ltex_to_texture[texture_id_points[o]][0]
        #average_color = ltex_to_texture[texture_id_points[o]][5]

        ltex = ltex_to_texture[116663][0]
        average_color = ltex_to_texture[116663][5]

        #temp_texture = np.array(Image.new("RGB", (int(tex_point_dim), int(tex_point_dim)), (0, 0, 0)))

        temp_texture = fill_chunk_JIT(offset[0], offset[1], grid, grid_hashes, ltex, average_color, chunk_dim)

        #for j in range(int(chunk_dim/2)): #i horizontal j vertical
        #    for i in range(int(chunk_dim/2)):
        #        r0, c0, r1, c1, r2, c2 = get_triangle(offset[1] + i, offset[0] + j)
        #        color0 = ltex[round(offset[0] + j - grid[r0][c0][1] + grid_hashes[r0][c0][1]*chunk_dim) % chunk_dim][round(offset[1] + i - grid[r0][r1][0] + grid_hashes[r0][c0][0]*chunk_dim) % chunk_dim]
        #        color1 = ltex[round(offset[0] + j - grid[r1][c1][1] + grid_hashes[r1][c1][1]*chunk_dim) % chunk_dim][round(offset[1]  + i - grid[r1][c1][0] + grid_hashes[r1][c1][0]*chunk_dim) % chunk_dim]
        #        color2 = ltex[round(offset[0] + j - grid[r2][c2][1] + grid_hashes[r2][c2][1]*chunk_dim) % chunk_dim][round(offset[1] + i  - grid[r2][c2][0] + grid_hashes[r2][c2][0]*chunk_dim) % chunk_dim]#
        #
        #        #temp_texture[j][i] = (vertices[0][2][0]*255, vertices[1][2][0]*255, vertices[2][2][0]*255)
        #       weights = get_barycentric_coordinates(offset[1] + i, offset[0] + j, *grid[r0][c0], *grid[r1][c1], *grid[r2][c2])
        #        temp_texture[j][i] = (color0 * weights[0] + color1 * weights[1] + color2 * weights[2] - average_color) / math.sqrt(weights[0] * weights[0] + weights[1] * weights[1] + weights[2] * weights[2]) + average_color


        texture = Image.blend(texture, Image.fromarray(temp_texture.astype(np.uint8)), 1) #opacities[o])
        #print(opacities[i])


    color_layer = Image.new("RGB", (int(tex_point_dim), int(tex_point_dim)), tuple(land.vertex_colors[y_point][x_point]))

    #texture = ImageChops.multiply(texture, color_layer)


    return texture
'''

@njit(cache=True)
def get_barycentric_coordinates(px, py, x1, y1, x2, y2, x3, y3):

    denom = ((y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3))

    l1 = ((y2 - y3)*(px - x3) + (x3 - x2)*(py - y3)) / denom
    l2 = ((y3 - y1)*(px - x3) + (x1 - x3)*(py - y3)) / denom
    l3 = 1 - l1 - l2

    return l1, l2, l3


def get_matching_land_coords(x, y, quad_x, quad_y):
    dimension = 32 * 32
    x_cell = math.floor(x / dimension * 32) + quad_x
    y_cell = math.floor(y / dimension * 32) + quad_y
    #print(x / tex_dimension * 32)
    x_point  = int(math.modf(x / dimension * 32)[0] * 32)
    y_point  = int(math.modf(y / dimension * 32)[0] * 32)
    if x_point < 0:
        x_point += 32
    if y_point < 0:
        y_point += 32

    return x_cell, y_cell, x_point, y_point

@njit(fastmath=True, cache=True)
def get_matching_land_coords_zeroquad(x, y):
    #dimension = 32 * 32
    dimension_inv_mult = 1 / 32 # 1 / (32 * 32) * 32
    x_cell = math.floor(x * dimension_inv_mult)
    y_cell = math.floor(y * dimension_inv_mult)
    x_point  = int((x * dimension_inv_mult - int(x * dimension_inv_mult)) * 32)
    y_point  = int((y * dimension_inv_mult- int(y * dimension_inv_mult)) * 32)
    if x_point < 0:
        x_point += 32
    if y_point < 0:
        y_point += 32
    return (x_cell * 32 + x_point), (y_cell *32 + y_point)



def get_land_from_cell_coords(x, y, worldspace, cell_data):
    if worldspace not in cell_data:
        return None
    if x not in cell_data[worldspace]:
        return None
    if y not in cell_data[worldspace][x]:
        return None
    return cell_data[worldspace][x][y]


@njit(fastmath=True, cache=True)
def fill_chunk_JIT(offset_y, offset_x, ltex, average_color, tiling_data, tiling_weights, chunk_dim):
    
    result = np.zeros((chunk_dim, chunk_dim, 3), dtype=np.float32)

    for j in range(chunk_dim): #i horizontal j vertical
        y_coord = offset_y + j
        for i in range(chunk_dim):
            
            x_coord = offset_x + i
            #print(y_coord, x_coord, offset_y) 
            #y1, x1, y2, x2, y3, x3 = PRECALCULATED_TILING_DATA[y_coord, x_coord, 6:6]
            color0_r, color0_g, color0_b = ltex[tiling_data[y_coord, x_coord, 0]][tiling_data[y_coord, x_coord, 1]]
            color1_r, color1_g, color1_b = ltex[tiling_data[y_coord, x_coord, 2]][tiling_data[y_coord, x_coord, 3]]
            color2_r, color2_g, color2_b = ltex[tiling_data[y_coord, x_coord, 4]][tiling_data[y_coord, x_coord, 5]]
            l1, l2, l3, s_w = tiling_weights[y_coord, x_coord]

            #this is 3x faster than numpy vectorized operations
            #note that this assumes that colors are normal-distributed which is likely NOT the case
            #technically we should reconstruct the original histogram but it looks ok even this way
            #and reconstructing will be much slowe
            result[j, i, 0] = max(min((color0_r * l1 + color1_r * l2 + color2_r * l3 - average_color[0]) / s_w + average_color[0], 255.0), 0.0)
            result[j, i, 1] = max(min((color0_g * l1 + color1_g * l2 + color2_g * l3 - average_color[1]) / s_w + average_color[1], 255.0), 0.0)
            result[j, i, 2] = max(min((color0_b * l1 + color1_b * l2 + color2_b * l3 - average_color[2]) / s_w + average_color[2], 255.0), 0.0)
            
    return result

@njit(fastmath=True, cache=True)
def fill_alpha_chunk_JIT(offset_y, offset_x, ltex_alpha, tiling_data, tiling_weights, chunk_dim):
    
    result = np.zeros((chunk_dim, chunk_dim), dtype=np.float32)

    for j in range(chunk_dim): 
        y_coord = offset_y + j
        for i in range(chunk_dim):
            
            x_coord = offset_x + i
            alpha0 = ltex_alpha[tiling_data[y_coord, x_coord, 0]][tiling_data[y_coord, x_coord, 1]]
            alpha1 = ltex_alpha[tiling_data[y_coord, x_coord, 2]][tiling_data[y_coord, x_coord, 3]]
            alpha2 = ltex_alpha[tiling_data[y_coord, x_coord, 4]][tiling_data[y_coord, x_coord, 5]]
            l1, l2, l3, s_w = tiling_weights[y_coord, x_coord]

            result[j, i] = max(min((alpha0 * l1 + alpha1 * l2 + alpha2 * l3) / s_w , 1.0), 0.0)
            
    return result

@njit(fastmath=True, cache=True)
def blend_textures(texture, new_texture, opacity, chunk_dim):
    for j in range(chunk_dim):
        for i in range(chunk_dim):
            for c in range(3):
                texture[j, i, c] = texture[j, i, c] * (1 - opacity) + new_texture[j, i, c] * opacity
    return texture

@njit(fastmath=True, cache=True)
def blend_textures_with_alpha(texture, new_texture, new_alpha, opacity, chunk_dim):
    for j in range(chunk_dim):
        for i in range(chunk_dim):
            effective_alpha = new_alpha[j, i] * opacity
            for c in range(3):
                texture[j, i, c] = (
                    texture[j, i, c] * (1 - effective_alpha) + 
                    new_texture[j, i, c] * effective_alpha
                )
    return texture


@njit(fastmath=True, cache=True)
def apply_vertex_colors(texture, vertex_colors, chunk_dim):
    for j in range(chunk_dim):
        for i in range(chunk_dim):
            for c in range(3):
                texture[j, i, c] = texture[j, i, c] * (vertex_colors[c] / 255)
    return texture




@njit(cache=True)
def generate_point_texture(x, y, quad_x, quad_y, worldspace, sampling_offset, ltex_to_texture_hashed, 
                           tiling_data, tiling_weights, texture_id_map, opacity_map, vertex_color_map, textures_nparray,
                           textures_nparray_avg, textures_nparray_alpha, chunk_dim):
    
    offset =  (int((y - quad_y) * chunk_dim), int((x - quad_x) * chunk_dim)) #pillow [vertical, horizontal]
    #print(offset)
    y = 32 * 32 - y - 1 #because of the way the texture is read

    x_point, y_point = get_matching_land_coords_zeroquad(x + sampling_offset[0], y + sampling_offset[1])
    
    texture_id_points = texture_id_map[y_point, x_point] #don't fuck up the order, VTXT is read row by row
    opacities = opacity_map[y_point, x_point]
    vertex_colors = vertex_color_map[y_point, x_point]

    texture = np.zeros((chunk_dim, chunk_dim, 3), dtype=np.float32)    
        #texture = generate_point_texture_loop(texture_id_points, offset, opacities)

    
    for o in range(9): #range(len(texture_id_points)):
        texture_id = texture_id_points[o]

        if o == 0:
            if texture_id == 0: texture_id = -1
        elif texture_id == 0:
            continue
        #if not texture_id in ltex_to_texture_hashed:
        #    continue
        
        k = ltex_to_texture_hashed[texture_id]
        ltex = textures_nparray[k]
        average_color = textures_nparray_avg[k]
        texture_alpha = textures_nparray_alpha[k]
        if o == 0:
            opacity = 1
        else:
            opacity = opacities[o]
        #ltex = np.array(ltex_to_texture[116663][0])
        #average_color = ltex_to_texture[116663][5]

        #print(o, texture_id, opacity, average_color)
        #temp_texture = np.zeros((int(tex_point_dim), int(tex_point_dim), 3), dtype=np.float32)

        #temp_texture = np.array(Image.new("RGB", (int(tex_point_dim), int(tex_point_dim)), (0, 0, 0)))

        temp_texture = fill_chunk_JIT(offset[0], offset[1], ltex, average_color, tiling_data, tiling_weights, chunk_dim)
        temp_alpha = fill_alpha_chunk_JIT(offset[0], offset[1], texture_alpha, tiling_data, tiling_weights, chunk_dim)

        
        #texture = blend_textures(texture, temp_texture, opacity)
        if o == 0:
            texture = blend_textures(texture, temp_texture, 1, chunk_dim)
        else:
            texture = blend_textures_with_alpha(texture, temp_texture, temp_alpha, opacity, chunk_dim)
        #texture = texture * (1 - opacity) + temp_texture * opacity
        #print(opacities[i])
        #print(temp_texture)
    
    texture = apply_vertex_colors(texture, vertex_colors, chunk_dim) #texture * (land.vertex_colors[y_point][x_point] / 255)
    #print(texture)

    return texture

@njit(fastmath=True, cache=True)
def blend_textures_matrix(texture, new_texture, opacity, chunk_dim):
    for j in range(chunk_dim):
        for i in range(chunk_dim):
            for c in range(3):
                texture[j, i, c] = texture[j, i, c] * (1 - opacity[j, i, c]) + new_texture[j, i, c] * opacity[j, i, c]
    return texture


@njit(fastmath=True, parallel=True, cache=True)
def generate_full_texture_layer(input, ltex_to_texture_hashed, opacity_matrix, sampling_offset, tiling_data, tiling_weights, texture_id_map, opacity_map, vertex_color_map, textures_nparray, textures_nparray_avg, textures_nparray_alpha, chunk_dim):
    output_texture = input
    for x in prange(32*32):
        #print(x)
        for y in range(32*32):
            if True: #x > 5*32 and x < 6*32 and y > 13*32 and y < 14*32:
                sample = generate_point_texture(x, y, 0, 0, 'Tamriel', sampling_offset,  ltex_to_texture_hashed, tiling_data, tiling_weights, texture_id_map, opacity_map, vertex_color_map, textures_nparray, textures_nparray_avg, textures_nparray_alpha, chunk_dim)
                for i in range(chunk_dim):
                    p = i + x * chunk_dim
                    for j in range(chunk_dim):
                        t = j + y * chunk_dim
                        for c in range(3): #at this point idk why i'm doing this in python
                            output_texture[t, p, c] = output_texture[t, p, c] + sample[j, i, c] * opacity_matrix[j, i]
                #output_texture[y * tex_point_dim : (y+1) * tex_point_dim, x * tex_point_dim : (x+1) * tex_point_dim] = 
    return output_texture


#https://numba.discourse.group/t/mistaken-poor-optimization-of-atan2/1777

@njit(fastmath=True, cache=True, inline='always', locals=dict(z=np.float32))
def atanFast(z):
  return (0.97239411-0.19194795*z*z)*z


@njit(fastmath=True, cache=True, inline='always', locals=dict(x=np.float32, y=np.float32))
def atan2fast1(y, x): 
  if x != 0:
    if abs(x) > abs(y):
      if x > 0:
        return atanFast(y/x)
      else: 
        return atanFast(y/x) + (3.14159 if y >= 0 else -3.14159)
    else: 
      return -atanFast(x/y) + (3.14159/2 if y > 0 else -3.14159/2)
  else:
    return 3.14159/2 if y > 0 else (-3.14159/2 if y < 0 else 0)

@njit(parallel=True, fastmath=True, cache=True)
def compute_ao_map(
    height: np.ndarray,
    total_range: float,
    spacing: float,
    x_origin: int,
    y_origin: int,
    window_size: int,
):
    
    H, W = height.shape
    #ao = np.zeros((H, W), dtype=np.float32)

    spacing = np.float32(spacing)
    spacing_sq = spacing * spacing

    tex_size = window_size
    ao = np.zeros((tex_size, tex_size), dtype=np.float32)

    total_range = int(total_range)

    #8 sampling directions
    dirs_x = np.array([-1, 1, 0, 0, -1, 1, -1, 1], dtype=np.int32)
    dirs_y = np.array([0, 0, -1, 1, -1, -1, 1, 1], dtype=np.int32)

    
    init_step = 128
    step_switch = init_step
    steps = int(np.ceil(np.log2(total_range / init_step + 1)) * init_step) #that's in how many iters we reach total_range if we double the step every 32 iters
    
    for _j in prange(tex_size): #prange(y_origin + 1, y_origin + size + 1, 2):
        j = _j + y_origin + 1
        for _i in range(tex_size): #range(x_origin + 1, x_origin + size + 1, 2):
            i = _i + x_origin + 1            
            h0 = height[j, i]
            sum_angles = 0.0

            
            for d in range(8):
                dx_step = dirs_x[d]
                dy_step = dirs_y[d]
                max_slope = np.float32(0.0)
                step_switch = init_step
                dxpix = 0
                dypix = 0

                for l in range(steps):



                    dxpix += dx_step #* step[precision_flag]
                    dypix += dy_step #* step[precision_flag]

                    #dist_pix = max(abs(dxpix), abs(dypix))

                    # Stop if out of range or map
                    #if dist_pix > total_range:
                    #    break

                    if not (0 <= i + dxpix < W and 0 <= j + dypix < H):
                        break
                    
                    # Height difference
                    dz = height[j + dypix, i + dxpix] - h0

                    if dz > np.float32(0.0):

                        # World-space distance
                        #distx = dxpix * spacing
                        #disty = dypix * spacing

                        #dist = math.sqrt(distx*distx + disty*disty)
                        dist2 = (dxpix*dxpix + dypix*dypix) * spacing_sq

                        # Horizon angle

                        slope = dz * dz / dist2
  
                        if slope > max_slope:
                            max_slope = slope

                        #angle = atan2fast1(dz, dist)
                        
                        #if angle > max_angle:
                        #    max_angle = angle

                    if l >= step_switch:
                        dx_step *= 2
                        dy_step *= 2
                        step_switch += init_step
                    
                sum_angles += math.atan(math.sqrt(max_slope)) #max_angle
            
            # average horizon angle across directions
            avg_angle = sum_angles / 8.0
            # map [0, pi/2] to [1, 0]
            ao[_j, _i] = 1.0 - (avg_angle / (3.14159271 / 2.0))

    #ao_final = ao[y_origin:y_origin + size + 1, x_origin:x_origin + size + 1]
    return ao


@njit(parallel=True, fastmath=True, cache=True)
def apply_ao_to_texture(texture, ao, ao_strength= 1.0):

    tex_size = texture.shape[0]
    ao_size = ao.shape[0]  # 1025
    
    scale = np.float32((ao_size - 1) / (tex_size - 1))
    strength_inv = np.float32(1.0 - ao_strength)
    ao_strength_f = np.float32(ao_strength)
    ao_max_idx = ao_size - 1
    
    for j in prange(tex_size):
        ao_y = j * scale
        y0 = int(ao_y)
        y1 = min(y0 + 1, ao_max_idx)
        fy = ao_y - y0
        fy_inv = 1.0 - fy
        
        for i in range(tex_size):
            ao_x = i * scale
            x0 = int(ao_x)
            x1 = min(x0 + 1, ao_max_idx)
            fx = ao_x - x0
            
            # Bilinear interpolation
            ao_val = (
                ao[y0, x0] * (1.0 - fx) * fy_inv +
                ao[y0, x1] * fx * fy_inv +
                ao[y1, x0] * (1.0 - fx) * fy +
                ao[y1, x1] * fx * fy
            )
            
            ao_val = ao_val * ao_strength_f + strength_inv

            texture[j, i, 0] = np.uint8(min(texture[j, i, 0] * ao_val, 255.0))
            texture[j, i, 1] = np.uint8(min(texture[j, i, 1] * ao_val, 255.0))
            texture[j, i, 2] = np.uint8(min(texture[j, i, 2] * ao_val, 255.0))

@njit(parallel=True, fastmath=True)
def distance_to_positive_numba(height: np.ndarray):

    H, W = height.shape
    inf = 1e20

    max_dim = max(H, W) // 2
    # 1) initialization: zero at height>0, inf elsewhere
    dist = np.empty((H, W), np.float32)
    for j in prange(H):
        for i in range(W):
           dist[j, i] = 0 if height[j, i] > 0.0 else inf

    
    for j in prange(H):
        for i in range(W):
            if dist[j, i] > inf/10:

                n = 1
                found = False
                while n < max_dim and not found:
                    
                    half = n 
                    top = i - half
                    bottom = i + half
                    left = j - half
                    right = j + half

                    for m in range(top, bottom + 1):
                        if m < 0 or m >= H:
                            continue

                        for n in range(left, right + 1):
                            
                            if n < 0 or n >= W:
                                continue
                            if dist[n, m] == 0:
                                dist[j, i] = n
                                found = True
                                break
                            else:
                                continue

                        if found:
                            break

                    n += 1

    return height
                
@njit(parallel=True, fastmath=True, cache=True)
def ao_map_transform(ao_map, min_b=0, c_exp=1.75, c_max=0.85):

    H, W = ao_map.shape
    

    
    for j in prange(H):
        for i in range(W):
            
            # t^c / (t^c + (1-t)^c) contrast curve
            t = ao_map[j, i]

            if t > c_max:
                ao_map[j, i] = 1.0
            else:
                t = t / c_max
                tc = t ** c_exp
                math = (tc + (1.0 - t) ** c_exp) 
                if math > 0:
                    curve = tc / math * (1.0 - min_b)
                    ao_map[j, i] = min_b + curve 
                else:
                    ao_map[j, i] = 1.0

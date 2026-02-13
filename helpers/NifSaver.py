
import pyffi.spells.nif.modify
import pyffi.formats.nif
import struct
import os
import numpy as np
import logging
import triangle
from numba import njit

def SaveGeometryIntoNif(final_verts, final_triangles, quad, folder, file_name, vertex_valid):         
    nif = pyffi.formats.nif.NifFormat.Data(version=int(0x14000005), user_version=0)
    nif.header._version_value_._value = int(0x14000005)
    nif.header.endian_type = 1
    #nif.header.version = int(0x14000005)
    nitristrip = pyffi.formats.nif.NifFormat.NiTriShape()
    nitristrip.flags = 14
    nitristrip.data = pyffi.formats.nif.NifFormat.NiTriShapeData()
    nitristrip.data.num_vertices = len(final_verts[vertex_valid.astype(bool)])
    nitristrip.data.num_triangles = len(final_triangles)
    nitristrip.data.num_triangle_points  = 3 *  len(final_triangles)
    nitristrip.data.extra_vectors_flags = 16 #UV1 | Tangents
    nitristrip.data.num_uv_sets = 1
    nitristrip.data.has_vertices = True
    nitristrip.data.has_triangles = True

    for vertice in final_verts[vertex_valid.astype(bool)]:
        temp_Vector3 = pyffi.formats.nif.NifFormat.Vector3()
        temp_Vector3.x = vertice[0] + quad[0] * 65536.0 * 2
        temp_Vector3.y = vertice[1] + quad[1] * 65536.0 * 2
        temp_Vector3.z = vertice[2]
        nitristrip.data.vertices.append(temp_Vector3)

    for _triangle in final_triangles:
        temp_Triangle = pyffi.formats.nif.NifFormat.Triangle()
        temp_Triangle.v_1 = _triangle[0]
        temp_Triangle.v_2 = _triangle[1]
        temp_Triangle.v_3 = _triangle[2]
        nitristrip.data.triangles.append(temp_Triangle)

    nitristrip.data.uv_sets.append(pyffi.object_models.xml.array._ListWrap(pyffi.formats.nif.NifFormat.TexCoord))

    for uv in final_verts[vertex_valid.astype(bool)]:
        temp_UV =  pyffi.formats.nif.NifFormat.TexCoord()
        temp_UV.u = min(max(uv[0] / 131072, 0.0001), 0.9999)
        temp_UV.v = min(max(uv[1] / 131072, 0.0001), 0.9999)
        nitristrip.data.uv_sets[0].append(temp_UV)

    nitristrip.data.center.x = 65536.0 + quad[0] * 65536.0 * 2
    nitristrip.data.center.y = 65536.0 + quad[1] * 65536.0 * 2
    nitristrip.data.center.z = (np.min(final_verts[vertex_valid.astype(bool)][:,2]) + np.max(final_verts[vertex_valid.astype(bool)][:,2]))/2
    nitristrip.data.radius = 1/2 * np.sqrt( (np.max(final_verts[vertex_valid.astype(bool)][:,2]) 
                                            - np.min(final_verts[vertex_valid.astype(bool)][:,2])) ** 2 +
                                        2 * 4 * 65536 ** 2 )

    nif.roots.append(nitristrip)
    if not os.path.exists(os.path.join(folder, r'meshes\landscape\lod')):
        os.makedirs(os.path.join(folder, r'meshes\landscape\lod'))
    new_stream = open((os.path.join(folder, r'meshes\landscape\lod', file_name + '.nif')), 'wb')
    nif.write(new_stream)
    new_stream.close()

def compare_verts_before_after(verts_before, verts_after, tolerance=0.001):

    before_set = set()
    for i in range(len(verts_before)):
        before_set.add((round(verts_before[i, 0], 3), round(verts_before[i, 1], 3)))
    
    after_set = set()
    for i in range(len(verts_after)):
        after_set.add((round(verts_after[i, 0], 3), round(verts_after[i, 1], 3)))
    
    added = after_set - before_set
    removed = before_set - after_set
    
    print(f"Verts before: {len(verts_before)}")
    print(f"Verts after:  {len(verts_after)}")
    print(f"Added by triangulation:   {len(added)}")
    print(f"Removed by triangulation: {len(removed)}")
    
    if added:
        print(f"Added coords (first 10): {list(added)[:10]}")
    if removed:
        print(f"Removed coords (first 10): {list(removed)[:10]}")
    
    return {
        'before_count': len(verts_before),
        'after_count': len(verts_after),
        'added': added,
        'removed': removed}

def show_added_border_verts(verts_before, verts_after, border_axis, border_value, tolerance=0.001):

    
    before_border = verts_before[np.abs(verts_before[:, border_axis] - border_value) < tolerance]
    after_border = verts_after[np.abs(verts_after[:, border_axis] - border_value) < tolerance]
    
    other_axis = 1 - border_axis
    before_coords = set(round(v, 3) for v in before_border[:, other_axis])
    after_coords = set(round(v, 3) for v in after_border[:, other_axis])
    
    added = after_coords - before_coords
    
    if added:
        print(f"Added to border: {sorted(added)}")
    else:
        print("No vertices added to border")
    
    return added

@njit
def get_height_at_xy(x, y, worldspace_heightmap, x_low, y_low):
#given full worldspace array and the cell id of the top left corner
#returns z coordinate at x, y

    local_x = (x / 4096.0) - x_low #in cells
    local_y = (y / 4096.0) - y_low

    map_x = local_x * 32 #in vert grid
    map_y = local_y * 32

    ix = int(map_x)
    iy = int(map_y)

    fx = map_x - ix
    fy = map_y - iy

    max_x = worldspace_heightmap.shape[1] - 2
    max_y = worldspace_heightmap.shape[0] - 2

    if ix > max_x or iy > max_y:
        return -65000.0
    if ix < 0 or iy < 0:
        return -65000.0


    #bilinear interpolation
    z00 = worldspace_heightmap[iy, ix]
    z10 = worldspace_heightmap[iy, ix + 1]
    z01 = worldspace_heightmap[iy + 1, ix]
    z11 = worldspace_heightmap[iy + 1, ix + 1]

    z0 = z00 * (1 - fx) + z10 * fx
    z1 = z01 * (1 - fx) + z11 * fx
    z = z0 * (1 - fy) + z1 * fy

    return z



def GenerateNifs(mesh_data, worldspace, worldspace_heightmap, x_low, y_low, folder, form_id):
    
    '''
    def get_border_segments(verts, border_mask, border_axis):
        other_axis = 1 - border_axis
        border_indices = np.nonzero(border_mask)[0]
        
        sorted_order = np.argsort(verts[border_indices, other_axis])
        sorted_indices = border_indices[sorted_order]
        
        segments = []
        for i in range(len(sorted_indices) - 1):
            segments.append([sorted_indices[i], sorted_indices[i + 1]])
        
        return segments
    '''

    for x_quad in  mesh_data:
        for y_quad in mesh_data[x_quad]:

            #print('Quad ', (x_quad, y_quad))
            quad = (x_quad, y_quad)
            file_name = f'{form_id}.{(x_quad*32):02}.{(y_quad*32):02}.32'
            logging.info(f"Saving {worldspace} {quad}: {file_name}.nif...")
            pairs = []

            rounded_ar = np.rint(mesh_data[x_quad][y_quad][0][:, :2]).astype(int)
            border = mesh_data[x_quad][y_quad][2]

            sorted_y_index = np.lexsort((rounded_ar[:, 0], rounded_ar[:, 1]))
            sorted_x_index = np.lexsort((rounded_ar[:, 1], rounded_ar[:, 0]))

            for k in range(len(sorted_y_index)):
                idx = sorted_y_index[k]
                if border[idx]:
                    y_ = rounded_ar[idx, 1]
                    if y_ % 4096 != 0:
                        continue
                    for i in range(k + 1, len(sorted_y_index)):
                        idx_ = sorted_y_index[i]
                        if rounded_ar[idx_, 1] != y_:
                            break
                        if abs(rounded_ar[idx_, 0] - rounded_ar[idx, 0]) > 4097:
                            break
                        if border[idx_]:
                            pairs.append((idx, idx_))
                            break

            for k in range(len(sorted_x_index)):
                idx = sorted_x_index[k]
                if border[idx]:
                    x_ = rounded_ar[idx, 0]
                    if x_ % 4096 != 0:
                        continue
                    for i in range(k + 1, len(sorted_x_index)):
                        idx_ = sorted_x_index[i]
                        if rounded_ar[idx_, 0] != x_:
                            break
                        if abs(rounded_ar[idx_, 1] - rounded_ar[idx, 1]) > 4097:
                            break
                        if border[idx_]:
                            pairs.append((idx, idx_))
                            break


            pairs = np.array(pairs)
            '''
            verts = mesh_data[x_quad][y_quad][0][:, :2]
            for border_axis, border_value in [(0, 0.0), (0, 131072.0), (1, 0.0), (1, 131072.0)]:
                border_mask = np.abs(verts[:, border_axis] - border_value) < 0.001
                border_segs = get_border_segments(verts, border_mask, border_axis)
                if border_segs:
                    print(border_segs)
                    pairs = np.concatenate([pairs, np.array(border_segs)], axis=0) if len(pairs) > 0 else np.array(border_segs)'''

            angle = 15
            while True:
                #tris = triangle.triangulate({'vertices':mesh_data[x_quad][y_quad][0][:, :2], 'segments':pairs}, f'p')
                
                verts_before = mesh_data[x_quad][y_quad][0][:, :2].copy()  # Save BEFORE


                tris = triangle.triangulate({'vertices':mesh_data[x_quad][y_quad][0][:, :2], 'segments':pairs}, f'pYq{int(angle)}')

                verts_after = tris['vertices']

                
                #show_added_border_verts(verts_before, verts_after, border_axis=0, border_value=0.0)

                if len(tris['triangles'])  < 65500:
                    break
                angle *= 0.8
                if angle > 2:
                    print("Triangulation failed, trying again with angle ", int(angle))
                else:
                    print("High-=quality triangulation failed, consider reducing verts target")
                    tris = triangle.triangulate({'vertices':mesh_data[x_quad][y_quad][0][:, :2], 'segments':pairs}, f'pY')
                    break
                

                                
        

            len_verts = len(mesh_data[x_quad][y_quad][0])
            len_verts_post_tri = len(tris['vertices'])
            updated_verts = np.zeros((len_verts_post_tri, 3), dtype=np.float64)
            updated_verts[:len_verts] = mesh_data[x_quad][y_quad][0]
            updated_verts[len_verts:,:2] = tris['vertices'][len_verts:]
            for i in range(len_verts, len_verts_post_tri):
                #print(updated_verts[i, 0] + x_quad * 131072.0, updated_verts[i, 1] + y_quad * 131072.0)
                updated_verts[i, 2] = get_height_at_xy(updated_verts[i, 0] + x_quad * 131072.0, updated_verts[i, 1] + y_quad * 131072.0, worldspace_heightmap, x_low, y_low)
            #print(updated_verts[i, 2])

            SaveGeometryIntoNif(updated_verts, tris['triangles'], (x_quad, y_quad), folder, file_name, np.full(len(updated_verts), True, dtype=bool))





def Vector3_fast_init(self, template = None, argument = None, parent = None):
    float_object1 = pyffi.object_models.common.Float()
    float_object2 = pyffi.object_models.common.Float()
    float_object3 = pyffi.object_models.common.Float()
    self._x_value_ = float_object1
    self._y_value_ = float_object2
    self._z_value_ = float_object3
    self._items = [float_object1, float_object2, float_object3]

def Triangle_fast_init(self, template = None, argument = None, parent = None):
    short_object1 = pyffi.object_models.common.UShort()
    short_object2 = pyffi.object_models.common.UShort()
    short_object3 = pyffi.object_models.common.UShort()
    self._v_1_value_ = short_object1
    self._v_2_value_ = short_object2
    self._v_3_value_ = short_object3
    self._items = [short_object1, short_object2, short_object3]

def TexCoord_fast_init(self, template = None, argument = None, parent = None):
    float_object1 = pyffi.object_models.common.Float()
    float_object2 = pyffi.object_models.common.Float()
    self._u_value_ = float_object1
    self._v_value_ = float_object2
    self._items = [float_object1, float_object2]

def Vector3_fast_write(self, stream, data):
    self._x_value_.write(stream, data)
    self._y_value_.write(stream, data)
    self._z_value_.write(stream, data)
def Triangle_fast_write(self, stream, data):
    self._v_1_value_.write(stream, data)
    self._v_2_value_.write(stream, data)
    self._v_3_value_.write(stream, data)
def TexCoord_fast_write(self, stream, data):
    self._u_value_.write(stream, data)
    self._v_value_.write(stream, data)

def Vector3_get_x_value(self):
    return self._x_value_._value
def Vector3_set_x_value(self, value):
    self._x_value_._value = value

def Vector3_get_y_value(self):
    return self._y_value_._value
def Vector3_set_y_value(self, value):
    self._y_value_._value = value

def Float_fast_init(self, **kwargs):
    self._value = 0.0




def UShort_fast_init(self, **kwargs):
    self._value = 0

ushort_struct = struct.Struct('<H')
def UShort_fast_write(self, stream, data):
    
    stream.write(ushort_struct.pack(self._value))

float_struct = struct.Struct('<f')

def Float_fast_read(self, stream, data):
    self._value = float_struct.unpack(stream.read(4))[0]


def Float_fast_write(self, stream, data):
    try:
        stream.write(float_struct.pack(self._value))
    except OverflowError:
        logger = logging.getLogger("pyffi.object_models")
        logger.warn("float value overflow, writing NaN")
        stream.write(struct.pack(data._byte_ord + 'I',
                                     0x7fc00000))



def Triangle_get_v_1_value(self):
    return self._v_1_value_._value
def Triangle_set_v_1_value(self, value):
    self._v_1_value_._value = value

def Triangle_get_v_2_value(self):
    return self._v_2_value_._value
def Triangle_set_v_2_value(self, value):
    self._v_2_value_._value = value

def Triangle_get_v_3_value(self):
    return self._v_3_value_._value
def Triangle_set_v_3_value(self, value):
    self._v_3_value_._value = value


def TexCoord_get_u_value(self):
    return self._u_value_._value
def TexCoord_set_u_value(self, value):
    self._u_value_._value = value

def TexCoord_get_v_value(self):
    return self._v_value_._value
def TexCoord_set_v_value(self, value):
    self._v_value_._value = value
def log_struct_fast(self, stream, attr):
    pass

def Vector3_fast_read(self, stream, data):
    self.x = self._x_value_
    self.y = self._y_value_
    self.z = self._z_value_
    self.x.arg = self.y.arg = self.z.arg = None
    self.x.read(stream, data)
    self.y.read(stream, data)
    self.z.read(stream, data)

def Triangle_fast_read(self, stream, data):
    self.v_1 = self._v_1_value_
    self.v_2 = self._v_2_value_
    self.v_3 = self._v_3_value_
    self.v_1.arg = self.v_2.arg = self.v_3.arg = None
    self.v_1.read(stream, data)
    self.v_2.read(stream, data)
    self.v_3.read(stream, data)

def apply_pyffi_patches():

    setattr(pyffi.formats.nif.NifFormat.TexCoord, 'v', property(TexCoord_get_v_value, TexCoord_set_v_value))
    setattr(pyffi.formats.nif.NifFormat.TexCoord, 'u', property(TexCoord_get_u_value, TexCoord_set_u_value))
    setattr(pyffi.formats.nif.NifFormat.Triangle, 'v_3', property(Triangle_get_v_3_value, Triangle_set_v_3_value))
    setattr(pyffi.formats.nif.NifFormat.Triangle, 'v_2', property(Triangle_get_v_2_value, Triangle_set_v_2_value))
    setattr(pyffi.formats.nif.NifFormat.Triangle, 'v_1', property(Triangle_get_v_1_value, Triangle_set_v_1_value))
    pyffi.object_models.common.Float.write = Float_fast_write
    pyffi.object_models.common.Float.read = Float_fast_read
    pyffi.object_models.xml.struct_.StructBase._log_struct = log_struct_fast
    pyffi.formats.nif.NifFormat.Vector3.read = Vector3_fast_read
    pyffi.formats.nif.NifFormat.Triangle.read = Triangle_fast_read
    pyffi.object_models.common.UShort.__init__ = UShort_fast_init
    pyffi.object_models.common.Float.__init__ = Float_fast_init
    setattr(pyffi.formats.nif.NifFormat.Vector3, 'y', property(Vector3_get_y_value, Vector3_set_y_value))
    pyffi.formats.nif.NifFormat.Vector3.__init__ = Vector3_fast_init
    pyffi.formats.nif.NifFormat.Triangle.__init__ = Triangle_fast_init
    pyffi.formats.nif.NifFormat.TexCoord.__init__ = TexCoord_fast_init
    pyffi.formats.nif.NifFormat.Vector3.write = Vector3_fast_write
    pyffi.formats.nif.NifFormat.Triangle.write = Triangle_fast_write
    pyffi.formats.nif.NifFormat.TexCoord.write = TexCoord_fast_write 
    setattr(pyffi.formats.nif.NifFormat.Vector3, 'x', property(Vector3_get_x_value, Vector3_set_x_value))

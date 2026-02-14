
import struct
import numpy as np
from numba import prange, njit, types
import zlib
import logging

class Subrecord:
    def __init__(self, sig, size, data, has_size=True, **kwargs):
        self.sig = sig   # str 4 bytes
        self.size = size # uint16
        self.data = data # raw bytes
        self.has_size = has_size
        self.form_id = None

    def serialize(self):
        if self.has_size:
            header = struct.pack('<4sH', self.sig.encode('utf-8'), self.size)
            return header + self.data
        else:  
            return self.sig.encode('utf-8') + self.data
        
class SubrecordBTXT(Subrecord):
    __slots__ = ['form_id', 'quadrant', 'sig']

    def __init__(self, sig, size, data, **kwargs):
        self.sig = sig
        self.form_id = None
        self.quadrant = None

class SubrecordATXT(Subrecord):
    __slots__ = ['form_id', 'quadrant', 'layer', 'sig']

    def __init__(self, sig, size, data, **kwargs):
        self.sig = sig
        self.form_id = None
        self.quadrant = None
        self.layer = None

class SubrecordVTEX(Subrecord):
    __slots__ = ['form_id', 'sig']

    def __init__(self, sig, size, data, **kwargs):
        self.sig = sig
        self.form_id = None

class SubrecordVTXT(Subrecord):
    __slots__ = ['point_data', 'sig']

    def __init__(self, sig, size, data, **kwargs):
        self.sig = sig
        self.point_data = None

class SubrecordVCLR(Subrecord):
    __slots__ = ['color_data', 'sig']

    def __init__(self, sig, size, data, **kwargs):
        self.sig = sig
        self.color_data = None

class SubrecordVNML(Subrecord):
    __slots__ = ['sig', 'normal_data']

    def __init__(self, sig, size, data, **kwargs):
        self.sig = sig
        self.normal_data = None

class SubrecordVHGT(Subrecord):
    __slots__ = ['height_data', 'sig', 'height_offset']

    def __init__(self, sig, size, data, **kwargs):
        self.sig = sig
        self.height_data = None
        self.height_offset = None

@njit
def VHGTParserHelper(offset, height_data, return_full_grid = True):

    grid_to_fill = np.zeros((33, 33), dtype=np.float32)
    for i in range(1089):
        row = i // 33
        col = i % 33

        if col == 0:
            row_offset = 0.0
            offset += np.float32(height_data[i]) * 8
        else:
            row_offset += np.float32(height_data[i]) * 8
        if (row > 0 and col > 0) or return_full_grid:
            grid_to_fill[row][col] = offset + row_offset

    return grid_to_fill

class Record:

    FLAG_ESM = 0x00000001
    FLAG_DELETED = 0x00000020
    FLAG_CASTS_SHADOWS = 0x00000200
    FLAG_PERSISTENT = 0x00000400
    FLAG_DISABLED = 0x00000800
    FLAG_IGNORED = 0x00001000
    FLAG_VISIBLE_WHEN_DISTANT = 0x00008000
    FLAG_DANGEROUS = 0x00020000
    FLAG_COMPRESSED = 0x00040000
    FLAG_CANT_WAIT = 0x00080000 

    def __init__(self, sig, data_size, flags, form_id, vc_info, data, parent_group, **kwargs):
        #logging.debug('Creating a record')
        self.sig = sig  # str 4 bytes
        self.data_size = data_size      # uint32
        self.flags = flags              # uint32
        self.form_id = form_id          # uint32
        self.vc_info = vc_info          # uint32
        self.data = data                # raw bytes
        self.subrecords = []
        self.parent_group = parent_group

        if self.is_compressed(): 
            data = zlib.decompress(data[4:])          
            self.parse_subrecords(data)
        else:
            self.parse_subrecords(data)

    def is_compressed(self):
        return (self.flags & self.FLAG_COMPRESSED) != 0
    

    def parse_subrecords(self, data):
        offset = 0

        while offset < len(data):
            sig = data[offset:offset+4].decode('utf-8')
            if sig == 'OFST':
                break #actually don't need it
                # OFST subrecord: no size field, consumes remaining data
                sub_data = data[offset+4:]
                subrecord = Subrecord(sig, len(sub_data), sub_data, has_size=False)
                self.subrecords.append(subrecord)
                offset = len(data) 
                break
            
            size = struct.unpack_from('<H', data, offset+4)[0]
            sub_data = data[offset+6:offset+6+size]
            subrecord = Subrecord(sig, size, sub_data)
            self.subrecords.append(subrecord)
            offset += 6 + size  # 6 bytes header + data



    def serialize(self):

        if self.is_compressed():
            header = struct.pack('<4sIIII', self.sig.encode('utf-8'),
                self.data_size, self.flags, self.form_id,
                self.vc_info)
            return header + self.data
        else:

            subrecords_data = b''.join(sub.serialize() for sub in self.subrecords)
            self.data_size = len(subrecords_data)
            header = struct.pack('<4sIIII', self.sig.encode('utf-8'),
                self.data_size, self.flags, self.form_id,
                self.vc_info)
            return header + subrecords_data
        
    def renumber_formids(self, formid_chg_map, formid_map):
        formid_bytes = bytearray(self.form_id.to_bytes(4, 'big'))
        mod_id = formid_bytes[0]
        if mod_id >= len(formid_chg_map):
            logging.warning(f'HITME record detected! Local FormID: {hex(self.form_id)} \n' 
                            'HITMEs most commonly occur when a master was improperly removed. The behavior of these plugins is undefined and may lead to them not working correctly or causing CTDs.')
            mod_id = len(formid_chg_map) - 1
        if formid_chg_map[mod_id] != mod_id:
            old_formid = self.form_id
            formid_bytes[0] = formid_chg_map[mod_id]
            self.form_id = int.from_bytes(formid_bytes, 'big')
            if old_formid in formid_map:
                formid_map.pop(old_formid)
            else:
                logging.warning(f'Warning: FormID {hex(self.form_id)} is used twice. \n'
                                'The behavior of plugins with such records is undefined and may lead to them not working correctly or causing CTDs.')
            formid_map[self.form_id] = self

    def get_matching_formid(self, form_id, formid_chg_map):
        formid_bytes = bytearray(form_id.to_bytes(4, 'big'))
        mod_id = formid_bytes[0]
        if mod_id >= len(formid_chg_map):
            logging.warning(f'HITME record detected! Local FormID: {hex(form_id)} \n' 
                            'HITMEs most commonly occur when a master was improperly removed. The behavior of these plugins is undefined and may lead to them not working correctly or causing CTDs.')
            mod_id = len(formid_chg_map) - 1
        if formid_chg_map[mod_id] != mod_id:
            formid_bytes[0] = formid_chg_map[mod_id]
            return int.from_bytes(formid_bytes, 'big')
        else:
            return form_id
        
    
class RecordTES4(Record):
    def __init__(self, sig, data_size, flags, form_id, vc_info, data, parent_group, **kwargs):
        self.master_files = []
        super().__init__(sig, data_size, flags, form_id, vc_info, data, parent_group, **kwargs)
        

    def parse_subrecords(self, data):
        super().parse_subrecords(data)
        self.master_files = []
        for subrecord in self.subrecords:
            if subrecord.sig == 'MAST':
                self.master_files.append(subrecord.data.decode('utf-8').rstrip('\x00'))
                #print('masterfile:', self.master_files[-1])
        #print(self.master_files)

class RecordREFR(Record):
    def __init__(self, sig, data_size, flags, form_id, vc_info, data, parent_group, parent_worldspace, **kwargs):
        self.position = None
        self.rotation = None
        self.scale = None
        self.parentformid = None
        self.stateoppositeofparent_flag = None
        self.baserecordformid = None
        self.parent_worldspace = None
        super().__init__(sig, data_size, flags, form_id, vc_info, data, parent_group, **kwargs)
        self.parent_worldspace = parent_worldspace

        
    def parse_subrecords(self, data):
        super().parse_subrecords(data)
        for subrecord in self.subrecords:
            if subrecord.sig == 'NAME':
                self.baserecordformid = struct.unpack_from('<I', subrecord.data)[0]
            if subrecord.sig == 'DATA':
                pos_x, pos_y, pos_z, rot_x, rot_y, rot_z = struct.unpack('<6f', subrecord.data[:24])
                self.position = (pos_x, pos_y, pos_z)
                self.rotation = (rot_x, rot_y, rot_z)
            if subrecord.sig == 'XSCL':    
                self.scale = struct.unpack_from('<f', subrecord.data)
                #print(self.scale)
            if subrecord.sig == 'XESP':
                self.parentformid, self.stateoppositeofparent_flag = struct.unpack_from('<II', subrecord.data)
        self.subrecords = None #clean memory a bit, we won't need raw subrecords

    def renumber_formids(self, formid_chg_map, formid_map):
        super().renumber_formids(formid_chg_map, formid_map)
        #note that the original XESP bytes are not modified
        if self.parentformid:
            formid_bytes = bytearray(self.parentformid.to_bytes(4, 'big'))
            mod_id = formid_bytes[0]
            if mod_id >= len(formid_chg_map):
                logging.warning(f'HITME record detected! Local FormID: {hex(self.form_id)} \n' 
                                'HITMEs most commonly occur when a master was improperly removed. The behavior of these plugins is undefined and may lead to them not working correctly or causing CTDs.')
                mod_id = len(formid_chg_map) - 1
            if formid_chg_map[mod_id] != mod_id:
                formid_bytes[0] = formid_chg_map[mod_id]
                self.parentformid = int.from_bytes(formid_bytes, 'big')
        if self.baserecordformid:
            formid_bytes = bytearray(self.baserecordformid.to_bytes(4, 'big'))
            mod_id = formid_bytes[0]
            if mod_id >= len(formid_chg_map):
                logging.warning(f'HITME record detected! Local FormID: {hex(self.form_id)} \n' 
                                'HITMEs most commonly occur when a master was improperly removed. The behavior of these plugins is undefined and may lead to them not working correctly or causing CTDs.')
                mod_id = len(formid_chg_map) - 1
            if formid_chg_map[mod_id] != mod_id:
                formid_bytes[0] = formid_chg_map[mod_id]
                self.baserecordformid = int.from_bytes(formid_bytes, 'big')

    def is_disabled(self):
        return (((self.flags & (Record.FLAG_DISABLED | Record.FLAG_IGNORED | Record.FLAG_DELETED)) != 0) \
            or (self.stateoppositeofparent_flag == 1 and self.parentformid == 20)) #20 = 14h is the formid of the player
    

    def get_parent_formid(self):
        for subrecord in self.subrecords:
            if subrecord.sig == 'XESP':
                return subrecord.parentformid, subrecord.stateoppositeofparent_flag
        return None, False

class RecordSTAT(Record):
    def __init__(self, sig, data_size, flags, form_id, vc_info, data, parent_group, **kwargs):
        self.model_filename = None
        super().__init__(sig, data_size, flags, form_id, vc_info, data, parent_group, **kwargs)
        

    def parse_subrecords(self, data):
        super().parse_subrecords(data)
        for subrecord in self.subrecords:
            if subrecord.sig == 'MODL':                
                self.model_filename = subrecord.data.decode('windows-1252').rstrip('\x00')


class RecordACTI(Record):
    def __init__(self, sig, data_size, flags, form_id, vc_info, data, parent_group, **kwargs):
        self.model_filename = None
        super().__init__(sig, data_size, flags, form_id, vc_info, data, parent_group, **kwargs)
        

    def parse_subrecords(self, data):
        super().parse_subrecords(data)
        for subrecord in self.subrecords:
            if subrecord.sig == 'MODL':                
                self.model_filename = subrecord.data.decode('windows-1252').rstrip('\x00')

class RecordTREE(Record):
    def __init__(self, sig, data_size, flags, form_id, vc_info, data, parent_group, **kwargs):
        self.model_filename = None
        
        super().__init__(sig, data_size, flags, form_id, vc_info, data, parent_group, **kwargs)
        

    def parse_subrecords(self, data):
        super().parse_subrecords(data)
        for subrecord in self.subrecords:
            if subrecord.sig == 'MODL':
                self.model_filename = subrecord.data.decode('windows-1252').rstrip('\x00')

class RecordWRLD(Record):
    def __init__(self, sig, data_size, flags, form_id, vc_info, data, parent_group, **kwargs):
        self.editor_id = None
        super().__init__(sig, data_size, flags, form_id, vc_info, data, parent_group, **kwargs)
        
        
    def parse_subrecords(self, data):
        super().parse_subrecords(data)
        self.parent_worldspace = None
        self.full_name = None
        for subrecord in self.subrecords:
            if subrecord.sig == 'EDID':
                self.editor_id = subrecord.data.decode('windows-1252').rstrip('\x00')
            elif subrecord.sig == 'WNAM':
                self.parent_worldspace = struct.unpack_from('<I', subrecord.data)[0]
            elif subrecord.sig == 'DATA':
                self.worldspace_flags = struct.unpack_from('B', subrecord.data)[0]
            elif subrecord.sig == 'FULL':
                self.full_name = subrecord.data.decode('windows-1252').rstrip('\x00')


class RecordCELL(Record):
    def __init__(self, sig, data_size, flags, form_id, vc_info, data, parent_group, parent_worldspace, **kwargs):
        self.parent_worldspace = parent_worldspace
        self.cell_coordinates = None
        self.water_type = None
        self.water_level = None
        super().__init__(sig, data_size, flags, form_id, vc_info, data, parent_group, **kwargs)
        

    def parse_subrecords(self, data):
        super().parse_subrecords(data)
        for subrecord in self.subrecords:
            if subrecord.sig == 'XCLC': #cell coordinates
                self.cell_coordinates = struct.unpack_from('<ii', subrecord.data)
            elif subrecord.sig == 'XCWT': #water type, formid
                self.water_type = struct.unpack_from('<I', subrecord.data)[0]
            elif subrecord.sig == 'XCLW': #water level, float
                self.water_level = struct.unpack_from('<f', subrecord.data)[0]

class RecordLAND(Record):
    def __init__(self, sig, data_size, flags, form_id, vc_info, data, parent_group, **kwargs):
        self.heightmap = []
        self.normals = []
        super().__init__(sig, data_size, flags, form_id, vc_info, data, parent_group, **kwargs)

    def renumber_formids(self, formid_chg_map, formid_map):
        super().renumber_formids(formid_chg_map, formid_map)
        for subrecord in self.subrecords:
           
            if subrecord.sig == 'BTXT' or subrecord.sig == 'ATXT':
                #print(subrecord.sig)
                subrecord.form_id = self.get_matching_formid(subrecord.form_id, formid_chg_map)
            elif subrecord.sig == 'VTEX':
                subrecord.form_id = [self.get_matching_formid(form_id, formid_chg_map) for form_id in subrecord.form_id]

    def parse_subrecords(self, data):
        
        self.land_data = []
        offset = 0       
        while offset < len(data):
            
            
            sig = data[offset:offset+4].decode('utf-8')
            size = struct.unpack_from('<H', data, offset+4)[0]

            #if sig != 'VHGT':
            #    offset += 6 + size
            #    continue

            sub_data = data[offset+6:offset+6+size]

            subrecord = None

            if sig == 'BTXT':
                subrecord = SubrecordBTXT(sig, size, None)
                subrecord.form_id, subrecord.quadrant, *_ = struct.unpack("<IB3B", sub_data)

            if sig == 'ATXT':
                subrecord = SubrecordATXT(sig, size, None)
                subrecord.form_id, subrecord.quadrant, _, subrecord.layer = struct.unpack("<IBBH", sub_data)

            if sig == 'VTEX':
                subrecord = SubrecordVTEX(sig, size, None)
                subrecord.form_id = struct.unpack(f"< {len(sub_data) // 4}I", sub_data)

            if sig == 'VTXT':
                subrecord = SubrecordVTXT(sig, size, None)
                
                
                vtxt_dtype = np.dtype([
                    ("position", "<u2"),   
                    ("_pad",     "V2"),    # skip 2 bytes
                    ("opacity",  "<f4"),   
                ])

                records = np.frombuffer(sub_data, dtype=vtxt_dtype)
                subrecord.point_data = records[["position", "opacity"]]
                records = None

                '''
                subrecord.point_data = []
                vtxt_offset = 0
                while vtxt_offset + 7 < len(sub_data):
                    subrecord.point_data.append(struct.unpack("<H2Bf", sub_data[vtxt_offset:vtxt_offset+8]))
                    vtxt_offset += 8'''

            if sig == 'VCLR':
                subrecord = SubrecordVCLR(sig, size, None)

                subrecord.color_data = np.frombuffer(sub_data, dtype=np.uint8).reshape(-1, 3)

                '''
                subrecord.color_data = []
                vclr_offset = 0
                while vclr_offset + 2 < len(sub_data):
                    subrecord.color_data.append(struct.unpack("<3B", sub_data[vclr_offset:vclr_offset+3]))
                    vclr_offset += 3'''
                
            if sig == 'VNML':
                subrecord = SubrecordVNML(sig, size, None)
                subrecord.normal_data = np.frombuffer(sub_data, dtype=np.int8).reshape(-1, 3) #need to check how z is stored
            
            if sig == 'VHGT':
                subrecord = SubrecordVHGT(sig, size, None)
                subrecord.height_offset = struct.unpack('<f', sub_data[:4])[0]
                subrecord.height_data = np.frombuffer(sub_data[4:], dtype=np.int8)

                    
            #subrecord.data = None
            if subrecord:
                self.subrecords.append(subrecord)

            offset += 6 + size  # 6 bytes header + data

        self.data = None

    def parse_land_data_VGHT_only(self):

        
        height_data = np.full((32, 32), -65000.0, dtype=np.float32)

        for subrecord in self.subrecords:
           

            if subrecord.sig == 'VHGT':



                offset = subrecord.height_offset * 8

                grid = subrecord.height_data

                height_data = VHGTParserHelper(offset, grid)

                '''
                for i in range(1089):
                    row = i // 33
                    col = i % 33

                    if col == 0:
                        row_offset = 0.0
                        offset += np.float32(grid[i]) * 8
                    else:
                        row_offset += np.float32(grid[i]) * 8
                    if row > 0 and col > 0:
                        height_data[row - 1][col - 1] = offset + row_offset
                '''


        return height_data    
        #self.texture_id_points = texture_id_points
        #self.opacity_points = opacity_points
        #self.vertex_colors = vertex_colors

    def parse_land_data(self):
        
        texture_id_points = np.zeros((32, 32, 9), dtype=np.uint32)
        opacity_points = np.zeros((32, 32, 9), dtype=np.float32)
        vertex_colors = np.full((32, 32, 3), 255, dtype=np.uint8) 
        normal_data = np.zeros((32, 32, 3), dtype=np.uint8)
        height_data = np.full((32, 32), -65000.0, dtype=np.float32)
        #texture_id_points = [[[0] * 9 for _ in range(32)] for _ in range(32)]
        #opacity_points = [[[0.0] * 9 for _ in range(32)] for _ in range(32)]
        #vertex_colors = [[(0, 0, 0)] * 32 for _ in range(32)]

        for subrecord in self.subrecords:
            if subrecord.sig == 'BTXT':
                if subrecord.quadrant == 0:
                    offset = (0, 0)
                elif subrecord.quadrant == 1:
                    offset = (0, 16)
                elif subrecord.quadrant == 2:
                    offset = (16, 0)
                elif subrecord.quadrant == 3:
                    offset = (16, 16)
                for i in range(16):
                    for j in range(16):
                        texture_id_points[i+offset[0]][j+offset[1]][0] = subrecord.form_id
                        opacity_points[i+offset[0]][j+offset[1]][0] = 1.0

            if subrecord.sig == 'ATXT':
                if subrecord.quadrant == 0:
                    offset = (0, 0)
                elif subrecord.quadrant == 1:
                    offset = (0, 16)
                elif subrecord.quadrant == 2:
                    offset = (16, 0)
                elif subrecord.quadrant == 3:
                    offset = (16, 16)
                texture_id = subrecord.form_id
                texture_layer = subrecord.layer
            
            if subrecord.sig == 'VTXT':
                if not texture_id:
                    continue

                if texture_layer is None:
                    continue

                if texture_layer < 0 or texture_layer > 7:
                    continue
                
                positions = subrecord.point_data["position"]
                opacities  = subrecord.point_data["opacity"]

                row = positions // 17 #vectorized
                col = positions % 17

                mask = (row != 0) & (col != 0)

                i = (row[mask] - 1) + offset[0]
                j = (col[mask] - 1) + offset[1]
                i_layer = texture_layer + 1

                texture_id_points[i, j, i_layer] = texture_id
                opacity_points[i, j, i_layer] = opacities[mask]
                
                '''
                for point in subrecord.point_data:
                    i = point[0] // 17 #vertical
                    j = point[0] % 17 #horizontal
                    if i == 0 or j == 0: #skip left/down border as it duplicates with the right/top border of the previous quad
                        continue
                    i -= 1
                    j -= 1
                    opacity = point[3]
                    #print(i+offset[0], j+offset[1], texture_layer)
                    texture_id_points[i+offset[0]][j+offset[1]][texture_layer+1] = texture_id
                    opacity_points[i+offset[0]][j+offset[1]][texture_layer+1] = opacity
                '''

            if subrecord.sig == 'VCLR':
                grid = subrecord.color_data.reshape(33, 33, 3)
                vertex_colors[:32, :32] = grid[1:, 1:]

            if subrecord.sig == 'VNML':
                grid = subrecord.normal_data.reshape(33, 33, 3)
                normal_data[:32, :32] = grid[1:, 1:]

            if subrecord.sig == 'VHGT':

                offset = subrecord.height_offset * 8

                grid = subrecord.height_data
                height_data = VHGTParserHelper(offset, grid)

                '''
                for i in range(1089):
                    row = i // 33
                    col = i % 33

                    if col == 0:
                        row_offset = 0.0
                        offset += np.float32(grid[i]) * 8
                    else:
                        row_offset += np.float32(grid[i]) * 8
                    if row > 0 and col > 0:
                        height_data[row - 1][col - 1] = offset + row_offset'''


            #self.subrecords = None #clean memory a bit, we won't need raw subrecords

        return texture_id_points, opacity_points, vertex_colors, normal_data, height_data    
        #self.texture_id_points = texture_id_points
        #self.opacity_points = opacity_points
        #self.vertex_colors = vertex_colors


    def parse_heightmap(self):
        heightmap = [[0] * 33 for _ in range(33)]
        cell_offset = None
        for subrecord in self.subrecords:
            if subrecord.sig == 'VHGT':
                cell_offset = struct.unpack('<f', subrecord.data[:4])[0] 
                gradient_data = struct.unpack('1089b', subrecord.data[4:4+1089])
            if subrecord.sig == 'VNML':
                normal_data = struct.unpack('3267b', subrecord.data)

        if not cell_offset:
            return False
        
        offset = cell_offset * 8
        for i in range(1089):
            row = i // 33
            col = i % 33

            if col == 0:
                row_offset = 0
                offset += gradient_data[i] * 8
            else:
                row_offset += gradient_data[i] * 8

            heightmap[row][col] = offset + row_offset

        return heightmap


class RecordLTEX(Record):
    def __init__(self, sig, data_size, flags, form_id, vc_info, data, parent_group, **kwargs):
        self.texture_filename = None
        super().__init__(sig, data_size, flags, form_id, vc_info, data, parent_group, **kwargs)
        

    def parse_subrecords(self, data):
        super().parse_subrecords(data)
        for subrecord in self.subrecords:
            if subrecord.sig == 'ICON':
                self.texture_filename = subrecord.data.decode('windows-1252').rstrip('\x00')


class RecordUseless(Record):
    def __init__(self, sig, data_size, flags, form_id, vc_info, data, parent_group, **kwargs):
        #print('Creating a record')
        self.sig = sig  # str 4 bytes
        if sig == 'REFR':
            self.sig = 'REFU'
        self.data_size = data_size      # uint32
        #self.flags = flags              # uint32
        self.form_id = form_id          # uint32
        #self.vc_info = vc_info          # uint32
        #self.data = None                # raw bytes
        #self.subrecords = []

        
    def parse_subrecords(self, data):
        pass



class Group:
    def __init__(self, sig, group_size, label, typeid, version, parent_group, records, parent_worldspace,**kwargs):
        self.sig = sig                  # 4 bytes
        self.size = group_size          # uint32
        self.label = label              # uint32
        self.typeid = typeid            # uint32
        self.version = version          # uint32
        self.records = records          # list
        self.parent_worldspace = parent_worldspace          # list of worldspace records
        self.parent_group = parent_group

    def serialize(self):
        content = b''.join(r.serialize() for r in self.records)
        group_size = len(content) + 20  # 24 bytes for the group header
        header = struct.pack('<4sIIII', b'GRUP', group_size, self.label,
                            self.typeid, self.version)
        return header + content
    
    def renumber_formids(self, formid_chg_map, formid_map):
        for record in self.records:
            record.renumber_formids(formid_chg_map, formid_map)

class ESPParser:
    def __init__(self):
        self.records = []               # Top-level records and groups
        self.formid_map = {}            # FormID to Record mapping
        self.load_order = []
        self.exterior_cell_list = []
        self.last_cell = None
        self.waterplanes = []

    def parse(self, filename):
        with open(filename, 'rb') as f:
            self._parse_data(f)       
    
    def _parse_data(self, f):
        while True:
            offset = f.tell()
            record_type_b = f.read(4)
            if not record_type_b or len(record_type_b) < 4:
                break
            record_type = record_type_b.decode('utf-8')
            if record_type == 'GRUP':
                group_size_bytes = f.read(4)
                if not group_size_bytes or len(group_size_bytes) < 4:
                    break
                group_size = struct.unpack('<I', group_size_bytes)[0]
                header_b = f.read(12)
                if not header_b or len(header_b) < 12:
                    break
                label, group_type, vc_info = struct.unpack('<III', header_b)
                group = Group(record_type, group_size, label, group_type, vc_info, None, [], None)
                group_end = offset + group_size
                self._parse_group(f, group_end, group)
                self.records.append(group)
                f.seek(group_end)
            else:
                f.seek(offset)
                record = self._parse_record(f, self)
                self.records.append(record)
                self.formid_map[record.form_id] = record
                f.seek(offset + 20 + record.data_size)  # 20 bytes header + data
    
    def _parse_group(self, f, end, group):
        while f.tell() < end:
            offset = f.tell()
            record_type_b = f.read(4)
            if not record_type_b or len(record_type_b) < 4:
                break
            record_type = record_type_b.decode('utf-8')
            if record_type == 'GRUP':
                group_size_bytes = f.read(4)
                group_size = struct.unpack('<I', group_size_bytes)[0]
                label_b = f.read(12)
                label, group_type, vc_info = struct.unpack('<III', label_b)
                subgroup = Group(record_type, group_size, label, group_type, vc_info, group, [], group.parent_worldspace)
                group_end = offset + group_size 
                self._parse_group(f, group_end, subgroup)
                group.records.append(subgroup)
                f.seek(group_end)
            else:
                f.seek(offset) 
                record = self._parse_record(f, group)
                if record.sig == 'WRLD':
                    group.parent_worldspace = record
                group.records.append(record)
                self.formid_map[record.form_id] = record
                f.seek(offset + 20 + record.data_size)  # 20 bytes header + data
                
    def _parse_record(self, f, parent_group):
        offset = f.tell()
        header_b = f.read(20)
        if not header_b or len(header_b) < 20:
            return None
        header = struct.unpack('<4sIIII', header_b)
        record_type = header[0].decode('utf-8')
        data_size = header[1]
        flags = header[2]
        form_id = header[3]
        vc_info = header[4]
        if record_type in ('REFR1', 'STAT', 'TREE', 'WRLD', 'TES4', 'ACHR', 'ACRE', 'CELL', 'ACTI', 'LAND', 'LTEX'):
            record_data = f.read(data_size)
        #if record_type == 'REFR' and parent_group.parent_worldspace:
        #    return RecordREFR(record_type, data_size, flags, form_id, vc_info, record_data, parent_group, parent_group.parent_worldspace)
        #if record_type == 'STAT':
        #    return RecordSTAT(record_type, data_size, flags, form_id, vc_info, record_data, parent_group)
        #elif record_type == 'ACTI':
        #    return RecordACTI(record_type, data_size, flags, form_id, vc_info, record_data, parent_group)
        #elif record_type == 'TREE':
        #    return RecordTREE(record_type, data_size, flags, form_id, vc_info, record_data, parent_group)
        if record_type == 'WRLD':
            return RecordWRLD(record_type, data_size, flags, form_id, vc_info, record_data, parent_group)
        elif record_type == 'TES4':
            TES4Record = RecordTES4(record_type, data_size, flags, form_id, vc_info, record_data, parent_group)
            for i, master in enumerate(TES4Record.master_files):
                self.load_order.append([i, master])
            return TES4Record
        elif record_type == 'CELL':
            cell_record = RecordCELL(record_type, data_size, flags, form_id, vc_info, record_data, parent_group, parent_group.parent_worldspace)
            if cell_record.cell_coordinates:
                self.exterior_cell_list.append([parent_group.parent_worldspace, cell_record.cell_coordinates, cell_record, parent_group])
            self.last_cell = cell_record
            return cell_record
        elif record_type == 'LAND' and self.last_cell:
            return RecordLAND(record_type, data_size, flags, form_id, vc_info, record_data, parent_group)
        #elif record_type in ('ACHR', 'ACRE'):
        #    return Record(record_type, data_size, flags, form_id, vc_info, record_data, parent_group)
        elif record_type == 'LTEX':
            return RecordLTEX(record_type, data_size, flags, form_id, vc_info, record_data, parent_group)
        else:
            return RecordUseless(record_type, data_size, flags, form_id, vc_info, None, parent_group)
        #everything else not needed for LOD
            

    def find_record_by_formid(self, formid):
        return self.formid_map.get(formid)
    
    def reconstruct(self, filename):
        with open(filename, 'wb') as f:
            for item in self.records:
                f.write(item.serialize())

    def renumber_formids(self, formid_chg_map):
        for record in self.records:
            record.renumber_formids(formid_chg_map, self.formid_map)




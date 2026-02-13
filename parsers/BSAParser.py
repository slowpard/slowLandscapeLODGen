import logging
import struct
import os 
import zlib

class BSAParser(): 

    FLAG_HAS_DIRECTORY_NAMES = 0x00000001   
    FLAG_HAS_FILE_NAMES = 0x00000002        
    FLAG_COMPRESSED = 0x00000004            
    FLAG_UNKNOWN2 = 0x00000008              
    FLAG_UNKNOWN3 = 0x00000010              
    FLAG_UNKNOWN4 = 0x00000020              
    FLAG_BIG_ENDIAN = 0x00000040            
    FLAG_UNKNOWN5 = 0x00000080              
    FLAG_UNKNOWN6 = 0x00000100             
    FLAG_UNKNOWN7 = 0x00000200             
    FLAG_UNKNOWN8 = 0x00000400              

    def __init__(self):
        self.files = {}
        self.flags = None
        self.folder_count = None
        self.files_count = None
        self.compressed = None
        self.filename = None
        self.full_path_in_block = None

    def parse(self, filename):
        self.filename = filename
        with open(filename, 'rb') as f:
            data = f.read(3000000) #around first 3MB, an overkill for all but some theoretical bsas
        self._parse_data(data)



    def CalculateHash(self, fileName):
        """Returns tes4's two hash values for filename.
        Based on TimeSlips code with cleanup and pythonization."""
        """Source: https://en.uesp.net/wiki/Oblivion_Mod:Hash_Calculation#Python"""
        root,ext = os.path.splitext(fileName.lower()) #--"bob.dds" >> root = "bob", ext = ".dds"
        #--Hash1
        chars = [ord(x) for x in root] #map(ord,root) #--'bob' >> chars = [98,111,98]

        if len(chars) > 1:
            hash1 = chars[-1] | (0,chars[-2])[len(chars)>2]<<8 | len(chars)<<16 | chars[0]<<24
        else:
            hash1 = chars[-1] | (0,chars[-1])[len(chars)>2]<<8 | len(chars)<<16 | chars[0]<<24
        
        #--(a,b)[test] is similar to test?a:b in C. (Except that evaluation is not shortcut.)
        if   ext == '.kf':  hash1 |= 0x80
        elif ext == '.nif': hash1 |= 0x8000
        elif ext == '.dds': hash1 |= 0x8080
        elif ext == '.wav': hash1 |= 0x80000000
        #--Hash2
        #--Python integers have no upper limit. Use uintMask to restrict these to 32 bits.
        uintMask, hash2, hash3 = 0xFFFFFFFF, 0, 0 
        for char in chars[1:-2]: #--Slice of the chars array
            hash2 = ((hash2 * 0x1003f) + char ) & uintMask
        for char in (ord(x) for x in ext): #--map(ord,ext)
            hash3 = ((hash3 * 0x1003F) + char ) & uintMask
        hash2 = (hash2 + hash3) & uintMask
        #--Done
        return (hash2<<32) + hash1 #--Return as uint64
    

    def _parse_data(self, data):
        #36 bytes
        magic_number, bsa_version, folder_offset, flags, \
        n_folders, n_files, l_foldernames, \
        l_filenames, content_flags = struct.unpack_from('<4sIIIIIIII', data[:36], 0)
        
        if magic_number != b'BSA\x00':
            logging.error('Error: Not a BSA')
            return
        self.flags = flags
        #print(self.flags)
        self.folder_count = n_folders
        self.files_count = n_files
        self.compressed = (self.flags & self.FLAG_COMPRESSED)
        #self.full_path_in_block = (self.flags & self.FLAG_FULL_PATH_IN_BLOCK)  #apparently wrong info in uesp
        

        #filenames
        offset = 36 + n_folders*16 + (l_foldernames + n_folders + 16 * n_files)
        files = {}
        for filename in data[offset:offset+l_filenames].split(b'\x00'):
            try:
                if filename != b'':
                    files[self.CalculateHash(filename.decode('windows-1252'))] = filename.decode('windows-1252')
            except:
                logging.error(f'Error: filename decoding error for {filename}')

        #folders
        for folder_index in range(n_folders):
            folder_hash, n_files, offset = struct.unpack_from('<QII', data, 36 + folder_index*16)
            offset -= l_filenames 
            string_length = data[offset]
            #string_length = int.from_bytes(data[offset], signed=False, byteorder='little')
            folder_path = data[offset + 1:offset + 1 + string_length].decode('windows-1252').rstrip('\x00')
            folder_files = []
            for i in range(n_files):
                try:
                    file_hash, file_size, file_pointer = struct.unpack_from('<QII', data, offset + string_length + 1 + i*16)
                    real_size = file_size & 0x3FFFFFFF  # Mask: 0011 1111 ....
                    compression_flag = (file_size >> 30) & 0x1  # Shift right by 30, and mask with 0x1 (bit 30)
                    file_name = files[file_hash]
                    full_path = os.path.join(folder_path, file_name)
                    self.files[full_path] = (file_pointer, real_size, compression_flag)
                except:
                    logging.error('Error: hashed file not found')


    def get_list_of_files(self):
        return self.files.keys()

    def extract(self, files, output_folder):
        with open(self.filename, 'rb') as f:
            for file in files:
                file = os.path.join(file)
                if file in self.files:
                    offset, size, compression_flag = self.files[file]
                    #if self.full_path_in_block: #apparenly wrong info in uesp
                    #    offset += data[offset] + 1
                    if (self.compressed and compression_flag == 0) or (not self.compressed and compression_flag): 
                        offset += 4
                        f.seek(offset)
                        data = f.read(size)
                        decompressed_file = zlib.decompress(data)
                    else:
                        f.seek(offset)
                        data = f.read(size)
                        decompressed_file = data
                    os.makedirs(os.path.dirname(os.path.join(output_folder, file)), exist_ok=True)
                    output_file = open(os.path.join(output_folder, file), 'wb')
                    output_file.write(decompressed_file)
                    output_file.close()
                    data = None
                else:
                    logging.error(f'Error: file {file} not found')
        
    def pack(self, output_filename, files, root):

        file_info = []
        folders = {}
        total_folder_name_length = 0
        total_file_name_length = 0
        n_folders = 0
        n_files = 0

        hashed_files = {}
        h_f = []

        for file in files:
            file = file.lower()
            full_path = os.path.join(root, file)
            if not os.path.isfile(full_path):
                logging.error(f"File not found: {full_path}")
                continue
            file_size = os.path.getsize(full_path)
            folder, file_name = os.path.split(file)
            file_entry = {
                'full_path': full_path,
                'relative_path': file,
                'folder': folder,
                'file_name': file_name,
                'size': file_size,
                'file_hash': self.CalculateHash(file_name),
                'file_name_bytes': file_name.encode('windows-1252') + b'\x00',
                'file_name_length': len(file_name.encode('windows-1252')) + 1
            }
            
            total_file_name_length += file_entry['file_name_length'] # (TotalFileNameLength) Total length of all file names, including \0's.

            h_f.append(file_entry)
            
            hashed_files[self.CalculateHash(file_name)] = file_entry

        #print("hashes calculated")
        #return h_f
        hashed_files = dict(sorted(hashed_files.items())) #sort by hash or Oblivion won't read it

        for file in hashed_files.values():            
            folder = file['folder']
            if folder not in folders:
                folders[folder] = {
                    'folder_name': folder,
                    'files': []
                }
            folders[folder]['files'].append(file)

        for folder_name, folder_data in folders.items():
            folder_data['folder_hash'] = self.CalculateHash(folder_data['folder_name'])
            folder_name_bytes = folder_data['folder_name'].encode('windows-1252')
            folder_data['folder_name_bytes'] = struct.pack('B', len(folder_name_bytes) + 1) + folder_name_bytes + b'\x00'
            folder_data['folder_name_length'] = len(folder_name_bytes) + 1  
            total_folder_name_length += folder_data['folder_name_length'] #Total length of all folder names, including \0's but not including the prefixed length byte.

        folders = dict(sorted(folders.items(), key=lambda x: x[1]['folder_hash'])) #sort by hash or Oblivion won't read it
        n_folders = len(folders)
        n_files = sum(len(folder_data['files']) for folder_data in folders.values())

        #OFFSETS
        header_size = 36
        folder_records_offset = header_size
        folder_records_size = n_folders * 16
        file_records_offset = folder_records_offset + folder_records_size
        #print(hex(file_records_offset))
        file_records_size = n_files * 16 + total_folder_name_length + n_folders
        file_names_offset = file_records_offset + file_records_size
        #print(hex(file_names_offset))
        file_data_offset = file_names_offset + total_file_name_length
        #print(hex(file_data_offset))
        offset = file_records_offset
        for folder in folders.values():
            folder['name_offset'] = offset + total_file_name_length #uint32 - Offset to name and file records for this folder. (Seems to include Total File Name Length
                                                                    # #:todd_grin:
            offset += folder_data['folder_name_length'] + 16

       
        offset = file_data_offset
        for folder in folders.values():
            for file in folder['files']:
                file['data_absolute_offset'] = offset
                offset += file['size']

        flags = self.FLAG_HAS_DIRECTORY_NAMES | self.FLAG_HAS_FILE_NAMES | self.FLAG_UNKNOWN5 | self.FLAG_UNKNOWN6 | self.FLAG_UNKNOWN7 | self.FLAG_UNKNOWN8
        content_flags = 0x00000000
        for folder_data in folders.values(): #how much perf would we save not doing this?
            for file_entry in folder_data['files']:
                ext = os.path.splitext(file_entry['file_name'].lower())[1]
                if ext == '.nif':
                    content_flags |= 0x00000001
                elif ext == '.lod':
                    content_flags |= 0x00000001
                    content_flags |= 0x00000100
                elif ext == '.dds':
                    content_flags |= 0x00000002
                elif ext == '.xml':
                    content_flags |= 0x00000004
                elif ext == '.wav':
                    content_flags |= 0x00000008
                elif ext == '.mp3':
                    content_flags |= 0x00000010
                elif ext == '.sdp':
                    content_flags |= 0x00000020
                elif ext == '.ctl':
                    content_flags |= 0x00000040
                elif ext == '.fnt':
                    content_flags |= 0x00000080
                else:
                    content_flags |= 0x00000100  

        with open(output_filename, 'wb') as f:
            
            magic_number = b'BSA\x00'
            version = 103 #Oblivion
            f.write(struct.pack(
                '<4sIIIIIIII',
                magic_number,
                version,
                folder_records_offset,
                flags,
                n_folders,
                n_files,
                total_folder_name_length,
                total_file_name_length,
                content_flags
            ))

            '''Folder records
            uint64 - Hash of the folder name (eg: menus[slash]chargen).
            uint32 - (FolderFileCount) Number of files in this folder.
            uint32 - Offset to name and file records for this folder. (Seems to include Total File Name Length'''

            for folder_data in folders.values():
                f.write(struct.pack(
                    '<QII',
                    folder_data['folder_hash'],
                    len(folder_data['files']),
                    folder_data['name_offset']
                ))

            '''Folder name and files
            bzstring - Name of the folder. (Only present if bit 1 of archive flags is set.)
            struct[FolderFileCount] - File records in the given folder.
                uint64 - Hash of the file name (eg: race_sex_menu.xml).
                uint32 - (FileSize) Size of the file data. Note that the top two bits have special meaning and are not actually part of the size.
                If bit 30 is set in the size, the default compression is inverted for this file (i.e., if the default is compressed, this file is not compressed, and vice versa).
                Bit 31 is used internally to determine if an archive has been checked yet.
                uint32 - Offset to raw file data for this folder. Note that an "offset" is offset from file byte zero (start), NOT from this location.
            '''

            for folder_data in folders.values():
                f.write(folder_data['folder_name_bytes'])
                for file_entry in folder_data['files']:
                    f.write(struct.pack(
                        '<QII',
                        file_entry['file_hash'],
                        file_entry['size'],
                        file_entry['data_absolute_offset']
                    ))

            '''File name block. (Only present if bit 2 of the archive flags is set.)
            A list of lower case file names, one after another, each ending in a \0. They are ordered in the same order as those generated with the file folder block contents in the BSA archive.
            These are all the files contained in the archive, such as "cuirass.nif" and "cuirass.dds", etc (no paths, just the root names).'''

            # Write file names
            for folder_data in folders.values():
                for file_entry in folder_data['files']:
                    f.write(file_entry['file_name_bytes'])

            # Write file data blocks
            for folder_data in folders.values():
                for file_entry in folder_data['files']:
                    with open(file_entry['full_path'], 'rb') as file_data:
                        f.write(file_data.read())

        logging.info(f"BSA archive '{output_filename}' created successfully.")
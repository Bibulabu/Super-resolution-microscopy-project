# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

rec_header_dtd = \
    [
        ("nx", "i4"),  # Number of columns
        ("ny", "i4"),  # Number of rows
        ("nz", "i4"),  # Number of sections

        ("mode", "i4"),  # Types of pixels in the image. Values used by IMOD:
        #  0 = unsigned or signed bytes depending on flag in imodFlags
        #  1 = signed short integers (16 bits)
        #  2 = float (32 bits)
        #  3 = short * 2, (used for complex data)
        #  4 = float * 2, (used for complex data)
        #  6 = unsigned 16-bit integers (non-standard)
        # 16 = unsigned char * 3 (for rgb data, non-standard)

        ("nxstart", "i4"),  # Starting point of sub-image (not used in IMOD)
        ("nystart", "i4"),
        ("nzstart", "i4"),

        ("mx", "i4"),  # Grid size in X, Y and Z
        ("my", "i4"),
        ("mz", "i4"),

        ("xlen", "f4"),  # Cell size; pixel spacing = xlen/mx, ylen/my, zlen/mz
        ("ylen", "f4"),
        ("zlen", "f4"),

        ("alpha", "f4"),  # Cell angles - ignored by IMOD
        ("beta", "f4"),
        ("gamma", "f4"),

        # These need to be set to 1, 2, and 3 for pixel spacing to be interpreted correctly
        ("mapc", "i4"),  # map column  1=x,2=y,3=z.
        ("mapr", "i4"),  # map row     1=x,2=y,3=z.
        ("maps", "i4"),  # map section 1=x,2=y,3=z.

        # These need to be set for proper scaling of data
        ("amin", "f4"),  # Minimum pixel value
        ("amax", "f4"),  # Maximum pixel value
        ("amean", "f4"),  # Mean pixel value

        ("ispg", "i4"),  # space group number (ignored by IMOD)
        ("next", "i4"),  # number of bytes in extended header (called nsymbt in MRC standard)
        ("creatid", "i2"),  # used to be an ID number, is 0 as of IMOD 4.2.23
        ("extra_data", "V30"),  # (not used, first two bytes should be 0)

        # These two values specify the structure of data in the extended header; their meaning depend on whether the
        # extended header has the Agard format, a series of 4-byte integers then real numbers, or has data
        # produced by SerialEM, a series of short integers. SerialEM stores a float as two shorts, s1 and s2, by:
        # value = (sign of s1)*(|s1|*256 + (|s2| modulo 256)) * 2**((sign of s2) * (|s2|/256))
        ("nint", "i2"),
        # Number of integers per section (Agard format) or number of bytes per section (SerialEM format)
        ("nreal", "i2"),  # Number of reals per section (Agard format) or bit
        # Number of reals per section (Agard format) or bit
        # flags for which types of short data (SerialEM format):
        # 1 = tilt angle * 100  (2 bytes)
        # 2 = piece coordinates for montage  (6 bytes)
        # 4 = Stage position * 25    (4 bytes)
        # 8 = Magnification / 100 (2 bytes)
        # 16 = Intensity * 25000  (2 bytes)
        # 32 = Exposure dose in e-/A2, a float in 4 bytes
        # 128, 512: Reserved for 4-byte items
        # 64, 256, 1024: Reserved for 2-byte items
        # If the number of bytes implied by these flags does
        # not add up to the value in nint, then nint and nreal
        # are interpreted as ints and reals per section

        ("extra_data2", "V20"),  # extra data (not used)
        ("imodStamp", "i4"),  # 1146047817 indicates that file was created by IMOD
        ("imodFlags", "i4"),  # Bit flags: 1 = bytes are stored as signed

        # Explanation of type of data
        ("idtype", "i2"),  # ( 0 = mono, 1 = tilt, 2 = tilts, 3 = lina, 4 = lins)
        ("lens", "i2"),
        # ("nd1", "i2"),  # for idtype = 1, nd1 = axis (1, 2, or 3)
        # ("nd2", "i2"),
        ("nphase", "i4"),
        ("vd1", "i2"),  # vd1 = 100. * tilt increment
        ("vd2", "i2"),  # vd2 = 100. * starting angle

        # Current angles are used to rotate a model to match a new rotated image.  The three values in each set are
        # rotations about X, Y, and Z axes, applied in the order Z, Y, X.
        ("triangles", "f4", 6),  # 0,1,2 = original:  3,4,5 = current

        ("xorg", "f4"),  # Origin of image
        ("yorg", "f4"),
        ("zorg", "f4"),

        ("cmap", "S4"),  # Contains "MAP "
        ("stamp", "u1", 4),  # First two bytes have 17 and 17 for big-endian or 68 and 65 for little-endian

        ("rms", "f4"),  # RMS deviation of densities from mean density

        ("nlabl", "i4"),  # Number of labels with useful data
        ("labels", "S80", 10)  # 10 labels of 80 charactors
    ]




#-------------------------------------------------------------------------------------------

# RawSIMData_gt is SIM raw 9-frame stack： 502x502x9
# SIM_gt is Ground Truth: 1004x1004x1
# Level 1-9: Noise level decreases to none: 502x502x9
# GT_all is GT of 55 samples: 1004x1004x55



def read_mrc(filename, filetype='image'):

    fd = open(filename, 'rb')
    header = np.fromfile(fd, dtype=rec_header_dtd, count=1)

    nx, ny, nz = header['nx'][0], header['ny'][0], header['nz'][0]
    print(nx,ny,nz)

    if header[0][3] == 1:
        data_type = 'int16'
    elif header[0][3] == 2:
        data_type = 'float32'
    elif header[0][3] == 4:
        data_type = 'single'
        nx = nx * 2
    elif header[0][3] == 6:
        data_type = 'uint16'

    data = np.ndarray(shape=(nx, ny, nz))
    imgrawdata = np.fromfile(fd, data_type)
    fd.close()

    if filetype == 'image':
        for iz in range(nz):
            data_2d = imgrawdata[nx*ny*iz:nx*ny*(iz+1)]
            data[:, :, iz] = data_2d.reshape(nx, ny, order='F')
    else:
        data = imgrawdata

    return header, data


def write_mrc(filename, img_data, header):

    if img_data.dtype == 'int16':
        header[0][3] = 1
    elif img_data.dtype == 'float32':
        header[0][3] = 2
    elif img_data.dtype == 'uint16':
        header[0][3] = 6

    fd = open(filename, 'wb')
    for i in range(len(rec_header_dtd)):
        header[rec_header_dtd[i][0]].tofile(fd)

    nx, ny, nz = header['nx'][0], header['ny'][0], header['nz'][0]
    imgrawdata = np.ndarray(shape=(nx*ny*nz), dtype='uint16')
    
    for iz in range(nz):
        imgrawdata[nx*ny*iz:nx*ny*(iz+1)]=img_data[:,:,iz].reshape(nx*ny, order='F')
    imgrawdata.tofile(fd)

    fd.close()
    return


# view
""" source = 'GT_all.mrc'
source1 = 'Cell_001\RawSIMData_gt.mrc'


import matplotlib.pyplot as plt
import numpy as np

# Call the read_mrc function and pass the filename of the MRC file as an argument
_, data = read_mrc(source)
_, data1 = read_mrc(source1)
# Display the data using matplotlib
fig, axs = plt.subplots(1, 2, figsize=(20, 10))
axs[0].imshow(data[:, :, 0])
axs[1].imshow(data1[:, :, 0])
#axs[2].imshow(data[:, :, 2])
plt.show() """

# save as tif
from PIL import Image
import numpy as np
import os

def save_multi_frame_tiff(filename, image_data):
    # image_data: 3D numpy array with shape (nx, ny, nz)
    # filename: output file path with .tif extension

    # Create empty list to store individual image arrays
    image_list = []

    # Convert each 2D slice into PIL Image and append to the list
    for z in range(image_data.shape[2]):
        image_slice = Image.fromarray(image_data[:, :, z])
        image_slice = image_slice.resize((512, 512), resample=Image.BICUBIC)
        image_list.append(image_slice)

    # Save the list of images as a multi-frame TIFF
    image_list[0].save(filename, format='TIFF', save_all=True, append_images=image_list[1:])

for i in range(1,55):
    if i < 10:
        file_name = os.path.join('Cell_00{:d}/RawSIMData_gt.mrc'.format(i))
        save_name = os.path.join('', "microtubules_raw/00{:d}.tif".format(i))
    else:
        file_name = os.path.join('Cell_0{:d}/RawSIMData_gt.mrc'.format(i))
        save_name = os.path.join('', "microtubules_raw/0{:d}.tif".format(i))

    

    # Read MRC file
    header, image_data = read_mrc(file_name)

    # Save multi-frame TIFF file
    save_multi_frame_tiff(save_name, image_data)


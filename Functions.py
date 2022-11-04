import numpy as np
import os, copy, pydicom
import nibabel as nib
from skimage.measure import label, regionprops
from skimage import measure
from scipy import ndimage as ndi

def load_nifty_image(img_dir):
    img = nib.load(img_dir)
    data = np.array(img.get_data())
    hdr = img.header
    affine = img.affine
    return img, data, hdr, affine


def save_nifty_image(img_dir, data, affine):
    # img = nib.spatialimages.SpatialImage(data, affine)
    # hdr = img.get_header()
    # img.update_header()
    new_image = nib.Nifti1Image(data, affine)#img.get_data()
    nib.save(new_image, img_dir)

def get_patient_number(path):
    word = copy.copy(path)
    pos = []
    for p,ch in enumerate(word):
        if '/' == ch:
            pos.append(p)
    pos = np.max(pos)
    return pos

def get_paths(path,list_dir,ext=".gz"):
    """
    Recursively find files with extension 'ext' inside folder path 'path'.

    @param path: folder path input
    @param list_dir: as input, auxiliar function for recursion
    @param ext: file extension to search
    @return: list of paths to files with input extension
    """
    for f in os.listdir(path):
        path1 = f'{path}/{f}'
        if path1.endswith(ext):
            print(f'File found: {path1}')
            list_dir.append(path1)
        elif os.path.isdir(path1):
            list_dir = get_paths(path1, list_dir, ext)

    return list_dir

def geometry(path_nifti):
    """
    Load nifti image and properties.
    @param path_nifti: path to nifti image '.nii'
    @return:
        affine: affine matrix
        airway: ndarray containing the airway mask. Preprocess is made
                to select the object with the biggest size, avoiding
                disconnected airways to been considered.
        perim: airways perimeter from morphological operations.
    """
    img,airway, hdr, affine = load_nifty_image(path_nifti)

    airway = np.round(airway)
    airway = selMax_vol(airway)
    perim = np.nonzero(BWperim(airway).ravel()==1)[0]
    perim = np.column_stack(np.unravel_index(perim,airway.shape))

    return affine, airway, perim

def BWperim(image,neighbourhood=1):
    """
    Calculate binary mask image perimeter using morphological operations.
    @param image: input mask image.
    @param neighbourhood:
    @return: image containing binary mask perimeter.
    """
    if neighbourhood == 1:
        strel = ndi.generate_binary_structure(3, 1)
    else:
        strel = ndi.generate_binary_structure(3, 2)

    image = image.astype(np.uint8)
    eroded_image = ndi.binary_erosion(image, strel, border_value=0)
    border_image = image - eroded_image

    return border_image

def selMax_vol(BW):
    m, n, d = BW.shape

    BW = ndi.binary_fill_holes(BW)
    L = label(BW)
    BW_stats = regionprops(L)

    volumen = [BW_stats[i].area for i in range(len(BW_stats))]
    volumen = np.asarray(volumen)

    ar = np.arange(0, len(volumen), 1.)

    volumen = ar[volumen == np.max(volumen)]

    BW = np.zeros((m, n, d))
    BW[L == volumen + 1] = 1

    return BW



def load_scan(path):
    slices = [pydicom.dcmread(f'{path}/{s}',force=True) for s in os.listdir(path)]
    slices = [s for s in slices if 'SliceLocation' in s]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2]
                                 -slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation
                                 -slices[1].SliceLocation)

    z_pos = np.inf
    for s in slices:
        s.SliceThickness = slice_thickness
        z_pos = np.min([z_pos, s.ImagePositionPatient[-1]])

    diag = np.hstack([slices[0].PixelSpacing, slice_thickness, 1])
    affine = np.diag(diag)
    affine[:,-1] = np.hstack([slices[0].ImagePositionPatient[0:2], z_pos, 1])

    return slices, affine

def create_dir(file_path):
    """
    Create input directory if it does not exist.

    @param file_path: inpu directory to create
    @return:
    """
    if not os.path.exists(file_path):
        os.makedirs(file_path)
        print(f'Directory {file_path} created')
    else:
        print(f'Directory {file_path} already exist')
    return file_path

def write_scan(image,pathr,pathw):

    create_dir(pathw)

    ds = [pydicom.dcmread(pathr + '/' + s,force=True)
          for s in os.listdir(pathr)]
    ds = [s for s in ds if 'InstanceNumber' in s]

    image = (image-np.min(image.ravel()))\
                   /(np.max(image.ravel())-np.min(image.ravel()))*1000

    m,_,_ = np.shape(image)
    for i,t in enumerate(ds):
        j = ds[i].InstanceNumber-1
        ds[i].PixelData = np.uint16(image[j,:,:])
        ds[i].save_as(pathw + 'CT' + "{number:06}".format(number=j)+'.dcm')

def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    image = image.astype(np.int16)
    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 100

    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)

    return np.array(image, dtype=np.int16)

def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)
    counts = counts[vals != bg]
    vals = vals[vals != bg]
    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

def min_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)
    counts = counts[vals != bg]
    vals = vals[vals != bg]
    if len(counts) > 0:
        return vals[np.argmin(counts)]
    else:
        return None

def segment_lung_mask(image, fill_lung_structures=True):
    # not actually binary, but 1 and 2.
    # 0 is treated as background, which we do not want
    binary_image = np.array(image >= -200, dtype=np.int8) + 1
    labels = measure.label(binary_image)

    # Pick the pixel in the very corner to determine which label is air.
    # Improvement: Pick multiple background labels from around the patient
    # More resistant to “trays” on which the patient lays cutting the air around the
    # person in half

    background_label = labels[0, 0, 0]

    # Fill the air around the person
    binary_image[background_label == labels] = 2

    # Method of filling the lung structures (that is superior to
    # something like morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice-1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)

            if l_max is not None:  # This slice contains some lung
                binary_image[i][labeling != l_max] = 1
    binary_image -= 1  # Make the image actual binary
    binary_image = 1 - binary_image  # Invert it, lungs are now 1

    # Remove other air pockets inside body
    # labels = measure.label(binary_image, background=0)
    # l_max = largest_label_volume(labels, bg=0)
    # if l_max is not None:  # There are air pockets
    #     binary_image[labels != l_max] = 0

    return binary_image





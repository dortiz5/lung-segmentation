import numpy as np
import os, copy, pydicom
import nibabel as nib
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
from skimage import measure
from scipy import ndimage as ndi
import pandas as pd

def load_nifty_image(img_dir):
    img = nib.load(img_dir)
    data = np.array(img.get_data())
    hdr = img.header
    affine = img.affine
    return img, data, hdr, affine

def voxvol_from_affine(affine):
    diag = affine.diagonal()
    return np.prod(diag)

def get_nifti_volume(path, write = True):
    _, lungs_binary, _, affine = load_nifty_image(path)
    lungs_binary, unit_vol= selMax_vol(lungs_binary, 0.6)
    volume = np.sum(unit_vol) * voxvol_from_affine(affine)

    if write:
        save_nifty_image(f'/home/david/Desktop/test_nii', lungs_binary, affine)

    return volume

def calc_strain(vols):
    return ((vols[1]-vols[0])/vols[0])*100

def update_volume_dict(lungs_path, patient_number,
                       volumes, dict = None):

    if dict is None:
        dict = {'patient': [], 'state': [],
                'parenchyma': [], 'airways': []}

    if 'ESP' in lungs_path:
        state = 'expiration'
    elif 'INSP' in lungs_path:
        state = 'inspiration'
    else:
        state = 'relative'

    dict['patient'].append(patient_number)
    dict['state'].append(state)
    dict['parenchyma'].append(volumes[0])
    dict['airways'].append(volumes[1])

    return dict

def write_dataframe(dict, path):
    data_df = pd.DataFrame(dict)
    data_df.to_csv(f'{path}/DB_lung_airway_vols.csv')

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

    list_dir.sort()
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
    airway = selMax_vol(airway)[0]
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

def selMax_vol(BW, th=0.):
    m, n, d = BW.shape

    BW = ndi.binary_fill_holes(BW)
    L = label(BW)
    BW_stats = regionprops(L)

    volumen = [BW_stats[i].area for i in range(len(BW_stats))]
    volumen = np.asarray(volumen)

    ar = np.arange(0, len(volumen), 1.)

    if th == 0:
        vol_indx = ar[volumen == np.max(volumen)]
        volumen_data = volumen[volumen == np.max(volumen)]

    else:
        max_vol_th = np.max(volumen)*th
        vol_indx = ar[volumen >= max_vol_th]
        volumen_data = volumen[volumen >= max_vol_th]

    BW = np.zeros((m, n, d))
    for vol in vol_indx:
        BW[L == vol + 1] = 1

    return BW, volumen_data

def flip_image(img, flip):
    if flip == 'UD':
        img = np.flip(img, axis = 0)
    elif flip == 'LR':
        img = np.flip(img, axis = 1)
    elif flip == 'BF':
        img = np.flip(img, axis = 2)
    return img

def reorient_nifti(img, flip):
    img = flip_image(img, flip)
    return np.transpose(img,(2,1,0))

def clean_spur_voxels(img):
    strel = ndi.generate_binary_structure(3, 1)

    img_dil = ndi.binary_dilation(img, strel,
                                       border_value=0, iterations=2)

    img_ero = ndi.binary_erosion(img_dil, strel,
                                 border_value=0, iterations=2)

    return (img_ero*1).astype(np.uint8)

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
    # image[image == -2000] = 100

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
    binary_image = np.array(image >= -300, dtype=np.int8) + 1
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
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None:  # There are air pockets
        binary_image[labels != l_max] = 0

    return binary_image





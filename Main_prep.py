from Functions import *

#%% LUNGS SEGMENTATION
def lungs_segmentation(patient, state, path_airways):
    pos = get_patient_number(state) + 1
    state_name = state[pos:]
    print(f'Patient state {state_name}')

    # Lungs -------------
    patient_dicom, affine = load_scan(state)
    patient_pixels = get_pixels_hu(patient_dicom)
    lungs_mask = segment_lung_mask(patient_pixels, fill_lung_structures=False)
    lungs_mask_clean = np.asarray(selMax_vol(lungs_mask)[0], dtype=bool)

    # Airways  ----------
    strel = ndi.generate_binary_structure(3, 1)
    airway_nifti_path = [path for path in path_airways if state_name in path
                         and '_cut' not in path and '_LUNGS' not in path]
    airway_binary = load_nifty_image(airway_nifti_path[0])[1]
    airway_binary = reorient_nifti(airway_binary, 'BF')
    airway_binary = ~ndi.binary_dilation(airway_binary, strel,
                                         border_value=0, iterations=2)

    # process images --------
    lungs_noairw = (airway_binary * 1) * (lungs_mask_clean * 1) == 1
    lungs_noairw = clean_spur_voxels(lungs_noairw)

    # Save segmented lung in nifti
    nii_lung_path = f'{write_path}/{patient}'
    create_dir(nii_lung_path)
    nii_lung_path_state = f'{nii_lung_path}/{patient}{state_name}_LUNGS.nii.gz'
    save_nifty_image(nii_lung_path_state, reorient_nifti(lungs_noairw, 'UD'), affine)

    print(f'Saved: {state_name}_LUNGS')


def segmentation_main(dicom_path):
    paths_read_dcm = get_paths(dicom_path, [], '')

    for i, patient in enumerate(paths_read_dcm):

        pos = get_patient_number(patient) + 1
        patient_number = patient[pos:]
        paths_states = get_paths(patient, [], '')

        path_patient_airs = f'{path_airway}/{patient_number}'
        path_patient_airs = get_paths(path_patient_airs, [], '.gz')
        print(f'Patient to process: {patient_number}')

        for state in paths_states:
            lungs_segmentation(patient_number, state, path_patient_airs)

        print(f'Finished patient {patient_number}\n')
    print('Finished.')

#%% LUNGS PREPROCESS

def preprocess_main(inpath, outpath):

    volumes = None

    paths_read_dcm = get_paths(inpath, [], '')
    for i, patient in enumerate(paths_read_dcm):
        pos = get_patient_number(patient) + 1
        patient_number = patient[pos:]
        print(f'Patient to process: {patient_number}')

        path_patient_nifti = get_paths(patient, [], '.gz')

        lungs_nifti_path = [path for path in path_patient_nifti if 'LUNGS' in path]
        airway_nifti_path = [path for path in path_patient_nifti if '_cut' not in path
                             and '_LUNGS' not in path]

        lungs_vols = []
        airways_vols = []
        for lungs_path, airway_path in zip(lungs_nifti_path, airway_nifti_path):

            lungs_vols.append(get_nifti_volume(lungs_path, False))
            airways_vols.append(get_nifti_volume(airway_path, False))
            volumes = update_volume_dict(lungs_path, patient_number,
                               [lungs_vols[-1],airways_vols[-1]], volumes)

        volumes = update_volume_dict('RELATIVE', patient_number,
                                     [calc_strain(lungs_vols), calc_strain(airways_vols)],
                                     volumes)

    write_dataframe(volumes, outpath)


if __name__ == '__main__':

    #%%
    # set path and load files
    # read_path = f'/media/david/61DA856534A7B963/David Ortiz DB/DATA/BDe'
    # f'/home/david/Desktop/BD'
    path_dicoms = f'/media/david/61DA856534A7B963/David Ortiz DB/DATA/BDe/DICOMS'
    path_airway = f'../IGA_code/raw-data/INRS2'
    write_path = f'../IGA_code/raw-data/INRS2'

    segmentation_main(path_dicoms)

    #%%
    out_path = f'../IGA_code/results-data/DATA_AIR/RESULTS'
    preprocess_main(path_airway, out_path)
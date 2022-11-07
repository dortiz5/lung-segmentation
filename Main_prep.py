from Functions import *

# set path and load files
main_path = f'/media/david/61DA856534A7B963/David Ortiz DB/DATA/BDe'#f'/home/david/Desktop/BD'
path_dicoms = f'{main_path}/DICOMS'
path_airway = f'../IGA_code/raw-data/INRS2'

def segmentation():

    paths_read_dcm = get_paths(path_dicoms, [], '')

    for i, patient in enumerate(paths_read_dcm):

        pos = get_patient_number(patient) + 1
        patient_number = patient[pos:]
        paths_states = get_paths(patient, [], '')

        path_patient_airs = f'{path_airway}/{patient_number}'
        path_patient_airs = get_paths(path_patient_airs, [], '.gz')

        print(f'Patient to process: {patient_number}')

        for state in paths_states:
            pos = get_patient_number(state) + 1
            state_name = state[pos:]
            print(f'Patient state {state_name}')

            # Lungs -------------
            patient_dicom, affine = load_scan(state)
            patient_pixels = get_pixels_hu(patient_dicom)
            lungs_mask = segment_lung_mask(patient_pixels,fill_lung_structures = False)
            lungs_mask_clean = np.asarray(selMax_vol(lungs_mask), dtype = bool)

            # Airways  ----------
            strel = ndi.generate_binary_structure(3, 1)
            airway_nifti_path = [path for path in path_patient_airs if state_name in path
                                            and '_cut' not in path]
            airway_binary = load_nifty_image(airway_nifti_path[0])[1]
            airway_binary = reorient_nifti(airway_binary, 'BF')
            airway_binary = ~ndi.binary_dilation(airway_binary, strel,
                                                 border_value = 0, iterations = 2)

            # process images --------
            lungs_noairw = (airway_binary*1)*(lungs_mask_clean*1) == 1
            lungs_noairw = fill_spur_voxels(lungs_noairw)

            nii_lung_path = f'{main_path}/NIFTI/{patient_number}'
            create_dir(nii_lung_path)
            nii_lung_path_state = f'{nii_lung_path}/{patient_number}{state_name}_LUNGS.nii.gz'
            save_nifty_image(nii_lung_path_state, reorient_nifti(lungs_noairw, 'UD'), affine)

            print(f'Saved: {state_name}_LUNGS')
        print(f'Finished patient {patient_number}\n')
    print('Finished.')

def preprocess():

    paths_read_dcm = get_paths(path_airway, [], '')
    for i, patient in enumerate(paths_read_dcm):
        pos = get_patient_number(patient) + 1
        patient_number = patient[pos:]
        print(f'Patient to process: {patient_number}')

        path_patient_nifti = get_paths(patient, [], '.gz')
        lungs_nifti_path = [path for path in path_patient_nifti if 'LUNGS' in path]

        for lungs in lungs_nifti_path:
            lungs_binary = load_nifty_image(lungs)[1]
            lungs_binary = selMax_vol(lungs_binary,0.6)


if __name__ == '__main__':
    # segmentation()
    preprocess()
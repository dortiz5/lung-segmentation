from Functions import *

# set path and load files
main_path = f'/media/david/61DA856534A7B963/David Ortiz DB/DATA/BDe'#f'/home/david/Desktop/BD'
path_read = f'{main_path}/DICOMS'
paths_read = get_paths(path_read, [], '')

for i, patient in enumerate(paths_read):

    pos = get_patient_number(patient) + 1
    patient_number = int(patient[pos:])
    paths_states = get_paths(patient, [], '')
    print(f'Patient to process: {patient_number}')

    for state in paths_states:
        pos = get_patient_number(state) + 1
        state_name = state[pos:]

        print(f'Patient state {state_name}')
        patient_dicom, affine = load_scan(state)
        patient_pixels = get_pixels_hu(patient_dicom)

        #%% Lung segmentation
        lungs_mask = segment_lung_mask(patient_pixels,fill_lung_structures=True)
        lungs_mask_clean = selMax_vol(lungs_mask)

        nii_lung_path = f'{main_path}/NIFTI/{patient_number}'
        create_dir(nii_lung_path)
        nii_lung_path_state = f'{nii_lung_path}/{state_name}_LUNGS'

        save_nifty_image(nii_lung_path_state,
                         np.transpose(np.flipud(lungs_mask),(2,1,0)),
                         affine)

        print(f'Saved: {state_name}_LUNGS')

print('Finished.')


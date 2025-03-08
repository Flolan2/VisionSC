import os
import pandas as pd
import matplotlib.pyplot as plt
from .lib.patients import Patient, PatientCollection


def construct_data(csv_files, fs, labels=None, scaling_factor=1, verbose=True, smooth=None):
    if isinstance(scaling_factor, int):
        scaling_factor = [scaling_factor] * len(csv_files)
    if isinstance(fs, int):
        fs = [fs] * len(csv_files)

    patients = []
    for i, file in enumerate(csv_files):
        # Get filename without extension.
        file_name = os.path.basename(file).split('.')[0]
        
        if verbose:
            print('Loading: {}'.format(file_name))
        
        # Load the CSV file with a multi-index header.
        pose_estimation = pd.read_csv(file, header=[0, 1], index_col=0)
        
        # Enforce that the CSV has a MultiIndex header.
        if not isinstance(pose_estimation.columns, pd.MultiIndex):
            raise ValueError(f"CSV file {file_name} does not have a MultiIndex header. Please update your CSV generation process.")
        
        # Normalize the multi-index columns: lowercase all strings and remove "marker_" prefix if present.
        pose_estimation = normalize_multiindex_columns(pose_estimation)
        
        # --- Debugging block to check for shoulder markers ---
        if verbose:
            shoulder_cols = [col for col in pose_estimation.columns if 'shoulder' in col[0].lower()]
            if shoulder_cols:
                print(f"Found shoulder markers in {file_name}:")
                print(shoulder_cols)
                print("First 2 rows for shoulder markers:")
                print(pose_estimation.loc[:, shoulder_cols].head(2))
                
                # Plot the shoulder marker data
                pose_estimation.loc[:, shoulder_cols].plot(title=f"Shoulder Markers Data for {file_name}")
                plt.show()
                
                # Plot the column names in a separate plot
                plt.figure()
                # Combine the column names into a single string
                col_names_str = '\n'.join([str(col) for col in shoulder_cols])
                plt.text(0.5, 0.5, col_names_str, ha='center', va='center', wrap=True, fontsize=10)
                plt.title("Shoulder Marker Column Names")
                plt.axis('off')
                plt.show()
            else:
                print(f"No shoulder markers found in {file_name}.")
            
            print("First 2 rows after normalizing (all columns retained):")
            print(pose_estimation.head(2))
        
        # Construct a Patient object with the entire pose_estimation DataFrame.
        p = Patient(
            pose_estimation,
            fs[i],
            patient_id=file_name,
            likelihood_cutoff=0,
            label=labels[i] if labels is not None else None,
            low_cut=0,
            high_cut=None,
            clean=True,
            scaling_factor=scaling_factor[i],
            normalize=True,
            spike_threshold=10,
            interpolate_pose=True,
            smooth=smooth,
        )
        patients.append(p)

    # Construct patient collection.
    pc = PatientCollection()
    pc.add_patient_list(patients)
    
    return pc


def normalize_multiindex_columns(data):
    """
    Enforces that data.columns is a MultiIndex. Converts all elements of the MultiIndex to lowercase strings.
    If the first-level element starts with "marker_", that prefix is removed.
    This normalization ensures that downstream processing uses the new standardized keypoint names.
    """
    if not isinstance(data.columns, pd.MultiIndex):
        raise ValueError("Expected data.columns to be a MultiIndex. Received: {}".format(type(data.columns)))
    
    new_tuples = []
    for tup in data.columns:
        # Process the first element: remove "marker_" prefix if present, then lowercase.
        first = str(tup[0]).strip().lower()
        if first.startswith("marker_"):
            first = first[len("marker_"):]
        # Process the rest of the tuple elements.
        rest = tuple(str(x).strip().lower() for x in tup[1:])
        new_tuples.append((first,) + rest)
    data.columns = pd.MultiIndex.from_tuples(new_tuples)
    return data

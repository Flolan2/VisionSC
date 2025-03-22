import os
import pandas as pd
import matplotlib.pyplot as plt
from .lib.patients import Patient, PatientCollection

def construct_data(csv_files, fs, labels=None, scaling_factor=1, verbose=False, smooth=None):
    if isinstance(scaling_factor, int):
        scaling_factor = [scaling_factor] * len(csv_files)
    if isinstance(fs, int):
        fs = [fs] * len(csv_files)

    patients = []
    for i, file in enumerate(csv_files):
        file_name = os.path.basename(file).split('.')[0]
        if verbose:
            print(f'Loading: {file_name}')
        
        # Load CSV file with a multi-index header.
        pose_estimation = pd.read_csv(file, header=[0, 1], index_col=0)
        
        # Enforce that the CSV has a MultiIndex header.
        if not isinstance(pose_estimation.columns, pd.MultiIndex):
            raise ValueError(f"CSV file {file_name} does not have a MultiIndex header. Please update your CSV generation process.")
        
        # Normalize the multi-index columns.
        pose_estimation = normalize_multiindex_columns(pose_estimation)
        
        # Construct a Patient object using the entire pose_estimation DataFrame.
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

    # Construct and return the patient collection.
    pc = PatientCollection()
    pc.add_patient_list(patients)
    return pc


def normalize_multiindex_columns(data):
    """
    Enforces that data.columns is a MultiIndex and converts all elements of the MultiIndex to lowercase strings.
    If the first-level element starts with "marker_", that prefix is removed.
    """
    if not isinstance(data.columns, pd.MultiIndex):
        raise ValueError(f"Expected data.columns to be a MultiIndex. Received: {type(data.columns)}")
    
    new_tuples = []
    for tup in data.columns:
        # Process the first element: remove "marker_" prefix if present, then lowercase.
        first = str(tup[0]).strip().lower()
        if first.startswith("marker_"):
            first = first[len("marker_"):]
        # Process the remaining elements.
        rest = tuple(str(x).strip().lower() for x in tup[1:])
        new_tuples.append((first,) + rest)
    
    data.columns = pd.MultiIndex.from_tuples(new_tuples)
    return data

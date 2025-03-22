import pandas as pd
from itertools import product

parameters = {'num_steps',
              'stride_duration', 
              'stride_duration_asymmetry', 
              'step_duration', 
              'step_duration_asymmetry', 
              'cadence', 
              'terminal_double_support', 
              'terminal_double_support_asymmetry', 
              'initial_double_support', 
              'initial_double_support_asymmetry',
              'double_support', 
              'double_support_asymmetry', 
              'single_limb_support', 
              'single_limb_support_asymmetry', 
              'stance', 
              'stance_asymmetry', 
              'swing', 
              'swing_asymmetry',
              'step_length', 
              'stride_length', 
              'gait_speed'
             }

def prepare_gait_dataframe(sides=['left', 'right']):
    """
    Prepares an empty DataFrame for gait parameters.
    
    Args:
        parameters (set): A set of gait parameters (e.g., stride duration, step duration).
        sides (list): List of sides ('left' and 'right') to calculate parameters for.
    
    Returns:
        pd.DataFrame: An empty multi-index DataFrame with gait parameters and sides.
    """
    # Create multi-level columns: (side, parameter)
    multi_columns = list(product(sides, parameters))
    
    # Initialize empty DataFrame with multi-index columns
    dataframe = pd.DataFrame(columns=multi_columns)
    dataframe.columns = pd.MultiIndex.from_tuples(dataframe.columns, names=['side', 'parameter'])
    
    return dataframe

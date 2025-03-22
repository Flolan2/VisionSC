# Kinematic Analysis

This project analyzes patient data from videos, performs statistical inferences

## Repo Structure
- `src/`: Source code containing the core components of the project.
  - `gait_parameters/`: Code for computing gait parameters.
  - `stats/`: Code for statistical analysis.
  - `ml/`: Code for machine learning models.
- `data/`: Raw and processed data files.
- `docs/`: Documentation

## Quickstart
1. Clone the repository:
   ```bash
   git clone https://github.com/Flolan2/VisionSC

2. Install dependencies:
    ```bash
    pip install -r requirements.txt

3. Run a pipeline.
     For example, to run the `gait_parameters` script for a directory, use the following command:
    <!-- Add arguments to command  -->
    ```bash
    python src/gait_parameters/main.py [--input_dir <input_dir>] [--config <config>] [--output_dir <output_dir>]
    ```
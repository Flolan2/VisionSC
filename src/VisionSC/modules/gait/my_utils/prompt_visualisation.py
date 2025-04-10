import os
import logging
import cv2
import matplotlib.pyplot as plt # Import plt

# Make sure plotting import is correct relative to this file's location
try:
    # If prompt_visualisation is inside my_utils
     from .plotting import display_plot_with_cv2
except ImportError:
     # If prompt_visualisation is somewhere else relative to plotting
     try:
          from modules.gait.my_utils.plotting import display_plot_with_cv2
     except ImportError:
          logging.error("Could not import display_plot_with_cv2. Check relative paths.")
          # Define a dummy function to avoid crashing if import fails
          def display_plot_with_cv2(fig):
               logging.error("display_plot_with_cv2 is not available.")


logger = logging.getLogger(__name__)

def prompt_visualisation(fig, input_file, normal_save_dir, dpi=300):
    """
    Displays the provided matplotlib figure interactively using OpenCV and prompts
    the user for validation ('g'ood/'b'ad). Saves the figure appropriately and
    renames the input file if rejected.

    Parameters:
      - fig: Matplotlib figure to display and save.
      - input_file: Path of the original file being processed (used for naming and potential renaming).
      - normal_save_dir: Directory where the figure should be saved if approved.
      - dpi: DPI setting for saving the figure.

    Returns:
      A tuple (approved, new_file_path) where:
        - approved is True if the user accepts the figure, False otherwise.
        - new_file_path is the potentially renamed path of the input_file if rejected, else None.
    """
    # Check if the figure exists
    if fig is None:
        logger.error("No figure provided for validation. Skipping prompt.")
        # Decide on return value - let's assume it's not approved if no figure
        return False, None

    # Ensure the normal save directory exists
    os.makedirs(normal_save_dir, exist_ok=True)

    # Derive base filename
    filename_base = os.path.splitext(os.path.basename(input_file))[0]
    plot_filename = f"{filename_base}_combined_extremas_toe.png" # Standard name for this plot

    # Display the figure interactively using the imported helper
    display_plot_with_cv2(fig) # This function now waits for key press

    # Prompt the user for validation *after* the plot window is closed.
    while True:
        user_input = input(f"-> Validation for '{filename_base}': Is the event detection plot correct? (g=good, b=bad): ").lower().strip()
        if user_input in ['g', 'b']:
            break
        else:
            print("   Invalid input. Please enter 'g' or 'b'.")

    approved = (user_input == 'g')
    new_file_path = None # Initialize

    if approved:
        # Save the approved figure to the normal directory
        final_save_path = os.path.join(normal_save_dir, plot_filename)
        try:
             # Use the _save_plot logic (or replicate it here)
             os.makedirs(os.path.dirname(final_save_path), exist_ok=True)
             fig.savefig(final_save_path, dpi=dpi, bbox_inches='tight')
             logger.info(f"Validation approved. Figure saved to: {final_save_path}")
        except Exception as e:
             logger.error(f"Failed to save approved plot {plot_filename}: {e}", exc_info=True)
    else:
        # Save the rejected figure to a 'skipped' subfolder
        skipped_save_dir = os.path.join(normal_save_dir, "skipped")
        os.makedirs(skipped_save_dir, exist_ok=True)
        final_save_path = os.path.join(skipped_save_dir, plot_filename)
        try:
            os.makedirs(os.path.dirname(final_save_path), exist_ok=True)
            fig.savefig(final_save_path, dpi=dpi, bbox_inches='tight')
            logger.warning(f"Validation rejected. Figure saved to: {final_save_path}")
        except Exception as e:
             logger.error(f"Failed to save rejected plot {plot_filename}: {e}", exc_info=True)

        # Rename the original input file to mark it as skipped
        input_dir = os.path.dirname(input_file)
        base, ext = os.path.splitext(os.path.basename(input_file))
        # Avoid double-skipping
        if not base.endswith("_skipped"):
             new_base = f"{base}_skipped"
             new_file_path = os.path.join(input_dir, f"{new_base}{ext}")
             try:
                 os.rename(input_file, new_file_path)
                 logger.warning(f"Input file '{os.path.basename(input_file)}' renamed to '{os.path.basename(new_file_path)}'")
             except OSError as e:
                 logger.error(f"Failed to rename skipped input file {input_file} to {new_file_path}: {e}")
                 new_file_path = input_file # Keep original path if rename fails
        else:
            logger.info(f"Input file '{os.path.basename(input_file)}' was already marked as skipped.")
            new_file_path = input_file


    # Explicitly close the figure object passed to the function
    plt.close(fig)

    return approved, new_file_path # Return approval status and the new path (or original if rename failed/not needed)
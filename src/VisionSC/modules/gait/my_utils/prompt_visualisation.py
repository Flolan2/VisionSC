import os
import logging
import cv2

logger = logging.getLogger(__name__)

def prompt_visualisation(fig, input_file, normal_save_dir, dpi=300):
    """
    Displays the provided matplotlib figure interactively and then prompts the user
    for validation. If the user indicates the figure is good, the figure is saved
    to the normal_save_dir. If the user indicates the figure is bad, the figure is
    saved to a 'skipped' subfolder of normal_save_dir and the input file is renamed
    with a "_skipped" suffix.
    
    Parameters:
      - fig: Matplotlib figure to display.
      - input_file: Path of the file being processed.
      - normal_save_dir: Directory where the figure should be saved if approved.
      - dpi: DPI setting for saving the figure.
      
    Returns:
      A tuple (approved, new_file) where:
        - approved is True if the user accepts the figure, or False if not.
        - new_file is None if approved, or the new (renamed) file path if not.
    """
    # Check if the figure exists
    if fig is None:
        logger.error("No figure available for saving. Skipping visualization.")
        return False, input_file

    # Ensure the normal save directory exists
    os.makedirs(normal_save_dir, exist_ok=True)

    # Save a temporary copy of the figure before displaying it.
    temp_save_path = os.path.join(normal_save_dir, "temp_figure.png")
    fig.savefig(temp_save_path, dpi=dpi)
    
    # Display the figure interactively using our updated in-memory method.
    from my_utils.plotting import display_plot_with_cv2
    display_plot_with_cv2(fig)
    
    # Prompt the user for validation.
    user_input = input("Does the combined extremas and toe figure look correct? (g=good, b=bad): ")
    
    # Close any OpenCV windows to prevent an extra (empty) window from lingering.
    cv2.destroyAllWindows()
    
    filename = os.path.basename(input_file).split('.')[0]
    
    if user_input.lower() == 'g':
        # If approved, rename the temp file to the final filename.
        final_save_path = os.path.join(normal_save_dir, f"{filename}_combined_extremas_toe.png")
        os.rename(temp_save_path, final_save_path)
        logger.info("Figure saved to %s", final_save_path)
        return True, None
    else:
        # If not approved, save to the 'skipped' folder.
        skipped_save_dir = os.path.join(normal_save_dir, "skipped")
        os.makedirs(skipped_save_dir, exist_ok=True)
        final_save_path = os.path.join(skipped_save_dir, f"{filename}_combined_extremas_toe.png")
        os.rename(temp_save_path, final_save_path)
        logger.info("Figure saved to %s", final_save_path)
        # Rename the input file to mark it as skipped.
        base, ext = os.path.splitext(input_file)
        new_file = f"{base}_skipped{ext}"
        os.rename(input_file, new_file)
        logger.info("Input file %s renamed to %s", input_file, new_file)
        return False, new_file

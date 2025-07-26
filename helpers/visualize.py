import numpy as np
import cv2
import os
from datetime import datetime
from PIL import Image
import glob


def save_rgb_observation_to_png(rgb_observation, output_path, filename=None):
    """
    Save RGB observation to PNG file.
    
    Args:
        rgb_observation: numpy array of shape (H, W, 3) with values in range [0, 255]
        output_path: directory path where to save the PNG file
        filename: optional filename, if None will generate timestamp-based name
    
    Returns:
        str: path to the saved PNG file
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"rgb_observation_{timestamp}.png"
        
        # Ensure filename has .png extension
        if not filename.endswith('.png'):
            filename += '.png'
        
        # Convert to uint8 if needed
        if rgb_observation.dtype != np.uint8:
            if rgb_observation.max() <= 1.0:
                rgb_observation = (rgb_observation * 255).astype(np.uint8)
            else:
                rgb_observation = rgb_observation.astype(np.uint8)
        
        # Convert BGR to RGB if needed (OpenCV uses BGR)
        if len(rgb_observation.shape) == 3 and rgb_observation.shape[2] == 3:
            rgb_observation = cv2.cvtColor(rgb_observation, cv2.COLOR_BGR2RGB)
        
        # Save the image
        full_path = os.path.join(output_path, filename)
        cv2.imwrite(full_path, rgb_observation)
        
        print(f"RGB observation saved to: {full_path}")
        return full_path
        
    except Exception as e:
        print(f"Error saving RGB observation: {e}")
        return None


def create_gif_from_pngs(png_directory, output_gif_path, duration=500, loop=0):
    """
    Create GIF from PNG files in a directory.
    
    Args:
        png_directory: directory containing PNG files
        output_gif_path: path where to save the GIF file
        duration: duration of each frame in milliseconds (default: 500ms)
        loop: number of loops, 0 means loop forever (default: 0)
    
    Returns:
        str: path to the saved GIF file, or None if error
    """
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_gif_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Find all PNG files in the directory
        png_pattern = os.path.join(png_directory, "*.png")
        png_files = glob.glob(png_pattern)
        
        if not png_files:
            print(f"No PNG files found in directory: {png_directory}")
            return None
        
        # Sort files to ensure consistent order
        png_files.sort()
        
        # Load all images
        images = []
        for png_file in png_files:
            try:
                img = Image.open(png_file)
                images.append(img)
                print(f"Loaded: {png_file}")
            except Exception as e:
                print(f"Error loading {png_file}: {e}")
                continue
        
        if not images:
            print("No valid images loaded")
            return None
        
        # Save as GIF
        images[0].save(
            output_gif_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=loop
        )
        
        print(f"GIF created successfully: {output_gif_path}")
        print(f"Total frames: {len(images)}")
        return output_gif_path
        
    except Exception as e:
        print(f"Error creating GIF: {e}")
        return None


def create_gif_from_rgb_observations(rgb_observations, output_gif_path, duration=500, loop=0):
    """
    Create GIF directly from a list of RGB observations (numpy arrays).
    
    Args:
        rgb_observations: list of numpy arrays of shape (H, W, 3) with values in range [0, 255]
        output_gif_path: path where to save the GIF file
        duration: duration of each frame in milliseconds (default: 500ms)
        loop: number of loops, 0 means loop forever (default: 0)
    
    Returns:
        str: path to the saved GIF file, or None if error
    """
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_gif_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        if not rgb_observations:
            print("No RGB observations provided")
            return None
        
        # Convert numpy arrays to PIL Images
        images = []
        for i, rgb_obs in enumerate(rgb_observations):
            try:
                # Convert to uint8 if needed
                if rgb_obs.dtype != np.uint8:
                    if rgb_obs.max() <= 1.0:
                        rgb_obs = (rgb_obs * 255).astype(np.uint8)
                    else:
                        rgb_obs = rgb_obs.astype(np.uint8)
                
                # Convert BGR to RGB if needed (OpenCV uses BGR)
                if len(rgb_obs.shape) == 3 and rgb_obs.shape[2] == 3:
                    rgb_obs = cv2.cvtColor(rgb_obs, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL Image
                img = Image.fromarray(rgb_obs)
                images.append(img)
                print(f"Processed observation {i+1}/{len(rgb_observations)}")
                
            except Exception as e:
                print(f"Error processing observation {i+1}: {e}")
                continue
        
        if not images:
            print("No valid images processed")
            return None
        
        # Save as GIF
        images[0].save(
            output_gif_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=loop
        )
        
        print(f"GIF created successfully: {output_gif_path}")
        print(f"Total frames: {len(images)}")
        return output_gif_path
        
    except Exception as e:
        print(f"Error creating GIF: {e}")
        return None

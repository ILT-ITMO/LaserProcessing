import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Profile():
    """
    A class for processing and analyzing profile data.
    """


  def auto_profile(profile, full_data):
    """
    Identifies and characterizes regions of interest within profile data.
    
    This method compares a profile dataset to a full dataset to pinpoint areas
    where the profile's 'z' values are notably higher than the median 'z'
    value of the full dataset. It then determines the width and maximum
    depth of these regions, returning them as a dictionary of parameters.
    This helps to highlight areas of significant variation or activity within the profile.
    
    Args:
        profile (pd.DataFrame): The profile data to analyze, expected to have 'x' and 'z' columns.
        full_data (pd.DataFrame): The full dataset used to calculate the median 'z' value, expected to have 'z' column.
    
    Returns:
        dict: A dictionary where keys are region numbers (starting from 1) and
            values are tuples containing the width (in mm) and depth of each
            identified region.  Returns an empty dictionary if no regions are found.
    """

    mediana = np.median(full_data['z'])
    print(mediana)

    result = []
    start = 0
    for i in range(len(profile)):
      if profile['z'].iloc[i] - mediana > mediana*5:
        if start == 0:
          start = i
      else:
        if start != 0:
          end = i
          result.append((start, end))
          start = 0

    parametrs = {}
    for key, pair in enumerate(result):
      width = float(round(profile.iloc[pair[1]]['x'] - profile.iloc[pair[0]]['x'], 5)) * 1000
      depth = profile.iloc[pair[0]: pair[1]]['z'].max()

      parametrs[key+1] = (width, depth)

    return parametrs


  def auto_profile_40(profile):
    """
    Calculates the maximum 'z' value for segments of a profile.
    
    This method divides the input profile into segments of approximately equal size
    (based on a chunk size of 40) and returns a list containing the maximum 'z'
    value found within each segment. This is useful for identifying peaks or significant
    heights within the profile data, which can be relevant for analyzing surface features
    or material characteristics.
    
    Args:
        profile (list): The input profile data, expected to be a list of dictionaries,
            where each dictionary contains a 'z' key.
    
    Returns:
        list: A list of floats, where each float represents the maximum 'z' value
            found in a segment of the input profile.
    """

    chunk = len(profile) // 40
    prev_i = 0
    result = []
    for i in range(chunk, len(profile), chunk):
      segment = profile[prev_i : i]
      result.append(max(segment['z']))
      prev_i = i

    return result


  def mass_profile():
    """
    Calculates and stores mass profiles for multiple samples to characterize laser-material interactions.
    
    This method processes data from eleven samples, extracting and analyzing mass distribution to understand the effects of laser processing. It reads data, filters it based on spatial coordinates, and generates averaged profiles representing the mass distribution for each sample.
    
    Args:
        None
    
    Returns:
        pd.DataFrame: A DataFrame containing the averaged mass profiles for each sample (sample_1 to sample_11).  The values in each column represent the rounded, averaged mass profile data.
    """

    result_frame = pd.DataFrame()

    for i in range(1, 12):
      sup_list = np.zeros([])
      for j in range(1 ,4):
        data = pd.read_csv(f'/content/sample_data/big/imgs4_x{i}_{j}.txt', names=['x', 'z'], sep=' ')
        data['x'] = data['x'].apply(dots).apply(float)
        data['z'] = data['z'].apply(dots).apply(float)
        data = data[(data['x'] > 5.0) & (data['x'] < 49)]

        result = Profile.auto_profile_40(data) 
        if sup_list.all():
          sup_list += np.array(result)
        else:
          sup_list = np.array(result)


      result_frame[f'sample_{i}'] = np.round(sup_list / 3, 5)


  def save_profile(result_frame):
    """
    Saves the processed data to a CSV file for further analysis and record-keeping.
    
    Args:
        result_frame: The DataFrame containing the profile data to be saved.
    
    Returns:
        None
    """

      result_frame = result_frame[result_frame.columns[::-1]][::-1]
      result_frame.to_csv('profile.csv')


  def plot_check():
    """
    Generates visualizations of data acquired from experiments.
    
    This method processes a sequence of CSV files, extracting and plotting data points 
    corresponding to specific conditions.  The plotted data helps in evaluating 
    experimental results and identifying trends or anomalies.
    
    Args:
        None
    
    Returns:
        None
    """

    for i in range(1, 12):
      sup_list = np.zeros([])
      for j in range(1 ,4):
        data = pd.read_csv(f'/content/sample_data/big/imgs4_x{i}_{j}.txt', names=['x', 'z'], sep=' ')
        data['x'] = data['x'].apply(dots).apply(float)
        data['z'] = data['z'].apply(dots).apply(float)
        data = data[(data['x'] > 5.0) & (data['x'] < 49)]

        fig, ax = plt.subplots(figsize=(8,5))
        plt.plot(data['x'], data['z'])
        ax.set_title(f'sample{i}_{j}')
        ax.set_xlabel("X, мм")
        ax.set_ylabel("z, мкм")
        plt.show()

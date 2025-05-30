import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Profile():

  def auto_profile(profile, full_data):
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
    chunk = len(profile) // 40
    prev_i = 0
    result = []
    for i in range(chunk, len(profile), chunk):
      segment = profile[prev_i : i]
      result.append(max(segment['z']))
      prev_i = i

    return result


  def mass_profile():
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
      result_frame = result_frame[result_frame.columns[::-1]][::-1]
      result_frame.to_csv('profile.csv')


  def plot_check():
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

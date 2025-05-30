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

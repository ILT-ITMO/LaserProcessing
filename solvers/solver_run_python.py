import subprocess

import struct
import numpy as np
import matplotlib.pyplot as plt

import imageio
import os
import h5py

                                                                ## freq     energy   radius   duration
def run_script(exe_path = r"./Release/PINN_cil_temp.exe", args = ["100e3", "3e-3", "25e-6", "40e-9"]):
# Example path to your compiled C++ exe

# Run the exe
    result = subprocess.run([exe_path] + args, capture_output=True)


    print("STDOUT:\n", result.stdout)
    print("STDERR:\n", result.stderr)   
    return 1


def open_binary_Trzt(file_path = "out_Trzt.bin", Nt = 775, Nz = 75, Nr = 75):
    with open(file_path, "rb") as f:
        content = f.read()
    num_doubles = len(content) // 8
    doubles = struct.unpack(f"{num_doubles}d", content)
    Nt = int(len(doubles)/Nz/Nr)
    data = np.array(doubles).reshape((Nt, Nz, Nr))
    return data 

def open_Tt(file_path = "out_Tt.txt"):
    data = np.genfromtxt(file_path)
    return data


def create_gif(frame_path, gif_path, data):
    os.makedirs(frame_path, exist_ok=True)
    # Create and save individual frames
    filenames = []
    max_scale = np.max(data)
    min_scale = np.min(data)

    for t in range(data.shape[0]):
        plt.imshow(np.log(data[t]), cmap='viridis', origin='lower', interpolation='none', aspect='auto', vmax=np.log(max_scale), vmin=np.log(min_scale))
        plt.axis('off')
        # plt.xlabel('R-coordinate')
        # plt.ylabel('Z-coordinate')
        # plt.title(f"Frame: {t}")
        plt.colorbar()
        filename = os.path.join(frame_path, f"frame_{t:03d}.png")
        plt.savefig(filename)
        plt.close()
        filenames.append(filename)  

    # Create GIF

    gif_path += "output.gif"
    with imageio.get_writer(gif_path, mode='I', duration=0.00001) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    print(f"GIF saved to: {gif_path}")

# freq = np.linspace(1e3, 1e6, 5)
# energy = np.linspace(1e-6, 1e-3, 10)
# r0 = np.linspace(5e-6, 30e-6, 3)
# t0 = np.linspace(5e-9, 100e-9, 5)
# for f_item in freq:
#     for e_item in energy: 
#         for r_item in r0:
#             for t_item in t0:
#                 data = run_script(args= [f"{f_item}", f"{e_item}", f"{r_item}", f"{t_item}"])

data = open_binary_Trzt(file_path='D:/results/T_250.750_778.000_30.000_76.250.bin', Nz = 200, Nr = 75)
create_gif(frame_path= "D:/frames", gif_path="D:/out.gif", data= data)

# data = run_script()
# with h5py.File('output.h5', 'w') as f:
#     f.create_dataset('my_dataset', data=data, compression='gzip', compression_opts=9)




# data = open_Tt()
# plt.plot(data[:,0],data[:,1])
# plt.show()




# Example 3D array: shape (time, height, width)
# Replace this with your actual array


# Temporary folder for storing frames
# frame_dir = "frames"


import os
from segmentation import Segmentation
from PIL import Image
import pandas as pd

# сортировка папки по номеру снимка
path = "sample_imgs4"
def sorting_folder(path):
    """
    Sorts files within a folder based on a naming convention, preparing them for sequential processing.
    
    The function sorts files in a given directory, assuming filenames start with a numerical prefix
    followed by an underscore. It first sorts alphabetically by the prefix, then numerically
    by the prefix (after removing the first element). This ensures files are ordered logically
    for tasks like time-series analysis or step-by-step processing.
    
    Args:
        path (str): The path to the folder containing the files to sort.
    
    Returns:
        list: A list of sorted filenames.
    """
    filenames = os.listdir(path)
    filenames.sort(key=lambda file: file.split('_')[0])
    filenames = filenames[1:]
    filenames.sort(key=lambda file: int(file.split('_')[0]))
    return filenames
#print(os.path.abspath("sample_imgs4"))


def segmentation_for_folder(folder):
    """
    Segments images in a folder and returns segmentation results.
    
    This method processes images within a specified folder, performing segmentation
    on each image to identify regions of interest, potentially related to laser-material interactions.
    It refines the segmentation by iteratively adjusting a black level parameter to optimize the results.
    Segmented images are saved to a designated directory for further analysis.
    
    Args:
        folder (str): The path to the folder containing the images to be segmented.
    
    Returns:
        tuple: A tuple containing three lists: `result_list`, `avg_result_list`,
            and `std_div_result_list`.
            `result_list` stores lists of segmentation widths for each image.
            `avg_result_list` stores lists of average widths for each image.
            `std_div_result_list` stores lists of standard deviations for each image.
    """
    result_list, sup_list, avg_result_list, avg_sup_list, std_div_result_list, std_div_sup_list   = [], [], [], [], [], []
    path = folder
    filenames = sorting_folder(path)
    for index, file in enumerate(filenames):
        black_level = 50
        img_path = os.path.join(os.path.abspath(folder), file)
        width, img_to_save, avg_width, std_div = Segmentation.segmentation(Segmentation.crop_center_square(img_path), Segmentation.calculate_percentile_brightness(img_path, black_level))
        prev_width, prev_img_to_save, prev_avg_width, prev_std_div = width, img_to_save, avg_width, std_div
        if width > 30:
            while width > 30:  
                prev_width, prev_img_to_save, prev_avg_width, prev_std_div = width, img_to_save, avg_width, std_div
                black_level -= 1
                width, img_to_save, avg_width, std_div = Segmentation.segmentation(Segmentation.crop_center_square(img_path), Segmentation.calculate_percentile_brightness(img_path, black_level))
        else:
            width, img_to_save, avg_width, std_div = Segmentation.segmentation(Segmentation.crop_center_square(img_path), Segmentation.calculate_percentile_brightness(img_path, black_level, type_='brigth'), type_='brigth')
            while width > 30:
                prev_width, prev_img_to_save, prev_avg_width, prev_std_div = width, img_to_save, avg_width, std_div
                black_level -= 2
                width, img_to_save, avg_width, std_div = Segmentation.segmentation(Segmentation.crop_center_square(img_path), Segmentation.calculate_percentile_brightness(img_path, black_level, type_='brigth'), type_='brigth')


        if prev_img_to_save is not None:
            prev_img_to_save.save(f'/Users/maximmikhalevich/Desktop/project/nirsii/sample_imgs4_segment_4/{file}')
            if index % 4 == 0:
                result_list.append(sup_list)
                avg_result_list.append(avg_sup_list)
                std_div_result_list.append(std_div_sup_list)
                sup_list = []
                avg_sup_list = []
                std_div_sup_list = []
            sup_list.append(prev_width)
            avg_sup_list.append(prev_avg_width)
            std_div_sup_list.append(prev_std_div)

        #логгироывние
        print(f'{index} из {len(filenames)} фото размечено')

    return (result_list, avg_result_list, std_div_result_list)


result, avg_result_list, std_div_result_list = segmentation_for_folder(path)
df = pd.DataFrame(result, columns=["width_0", "width_1", "width_2", "width_3"])
avg_df = pd.DataFrame(avg_result_list, columns=["avg_width_0", "avg_width_1", "avg_width_2", "avg_width_3"])
std_div_df = pd.DataFrame(std_div_result_list, columns=["std_div_0", "std_div_1", "std_div_2", "std_div_3"])
df.to_csv("segmen_result.csv", index=True)
avg_df.to_csv("avg_segmen_result.csv", index=True)
std_div_df.to_csv("std_div_segmen_result.csv", index=True)

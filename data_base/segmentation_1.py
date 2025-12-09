import cv2
import numpy as np
from PIL import Image
import os

class Segmentation():
    """
    This class performs image segmentation tasks, including applying masks based on contrast and converting images to BLOB format.
    
        Attributes:
            None
    
        Class Methods:
        - segmentation: Applies a mask to the input image based on its contrast, preparing it for database storage by converting it to BLOB format.
        - calculate_percentile_brightness: Calculates the percentile brightness of an image, allowing for calculation from either the darker or brighter side.
        - crop_center_square: Crops the center square region from an image.
    """

    def segmentation(img, black_lavel, type_='dark'):

        """
        Performs image segmentation based on contrast, preparing the image for database storage by converting it to a BLOB format.
        
        Args:
            img (str): Path to the image file.
            black_lavel (int): Threshold value for segmentation, ranging from 0 to 255.
            type_ (str, optional): Segmentation type. 'dark' for highlighting dark objects, otherwise highlights bright objects. Defaults to 'dark'.
        
        Returns:
            tuple: A tuple containing the width of the segmented object in micrometers, the segmented image as a PIL object, and the average width and standard deviation of the segmented object.
                - width (float): Width of the segmented object in micrometers.
                - image_pil (PIL.Image.Image): Segmented image as a PIL object.
                - avg_width (float): Average width of the segmented object in micrometers.
                - std_width (float): Standard deviation of the segmented object's width in micrometers.
        """
            
        # Чтение изображения
        try:
            image = cv2.imread(img)
        except:
            image = img

        sup = Image.fromarray(image)

        # Преобразование в оттенки серого
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Бинаризация
        if type_ == 'dark':
            _, thresh = cv2.threshold(
                gray, black_lavel, 255, cv2.THRESH_BINARY_INV)
        else:
            _, thresh = cv2.threshold(
                gray, black_lavel, 255, cv2.THRESH_BINARY)


        def other_segment(thresh):
            thresh_test = thresh.T
            width_result  = []
            width_count = 0
            for i in range(thresh_test.shape[0]):
                width_iter = []
                for j in range(thresh_test.shape[1]):
                    if thresh_test[i][j] != 0:
                        width_count += 1
                    else:
                        if width_count != 0:
                            width_iter.append(width_count)
                            width_count = 0
                width_iter.append(width_count)
                width_result.append(max(width_iter))

            #print(f'СПИСОК {width_result}')    
            return sum(width_result) / len(width_result) * 0.53836990

        other_width = other_segment(thresh)
        #print(f' TEST OTVET {other_segment(thresh) * 0.53836990} micrometetrs')

        sup = Image.fromarray(thresh)
        sup.save('sup_image.jpg')

        # Поиск контуров
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Поиск самого темного объекта
        def darkest_object_func(contours):
            darkest_object = None
            min_avg_intensity = 256
            for cnt in contours:
                mask = np.zeros(gray.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                mean_intensity = cv2.mean(gray, mask=mask)[0]
                if mean_intensity < min_avg_intensity:
                    darkest_object = cnt
                    min_avg_intensity = mean_intensity
            return darkest_object

        # Поиск самого светлого объекта

        def brightest_object_func(contours):
            brightest_object = None
            min_avg_intensity = -1
            for cnt in contours:
                mask = np.zeros(gray.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                mean_intensity = cv2.mean(gray, mask=mask)[0]
                if mean_intensity > min_avg_intensity:
                    brightest_object = cnt
                    min_avg_intensity = mean_intensity
            return brightest_object

        # Нахождение крайних точек самого темного объекта
        if type_ == 'dark':
            # print('USE DARK TYPE')
            segment_object = darkest_object_func(contours)
        else:
            # print('USE BRIGHT TYPE')
            segment_object = brightest_object_func(contours)


        def other_segment_3(arr):
            try:
                arr_reshaped = arr.reshape(-1, 2)

                # 2. Сортируем по X (первая колонка)
                sorted_indices = np.argsort(arr_reshaped[:, 0])
                sorted_arr_reshaped = arr_reshaped[sorted_indices]

                # 3. Возвращаем исходную форму (опционально)
                sorted_arr = sorted_arr_reshaped.reshape(-1, 1, 2)

                '''
                for elem in sorted_arr:
                    print('elem',elem[0][0])
                '''
                prev_elem = sorted_arr[0][0]
                sup_list = []
                result_list = []
                for elem in sorted_arr:
                    if elem[0][0] == prev_elem[0]:
                        sup_list.append(elem[0])        
                        prev_elem = elem[0]
                    else:
                        # prev_elem_ = sup_list[0]
                        max_width = float('-inf')
                        for i, elem_1 in enumerate(sup_list, 1):
                            for elem_2 in sup_list[i:]:
                                if abs(elem_1[1] - elem_2[1]) > max_width:
                                    max_width = abs(elem_1[1] - elem_2[1])
            
                            # if abs(elem_[1] - prev_elem_[1]) > max_width:
                            #     max_width = abs(elem_[1] - prev_elem_[1])
                            # prev_elem_ = elem_    
                        result_list.append(max_width)
                        max_width = float('inf')
                        sup_list = []
                        sup_list.append(elem[0])        
                        prev_elem = elem[0]

                result_list = list(filter(lambda elem: elem > 60, result_list))
                result_array = np.array(result_list)

                try:
                    result_1 = sum(result_list)/len(result_list) * 0.53836990
                    result_2 = np.std(result_array) * 0.53836990
                    return(result_1, result_2)
                except:
                    return (0, 0)
            except:
                return (0, 0)    

        # Результатом при разметке фотографии '4_image_0_1600.jpeg' параметрами 
        #~~~ Segmentation.segmentation(test_img, Segmentation.calculate_percentile_brightness(test_img, 30)) ~~~ 
        # являлся список длины 514, из которого 272 значения != 0, при количестве уникальных X = 515 и общем количестве точек в контуре = 950
        # При повышении нижнего барьера фильтрации в диапазоне > [20:60] длина итогового массива не изменяется, значит в этой облатси значений нет, 
        # учитывая, что 60 px эквивалентно значению +- в 25 мкм, что меньше диаметра лазерного отпечатка на используемой конфигурации лазерного комплекса, 
        # предлагаю оставить layer_width = 60 px, как пороговое значения для отсечения с нижней стороны  

        #other_segment_3(contours)
        # print(len(np.unique(segment_object[:, :, 0])))
        # print(len(segment_object[:, :, 0]))
        avg_width = other_segment_3(segment_object)

              
        if segment_object is not None:
            leftmost = tuple(
                segment_object[segment_object[:, :, 0].argmin()][0])
            rightmost = tuple(
                segment_object[segment_object[:, :, 0].argmax()][0])
            topmost = tuple(
                segment_object[segment_object[:, :, 1].argmin()][0])
            bottommost = tuple(
                segment_object[segment_object[:, :, 1].argmax()][0])

            lenght = np.round(0.53836990 * (rightmost[0] - leftmost[0]), 5)
            width = np.round(0.53836990 * (bottommost[1] - topmost[1]), 5)

        else:
            print(f"Самый {type_} объект не найден.")
            return (0, 0, 0, 0)

        # Рисуем контур самого темного объекта
        if segment_object is not None:
            cv2.drawContours(image, [segment_object], -1, (0, 255, 0), 2)

            # Добавляем текст с размерами
            cv2.putText(image, f'Width: {width} micrometers', (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f'Lenght: {lenght} micrometers', (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f'avg width: {avg_width[0]} micrometers', (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f'standart deviation : {avg_width[1]} micrometers', (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            

        # Преобразуем изображение из BGR в RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Преобразуем в объект PIL
        image_pil = Image.fromarray(image_rgb)

        # Сохраняем и показываем изображение
        image_pil.save('test_pic_2.jpg')

        def image_to_blob(image_path):
            with open(image_path, 'rb') as file:
                blob_data = file.read()
            return blob_data

        return (width, image_pil, avg_width[0], avg_width[1])

    def calculate_percentile_brightness(image_path, procentage: int, type_='dark'):
        """
        Calculates a specified percentile of image brightness.
        
        Args:
            image_path (str or numpy.ndarray): The path to the image file or a numpy array representing the image.
            procentage (int): The percentile value to calculate (0-100).
            type_ (str, optional):  If 'dark', calculates the percentile from the darker side of the image. 
                                    Otherwise, calculates from the brighter side. Defaults to 'dark'.
        
        Returns:
            float: The calculated percentile brightness value.
        
        This method determines the brightness level that corresponds to a given percentile within the image's pixel intensity distribution. 
        By default, it finds the brightness value below which a specified percentage of pixels fall (darker side). 
        When `type_` is not 'dark', it effectively calculates the percentile from the brighter side of the image by inverting the percentage value.
        """
        # Открываем изображение и конвертируем в черно-белое
        try:
            image = Image.open(image_path).convert('L')
            pixels = np.array(image)
        except:
            pixels = np.array(image_path)

        if type_ != 'dark':
            procentage = abs(100-procentage)
        percentile = np.percentile(pixels, procentage)

        return percentile

    def crop_center_square(image_path):
        """
        Crops the center square from an image, ensuring consistent input for subsequent analysis.
        
        Args:
            image_path (str): The path to the image file.
        
        Returns:
            numpy.ndarray: A NumPy array representing the cropped image.
        """
        # Открываем изображение
        img = Image.open(image_path)

        # Получаем размеры изображения
        width, height = img.size
        size = height // 2
        # Вычисляем координаты для обрезки центрального квадрата
        left = width // 2
        top = (height - size) // 2
        right = width
        bottom = (height + size) // 2

        # Обрезаем изображение
        cropped_img = img.crop((left, top, right, bottom))
        # image_pil = Image.fromarray(cropped_img)
        return np.array(cropped_img)


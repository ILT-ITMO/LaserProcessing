# LaserProcessing
## data_base

В директории находится тестовый массив размеченных данных, представленный в ILT_data_base.db, а также вспомогательные скрипты для упрощения взаимодействия с данными. Файл содержит три таблицы: laser_modes, microscope_results и profile_results, связанных по полю mode_name типом связи «один к одному».

<center><img src="https://github.com/ILT-ITMO/LaserProcessing/blob/main/data_base/scheme" width="500"/></center>







Данный массив размеченных данных поможет в обучении моделей для прогнозирования результатов взаимодействия лазерного излучения с различными материалами, а также может стать вашим фундаментом при создании новых режимов лазерной обработки.


В директории можно найти следующие полезные файлы:

- LaserProcessing/data_base/ILT_data_base.db — файл базы данных (его необходимо загрузить на свой жёсткий диск);
- LaserProcessing/data_base/data_extract.py — скрипты на Python для извлечения текстовых и медиаданных. Они требуют подключения к файлу базы данных, поэтому для их использования необходимо поместить файл ILT_data_base.db и data_extract.py в одну директорию;
- LaserProcessing/data_base/segmentation_1.py — алгоритм для автоматического определения ширины лазерного трека на микрофотографии. Реализован с помощью модулей cv2 и Image, для их использования необходимо установить библиотек OpenCV и PIL соответственно, а также включить в проект фреймворк Numpy. Сделать это можно следующими командами:
~~~
pip3 install opencv-python
~~~
~~~
pip install pillow
~~~
~~~
pip install numpy 
~~~

  
Для удобной визуализации файла .db можно воспользоваться расширением на браузер Google Chrome, [установить его можно тут](https://chromewebstore.google.com/detail/sqlite-browser-%D0%B4%D0%BB%D1%8F-%D0%BF%D1%80%D0%BE%D1%81%D0%BC%D0%BE/iclckldkfemlnecocpphinnplnmijkol?hl=ru)   

<u>NB</u>: Поле micro_photo таблицы microscope_results представлено в формате BLOB, поэтому, используя обычный запрос к извлечению данных, получить информативный ответ не получится. Для получения корректного графического изображения воспользуйтесь data_extract.jpeg_extract и пропишите в теле функции необходимый вам SQL-запрос.

---
Далее, командой проекта было принято решение о реорганизации системы хранения данных. Датасет будет располагаться в открытом формате на платформе hugging face, так как данный ресурс предоставляет способы быстрого и простого извлечения и дополнения данных. Также, используя hugging face, возможно легко ссылаться на собранный датасет по фиксированному doi, который присваивается при его создании. Помимо этого данные остаются доступными для визуального восприятия пользователями, чего невозможно достичь при работе с изображениями в формате database. Уже были собраны первые 500 сэмплов, каждый из которых включает в себя режимы лазерного воздействия, микрофотографию полученного рельефа, автоматически размеченную ширину трека и стандартные отклонения для ширины каждого трека. 
Данные уже доступны по [ссылке](https://huggingface.co/datasets/maximmikhalevich/laser_ablation_of_titanium_2) 

---

## solvers/gfmd 
- Contact mechanics simulation by Green's function molecular dynamics (GFMD) method in continuum formulation.

This is the reference implementation for flat two-dimensional elastic half-space statically indented by a rigid friction-less probe of a defined height profile.
This can be used for engineering of contact between a linearly grooved and a smooth surfaces.

The two following system state quantities are output of the numerical simulation: 
1. Normal force distribution along the surface in the rigid punch problem.
2. Height function of the punch and surface displacement in the rigid punch problem.

In order to use the program you will need:
1. Unix-like operating system.
2. Fastest Fourier Transform in the West (FFTW) software library (http://www.fftw.org/). Tested version 3.3.7.
3. Clang/LLVM compiler tools (https://clang.llvm.org/). Tested version 3.8.1.
4. GNU "Make" (https://www.gnu.org/software/make/). Tested version 4.1.

In order to build and run the program, execute the command "make" in the working directory with these files.

For more details on the physical model see the following work:
1. N. Prodanov, W. B. Dapp, M. H. Mueser, "On the Contact Area and Mean Gap of Rough, Elastic Contacts: Dimensional Analysis, Numerical Corrections, and Reference Data", Tribol. Lett. (2014) 53:433-448, DOI: 10.1007/s11249-013-0282-z.

---

Leonid Dorogin, St.-Petersburg, 2025.


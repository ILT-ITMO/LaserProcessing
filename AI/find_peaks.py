#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ищет впадины (локальные минимумы) в 1D-профиле абляции и измеряет их ширину
на уровне половины проминентности (half-prominence). Результаты сохраняются в CSV
и (опционально) строится график.

Как использовать:
1) поместите профиль в .npy (1D-массив) и укажите путь в PROFILE_PATH ниже;
2) при необходимости задайте шаг дискретизации DX (например, мкм/отсчёт);
3) запустите файл обычным способом (двойной клик / Run в IDE).
"""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import os
from scipy.signal import find_peaks, peak_widths, peak_prominences  # type: ignore

from matplotlib.pyplot import yticks

# ===================== НАСТРОЙКИ ДЛЯ ПОЛЬЗОВАТЕЛЯ =====================
PROFILE_PATH = "profile.npy"   # путь к профилю (можно абсолютный). В этой сессии был: "/mnt/data/profile.npy"
OUTPUT_CSV   = "valley_metrics.csv"
SHOW_PLOT    = True            # построить ли график
DX           = 1.0             # шаг дискретизации по X (единицы пользователя, напр., мкм/отсчёт)
X0           = 0.0             # смещение начала координат по X
SMOOTH_WIN   = None            # окно сглаживания (число отсчётов), например 51; None — без сглаживания
PROMINENCE   = None            # порог проминентности; None => 0.5 * std(-y_smooth)
# =====================================================================


def _moving_average(y: np.ndarray, win: Optional[int]) -> np.ndarray:
    if not win or win <= 1:
        return y.astype(float)
    win = int(win)
    if win % 2 == 0:
        win += 1
    k = np.ones(win) / win
    return np.convolve(y, k, mode="same")


def find_valleys(
    y: np.ndarray,
    dx: float = 1.0,
    x0: float = 0.0,
    smooth_win: Optional[int] = None,
    prominence: Optional[float] = None,
) -> pd.DataFrame:
    """
    Возвращает DataFrame со столбцами:
      valley_index, x_center, y_at_center, width, x_left_ip, x_right_ip, prominence
    (width — в тех же единицах, что и dx; координаты x с учётом dx и x0)
    """
    y = np.asarray(y).astype(float)
    x = x0 + np.arange(len(y)) * dx
    y_s = _moving_average(y, smooth_win)

    # Основной путь — SciPy, при отсутствии используем fallback на NumPy
    # try:

    inv = -y_s
    if prominence is None:
        prominence = 0.5 * np.nanstd(inv)
    peaks, _ = find_peaks(inv, prominence=prominence if prominence > 0 else None)

    prom = peak_prominences(inv, peaks)[0]
    w_res = peak_widths(inv, peaks, rel_height=0.5)  # (widths, h, left_ips, right_ips)

    valley_idx = peaks.astype(int)
    left_ips, right_ips = w_res[2], w_res[3]
    widths_samples = w_res[0]

    df = pd.DataFrame({
        "valley_index": valley_idx,
        "x_center": x[valley_idx],
        "y_at_center": y[valley_idx],
        "width": widths_samples * dx,
        "x_left_ip": x0 + left_ips * dx,
        "x_right_ip": x0 + right_ips * dx,
        "prominence": prom,
    }).sort_values("x_center", ignore_index=True)
    return df



def extract_valley_segment(
    y: np.ndarray,
    df: pd.DataFrame,
    valley_number: int,
    dx: float = 1.0,
    x0: float = 0.0,
    margin: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Вырезает участок профиля вокруг одной впадины.

    Parameters
    ----------
    y : np.ndarray
        Исходный профиль (1D).
    df : pd.DataFrame
        Таблица с результатами find_valleys.
    valley_number : int
        Номер впадины в df (0,1,2,...).
    dx : float
        Шаг дискретизации по X.
    x0 : float
        Начальное смещение по X.
    margin : int
        Дополнительно добавить отсчётов слева/справа к границам.

    Returns
    -------
    x_seg, y_seg : np.ndarray
        Координаты и значения профиля на выбранном интервале.
    """
    if valley_number < 0 or valley_number >= len(df):
        raise IndexError("valley_number вне диапазона!")

    row = df.iloc[valley_number]
    left_ip = int(np.floor((row["x_left_ip"] - x0) / dx)) - margin
    right_ip = int(np.ceil((row["x_right_ip"] - x0) / dx)) + margin
    left_ip = max(0, left_ip)
    right_ip = min(len(y) - 1, right_ip)

    x = x0 + np.arange(len(y)) * dx
    return x[left_ip:right_ip + 1], y[left_ip:right_ip + 1]


if __name__ == "__main__":
    # Загрузим профиль и найдём впадины (как в предыдущем коде)
    if __name__ == "__main__":
        y = np.load(PROFILE_PATH)
        VALLEY_NUMBER = 10
        df = find_valleys(y, dx=DX, x0=X0, smooth_win=SMOOTH_WIN, prominence=PROMINENCE)

        # Вывод ширины выбранной впадины (в тех же единицах, что DX)
        if not df.empty and 0 <= VALLEY_NUMBER < len(df):
            width = float(df.loc[VALLEY_NUMBER, "width"])
            print(f"Ширина впадины №{VALLEY_NUMBER}: {width:.6g} (в единицах X)")
        else:
            print("Внимание: таблица пуста или VALLEY_NUMBER вне диапазона.")

        # Вырежем выбранную впадину
        x_seg, y_seg = extract_valley_segment(y, df, valley_number=VALLEY_NUMBER, dx=DX, x0=X0, margin=40)

        x_seg -= np.mean(x_seg)


        import matplotlib.pyplot as plt
        # plt.plot(X0 + np.arange(len(y)) * DX, y, label="full profile")
        plt.plot(x_seg, y_seg, label="selected valley", color="red")
        plt.legend()
        plt.show()


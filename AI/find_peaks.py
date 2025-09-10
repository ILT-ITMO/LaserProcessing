#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ищет впадины (локальные минимумы) в 1D-профиле абляции и измеряет их ширину
на уровне половины проминентности (half-prominence). Результаты сохраняются в CSV
и (опционально) строится график.

Теперь добавлен фитинг формы отдельной бороздки (x_seg, y_seg)
моделью "пороговая парабола":
    y(x) = baseline - max(0, h0 - k * x^2)

Где:
 - baseline — уровень ненарушенной поверхности,
 - h0 — центральная глубина бороздки,
 - k  — кривизна (связана с эффективной шириной R = sqrt(h0/k)).

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
from scipy.optimize import curve_fit  # NEW: для фита

from matplotlib.pyplot import yticks
import matplotlib.pyplot as plt  # перемещено сюда, чтобы использовать в фит-плоте

# ===================== НАСТРОЙКИ ДЛЯ ПОЛЬЗОВАТЕЛЯ =====================
PROFILE_PATH = "profile.npy"   # путь к профилю (можно абсолютный). В этой сессии был: "/mnt/data/profile.npy"
OUTPUT_CSV   = "valley_metrics.csv"
SHOW_PLOT    = True            # построить ли график
DX           = 1.0             # шаг дискретизации по X (единицы пользователя, напр., мкм/отсчёт)
X0           = 0.0             # смещение начала координат по X
SMOOTH_WIN   = None            # окно сглаживания (число отсчётов), например 51; None — без сглаживания
PROMINENCE   = None            # порог проминентности; None => 0.5 * std(-y_smooth)

# Флаг: выполнять ли показательный фит для выбранного сегмента
DO_FIT       = True
VALLEY_NUMBER = 9
MARGIN_SEG    = 40
# =====================================================================

def load_grid_txt(
    path: str,
    delimiter=None,
    skiprows: int = 0,
    y_tol: float = 1e-12,
    ):
    """
    Загружает файл с колонками: x, y, z.
    Ожидается порядок: для каждого y идут все x.
    Возвращает: X2D, Y2D, Z2D, x_unique, y_unique
    """
    # читаем 3 колонки, игнорируем строки-комментарии (#)
    data = np.loadtxt(path, delimiter=delimiter, comments="#", skiprows=skiprows)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError("Ожидаю минимум 3 колонки: x, y, value")

    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    # 1) определим размер ряда по x: длина первого блока с одинаковым y
    y0 = y[0]
    Nx = 1
    for yy in y[1:]:
        if abs(yy - y0) < y_tol:
            Nx += 1
        else:
            break

    if Nx <= 0 or len(x) % Nx != 0:
        raise ValueError("Не удаётся определить размер блока по x (Nx). Проверьте порядок данных.")

    Ny = len(x) // Nx

    # 2) уникальные координаты по порядку следования
    x_unique = x[:Nx].copy()
    y_unique = y[::Nx].copy()

    # (необязательно) лёгкая валидация сетки
    # проверим, что во всех блоках x повторяется одинаково
    for row in range(1, Ny):
        xb = x[row * Nx : (row + 1) * Nx]
        if not np.allclose(xb, x_unique, rtol=0, atol=1e-12):
            raise ValueError(f"x в блоке {row} не совпадает с первым блоком — данные не на прямоугольной сетке?")

    # проверим, что y внутри блока практически постоянен
    for row in range(Ny):
        yb = y[row * Nx : (row + 1) * Nx]
        if not np.allclose(yb, yb[0], rtol=0, atol=y_tol):
            raise ValueError(f"y в блоке {row} меняется — ожидался фиксированный y.")

    # 3) формируем 2D-поля
    Z2D = z.reshape(Ny, Nx)              # строки — разные y, столбцы — x
    X2D, Y2D = np.meshgrid(x_unique, y_unique)

    return X2D, Y2D, Z2D, x_unique, y_unique


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


# ===================== НОВОЕ: ФУНКЦИИ ДЛЯ ФИТА =====================

def _model_threshold_parabola(x: np.ndarray, baseline: float, h0: float, k: float) -> np.ndarray:
    """
    y(x) = baseline - max(0, h0 - k*x^2)
    - baseline: уровень поверхности (вне бороздки)
    - h0: центральная глубина бороздки (>=0)
    - k: кривизна (>=0); эффективная половинная ширина R = sqrt(h0/k)
    """
    return baseline - np.maximum(0.0, h0 - k * x**2)


def _initial_guess_threshold_parabola(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """
    Грубые начальные оценки (устойчивые для curve_fit):
      baseline ~ медиана верхних 20% значений,
      h0 ~ baseline - y_min (не меньше 0),
      k  ~ h0 / R^2, где R ~ 1/3 ширины сегмента.
    """
    x = np.asarray(x).astype(float)
    y = np.asarray(y).astype(float)

    # baseline: медиана верхних значений
    q80 = np.quantile(y, 0.8)
    baseline = np.median(y[y >= q80]) if np.any(y >= q80) else np.median(y)

    y_min = float(np.min(y))
    h0 = max(1e-6, baseline - y_min)

    # ширина по x
    R0 = max(1e-6, 0.33 * (np.max(x) - np.min(x)))
    k = max(1e-9, h0 / (R0**2))
    return baseline, h0, k


def fit_valley_threshold_parabola(
    x_seg: np.ndarray,
    y_seg: np.ndarray,
    *,
    bounds: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Фитит сегмент (x_seg, y_seg) моделью:
        y(x) = baseline - max(0, h0 - k*x^2)

    Parameters
    ----------
    x_seg, y_seg : np.ndarray
        Данные сегмента (желательно центрировать x_seg около нуля для устойчивости).
    bounds : ((low_baseline, low_h0, low_k), (high_baseline, high_h0, high_k)) или None
        Ограничения для curve_fit.

    Returns
    -------
    popt : np.ndarray
        Оценки параметров [baseline, h0, k].
    pcov : np.ndarray
        Оценка ковариационной матрицы (от curve_fit).
    """
    x = np.asarray(x_seg, dtype=float)
    y = np.asarray(y_seg, dtype=float)

    # Начальные оценки
    b0, h0, k0 = _initial_guess_threshold_parabola(x, y)

    # Границы по умолчанию: baseline в пределах ±(3*h0) от b0; h0>=0; k>=0
    if bounds is None:
        low = (b0 - 3.0 * max(1e-6, h0), 0.0, 0.0)
        high = (b0 + 3.0 * max(1e-6, h0), 10.0 * max(1e-6, h0), 1e6)
        bounds = (low, high)

    popt, pcov = curve_fit(
        _model_threshold_parabola,
        x, y,
        p0=(b0, h0, k0),
        bounds=bounds,
        maxfev=10000
    )
    return popt, pcov

# ========= НОВОЕ: плавная кромка (soft-edge) =========

def _softplus(z: np.ndarray, beta: float) -> np.ndarray:
    # гладкая аппроксимация max(0, z); чем больше beta, тем ближе к "max"
    return (1.0 / beta) * np.log1p(np.exp(beta * z))

def _sigmoid(u: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-u))

def _model_smooth_edge_parabola(x: np.ndarray, baseline: float, h0: float, k: float, sigma_edge: float) -> np.ndarray:
    """
    Модель: y(x) = baseline - softplus(h0 - k*x^2, beta) * S((R - |x|)/sigma_edge)

    где
      R = sqrt(h0/k) — эффективный радиус "бороздки",
      softplus(...) сглаживает порог в центре,
      S(...) = sigmoid(...) плавно "выключает" бороздку у кромки (мягкие стенки).

    Параметры:
      baseline   — уровень ненарушенной поверхности,
      h0 >= 0    — центральная глубина,
      k  >= 0    — кривизна,
      sigma_edge > 0 — ширина размытия кромки (чем больше, тем мягче стенки).
    """
    # защищаемся от нулевых/отрицательных k, h0
    h0 = np.maximum(h0, 0.0)
    k  = np.maximum(k,  1e-18)
    sigma_edge = np.maximum(sigma_edge, 1e-9)

    R = np.sqrt(h0 / k)

    # Берём одну ручку "мягкости" на всё: beta = 1/sigma_edge
    beta = 1.0 / sigma_edge

    core = _softplus(h0 - k * x**2, beta=beta)         # сглаженный "max(0, h0 - kx^2)"
    edge = _sigmoid((R - np.abs(x)) / sigma_edge)      # плавное выключение у кромки
    return baseline - core * edge

def _initial_guess_smooth_edge(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    # Оценки близкие к тем, что для жёсткой модели + мягкость ~ 10% ширины сегмента
    x = np.asarray(x, float); y = np.asarray(y, float)
    q80 = np.quantile(y, 0.8)
    baseline = np.median(y[y >= q80]) if np.any(y >= q80) else np.median(y)
    y_min = float(np.min(y))
    h0 = max(1e-6, baseline - y_min)
    R0 = max(1e-6, 0.33 * (np.max(x) - np.min(x)))
    k0 = max(1e-9, h0 / (R0**2))
    sigma0 = 0.1 * R0  # мягкость по умолчанию ~10% от радиуса
    return baseline, h0, k0, sigma0

def fit_valley_smooth_parabola(
    x_seg: np.ndarray,
    y_seg: np.ndarray,
    *,
    bounds: tuple[tuple[float, float, float, float], tuple[float, float, float, float]] | None = None,
):
    """
    Фит плавной модели бороздки:
        y(x) = baseline - softplus(h0 - k*x^2, 1/sigma_edge) * sigmoid((R - |x|)/sigma_edge)
    Возвращает (popt, pcov), где popt = [baseline, h0, k, sigma_edge].
    """
    x = np.asarray(x_seg, float)
    y = np.asarray(y_seg, float)
    b0, h0, k0, s0 = _initial_guess_smooth_edge(x, y)

    if bounds is None:
        # baseline +/- 3*h0; h0>=0..10*h0; k>=0..1e6; sigma_edge в разумных пределах
        low  = (b0 - 3.0 * max(h0, 1e-6), 0.0, 0.0, 1e-9)
        high = (b0 + 3.0 * max(h0, 1e-6), 10.0 * max(h0, 1e-6), 1e6, max(1.0, 10.0 * s0))
        bounds = (low, high)

    popt, pcov = curve_fit(
        _model_smooth_edge_parabola,
        x, y,
        p0=(b0, h0, k0, s0),
        bounds=bounds,
        maxfev=20000
    )
    return popt, pcov


# ===================== ОСНОВНОЙ БЛОК =====================

if __name__ == "__main__":
    # Загрузим профиль и найдём впадины
    y = np.load(PROFILE_PATH)
    df = find_valleys(y, dx=DX, x0=X0, smooth_win=SMOOTH_WIN, prominence=PROMINENCE)

    ##### Отрисовка всех найденных пиков
    # x = X0 + np.arange(len(y)) * DX
    #
    # plt.figure()
    # plt.plot(x, y, lw=1.5, label="профиль y(x)")
    #
    # # точки центров впадин
    # plt.scatter(df["x_center"], df["y_at_center"], s=25, zorder=3, label="впадины")
    #
    # # подписи номеров (0,1,2,...) у каждой найденной впадины
    # for i, (xc, yc) in enumerate(zip(df["x_center"], df["y_at_center"])):
    #     plt.annotate(str(i), (xc, yc), textcoords="offset points",
    #                  xytext=(0, 8), ha="center", fontsize=9)
    #
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.legend()
    # plt.show()
    # Вывод ширины выбранной впадины (в тех же единицах, что DX)
    if not df.empty and 0 <= VALLEY_NUMBER < len(df):
        width = float(df.loc[VALLEY_NUMBER, "width"])
        print(f"Ширина впадины №{VALLEY_NUMBER}: {width:.6g} (в единицах X)")
    else:
        print("Внимание: таблица пуста или VALLEY_NUMBER вне диапазона.")

    # Вырежем выбранную впадину
    x_seg, y_seg = extract_valley_segment(y, df, valley_number=VALLEY_NUMBER, dx=DX, x0=X0, margin=MARGIN_SEG)
    # Центрируем x для устойчивого фита (не обязательно, но полезно)
    # индекс минимального значения в вырезанном сегменте
    x_seg -= np.mean(x_seg)
    # idx_min = np.argmin(y_seg)
    # # координата x в этом индексе
    # x_peak = x_seg[idx_min]
    # # сдвигаем так, чтобы минимум оказался в 0
    # x_seg = x_seg - x_peak

    # Показ сегмента и (опционально) фита
    plt.figure()
    plt.plot(x_seg, y_seg, label="selected valley", lw=2)
    DO_FIT_SMOOTH = True
    if DO_FIT:
        if DO_FIT_SMOOTH:
            popt, pcov = fit_valley_smooth_parabola(x_seg, y_seg)
            baseline, h0, k, sigma_edge = popt
            R = np.sqrt(h0 / max(k, 1e-18))
            print(
                f"[FIT smooth] baseline={baseline:.6g}, h0={h0:.6g}, k={k:.6g}, sigma_edge={sigma_edge:.6g}, R={R:.6g}")
            x_fit = np.linspace(x_seg.min(), x_seg.max(), 1000)
            y_fit = _model_smooth_edge_parabola(x_fit, *popt)
            plt.plot(x_fit, y_fit, label="fit: smooth-edge parabola", ls="--")
        else:
            popt, pcov = fit_valley_threshold_parabola(x_seg, y_seg)
            baseline, h0, k = popt
            R = np.sqrt(h0 / max(k, 1e-18))
            print(f"[FIT] baseline={baseline:.6g}, h0={h0:.6g}, k={k:.6g}, R={R:.6g}")
            x_fit = np.linspace(x_seg.min(), x_seg.max(), 1000)
            y_fit = _model_threshold_parabola(x_fit, *popt)
            plt.plot(x_fit, y_fit, label="fit: thresholded parabola", ls="--")

    plt.xlabel("x (в единицах X)")
    plt.ylabel("y (профиль)")
    plt.legend()
    if SHOW_PLOT:
        plt.show()

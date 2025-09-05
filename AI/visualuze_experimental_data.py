import numpy as np

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

import matplotlib.pyplot as plt
# --- пример использования ---
if __name__ == "__main__":
    X2D, Y2D, Z2D, xs, ys = load_grid_txt("kanali obsh.txt")
    print(f"Nx={len(xs)}, Ny={len(ys)}")
    print("x:", xs[:5], "...")
    print("y:", ys[:5], "...")
    print("Z shape:", Z2D.shape)
    np.save("profile.npy", (Z2D[14,:]))
    plt.plot(Z2D[14,:])
    plt.show()

    # # Пример визуализации (опционально)
    # try:
    #     import matplotlib.pyplot as plt
    #
    #     plt.figure()
    #     # изображение без интерполяции, оси — реальные значения
    #     extent = [xs.min(), xs.max(), ys.min(), ys.max()]
    #     plt.imshow(Z2D, origin="lower", extent=extent, aspect="auto")
    #     plt.xlabel("x")
    #     plt.ylabel("y")
    #     plt.title("value(x, y)")
    #     plt.colorbar(label="value")
    #     plt.show()
    # except Exception as e:
    #     print("Визуализация не выполнена:", e)

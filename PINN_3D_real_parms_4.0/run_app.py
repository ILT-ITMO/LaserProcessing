# run_app.py
import panel as pn
from config_gui_panel import LaserConfigApp

app = LaserConfigApp()

if __name__ == "__main__":
    pn.serve(
        app.show(),
        title="Конфигуратор PINN",
        port=5006,
        show=True, 
        threaded=True,
        verbose=True
    )
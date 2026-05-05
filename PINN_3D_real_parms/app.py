import os
import threading
import time
import json
import shutil
from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for background threads
import matplotlib.pyplot as plt

# Import project modules
import config
from pinn import PINN, train_pinn
from conditions import convert_to_physical_temperature
from visual import create_animation
from visual import laser_crater_3d

app = FastAPI(title="PINN Laser Model Interface")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Ensure directories exist
os.makedirs("results/runs", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Web defaults: load continuous preset on startup
def _load_web_default_config():
    preset_path = os.path.join(os.path.dirname(__file__), "continuous_config.json")
    if not os.path.exists(preset_path):
        return
    try:
        with open(preset_path, "r", encoding="utf-8") as f:
            preset = json.load(f)
        if isinstance(preset, dict):
            config.CONFIG.config = config.CONFIG.deep_update(config.CONFIG.config, preset)
            config.CONFIG.calculate_derived_parameters()
    except Exception as e:
        print(f"[WARN] Failed to load web default config from {preset_path}: {e}")

_load_web_default_config()

# Mount static for general results
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/results", StaticFiles(directory="results"), name="results")

# Global state
class TrainingState:
    is_running = False
    stop_requested = False
    current_epoch = 0
    total_epochs = 0
    current_loss = 0.0
    history: List[float] = []
    message = "Idle"
    run_id = ""
    mode = ""
    metrics: Dict[str, Any] = {}
    learned: Dict[str, Any] = {}
    param_name: str = ""
    param_current: Optional[float] = None
    param_history: List[Dict[str, Any]] = []
    
state = TrainingState()

# Pydantic models
class LaserConfigModel(BaseModel):
    laser: Dict[str, Any]
    material: Dict[str, Any]
    pinn: Dict[str, Any]
    training: Dict[str, Any]
    crater: Optional[Dict[str, Any]] = None
    inverse: Optional[Dict[str, Any]] = None

def background_train(cfg_dict, run_id):
    """Background training task"""
    global state
    state.is_running = True
    state.stop_requested = False
    state.history = []
    state.message = "Initializing..."
    state.run_id = run_id
    state.mode = cfg_dict["laser"]["mode"]
    state.metrics = {}
    state.learned = {}
    state.param_name = ""
    state.param_current = None
    state.param_history = []
    
    run_dir = f"results/runs/{run_id}"
    os.makedirs(run_dir, exist_ok=True)
    
    # Save config used for this run
    with open(f"{run_dir}/config.json", 'w') as f:
        json.dump(cfg_dict, f, indent=2)
    
    try:
        # Update config object for training logic
        config.CONFIG.config = config.CONFIG.deep_update(config.CONFIG.config, cfg_dict)
        config.CONFIG.calculate_derived_parameters()
        
        # Setup device
        device_name = config.CONFIG.config["training"]["device"]
        if device_name == "auto":
            device = torch.device("mps" if torch.backends.mps.is_available() else 
                                 "cuda" if torch.backends.cuda.is_built() else "cpu")
        else:
            device = torch.device(device_name)
            
        state.message = f"Training on {device}"
        
        # Init model
        model = PINN([4, 128, 128, 128, 1]).to(device)
        diff_coef = float(getattr(config, "INITIAL_DIFF_COEF", 1.0))
        
        num_epochs = config.CONFIG.config["training"]["num_epochs"]
        state.total_epochs = num_epochs
        
        # Callback
        def progress_cb(epoch, loss, extras=None):
            if state.stop_requested:
                raise InterruptedError("Training stopped by user")
            state.current_epoch = epoch
            state.current_loss = loss
            state.history.append(loss)
            if extras and isinstance(extras, dict):
                param_name = extras.get("param_name")
                param_value = extras.get("param_value")
                if param_name in {"m2", "coef"} and isinstance(param_value, (int, float)):
                    state.param_name = str(param_name)
                    state.param_current = float(param_value)
                    state.param_history.append({"epoch": int(epoch), "value": float(param_value)})
            
        # Train
        try:
            inverse_cfg = config.CONFIG.config.get("inverse", {}) or {}
            training_cfg = config.CONFIG.config.get("training", {}) or {}

            m2_loss_weight = float(inverse_cfg.get("m2_loss_weight", getattr(config, "M2_LOSS_WEIGHT", 0.0)))
            target_width_um = float(inverse_cfg.get("target_width_um", getattr(config, "TARGET_WIDTH_UM", 145.0)))

            train_out = train_pinn(
                model, 
                diff_coef, 
                num_epochs=num_epochs,
                lr=config.CONFIG.config["training"]["learning_rate"],
                device=device,
                laser_mode=state.mode,
                progress_callback=progress_cb,
                inverse_enabled=bool(inverse_cfg.get("enabled", getattr(config, "INVERSE_ENABLED", False))),
                inverse_mode=str(training_cfg.get("inverse_mode", getattr(config, "INVERSE_MODE", "classic"))),
                pretrained_checkpoint_path=training_cfg.get("pretrained_checkpoint_path", None),
                pretrained_strict=bool(training_cfg.get("pretrained_strict", True)),
                log_every=training_cfg.get("log_every", getattr(config, "LOG_EVERY", 100)),
                m2_loss_weight=m2_loss_weight,
                target_width_um=target_width_um,
                progress_param_every=50,
                return_info=True,
            )
            loss_hist, info = train_out
            state.learned = info or {}
            state.message = "Finished"
        except InterruptedError:
            state.message = "Stopped by user"
            
        # Post-processing (always attempt to generate results for what we have)
        if state.current_epoch > 0:
            try:
                state.message = "Generating artifacts..."
                metrics = generate_results_to_folder(model, device, run_dir, state.mode)
                state.metrics = metrics or {}
                if "Stopped" not in state.message:
                    state.message = "Finished"
            except Exception as viz_e:
                print(f"Visualization error: {viz_e}")
                state.message = f"Error in visualization: {str(viz_e)}"
            
            # Save final state/metrics
            with open(f"{run_dir}/summary.json", 'w') as f:
                json.dump({
                    "epochs": state.current_epoch,
                    "final_loss": state.current_loss,
                    "mode": state.mode,
                    "timestamp": datetime.now().isoformat(),
                    "status": state.message,
                    "metrics": state.metrics,
                    "learned": state.learned,
                    "param_name": state.param_name,
                    "param_history": state.param_history,
                }, f)
            
    except Exception as e:
        state.message = f"Error: {str(e)}"
        print(f"Training error: {e}")
    finally:
        state.is_running = False

def generate_results_to_folder(model, device, run_dir, mode):
    """Generate predictions and animation after training into specific folder"""
    viz_points = config.CONFIG.config["pinn"]["visualization_points"]
    nx_plot, ny_plot, nz_plot, nt_plot = (
        viz_points["x"], viz_points["y"], viz_points["z"], viz_points["t"]
    )
    
    x_plot = np.linspace(-1, 1, nx_plot)  
    y_plot = np.linspace(-1, 1, ny_plot)  
    z_plot = np.linspace(0, 1, nz_plot)   
    t_plot = np.linspace(0, config.SIMULATION_TIME_NORM, nt_plot)  
    
    with torch.no_grad():
        Xp, Yp, Zp, Tp = np.meshgrid(x_plot, y_plot, z_plot, t_plot, indexing='ij')
        x_t = torch.tensor(Xp.flatten(), dtype=torch.float32, device=device)
        y_t = torch.tensor(Yp.flatten(), dtype=torch.float32, device=device)
        z_t = torch.tensor(Zp.flatten(), dtype=torch.float32, device=device)
        t_t = torch.tensor(Tp.flatten(), dtype=torch.float32, device=device)
        
        U_pred_norm = model(x_t, y_t, z_t, t_t).cpu().numpy().reshape(
            nx_plot, ny_plot, nz_plot, nt_plot
        )
        
    gif_path = f'{run_dir}/solution.gif'
    title = f'PINN Solution: {mode} mode'
    create_animation(U_pred_norm, x_plot, y_plot, z_plot, t_plot, title, gif_path)

    # Metrics: MAE/MSE between simulated isotherm line and crater-based line (same as main.py)
    metrics: Dict[str, Any] = {}
    try:
        U_pred_physical = convert_to_physical_temperature(U_pred_norm)

        x_phys = np.array(x_plot) * config.CHARACTERISTIC_LENGTH * 1e6
        y_phys = np.array(y_plot) * config.CHARACTERISTIC_LENGTH * 1e6
        z_phys = np.array(z_plot) * config.CHARACTERISTIC_LENGTH * 1e6

        time_idx = -1
        isotherm_temp = float(getattr(config, "CRATER_TARGET_TEMP", 1900.0))
        center_y = len(y_phys) // 2
        surface_z_idx = len(z_phys) - 1

        simulated_line = U_pred_physical[:, center_y, surface_z_idx, time_idx].astype(np.float64)

        crater_field = laser_crater_3d(
            x_phys, y_phys, z_phys,
            max_depth_um=float(getattr(config, "CRATER_PEAK_DEPTH_UM", 30.0)),
            crater_width_um=float(getattr(config, "CRATER_WIDTH_99_UM", 145.0)),
            max_height_um=float(getattr(config, "CRATER_PEAK_DEPTH_UM", 30.0)),
            decay_length_um=float(getattr(config, "CRATER_PEAK_DEPTH_UM", 30.0)),
        )
        crater_surface = crater_field[:, :, int(np.argmin(np.abs(z_phys - 0.0)))]
        crater_peak = float(getattr(config, "CRATER_PEAK_DEPTH_UM", 30.0))
        crater_line = crater_surface[:, center_y].astype(np.float64)
        experimental_line = (isotherm_temp - crater_peak + crater_line).astype(np.float64)

        n = int(min(simulated_line.shape[0], experimental_line.shape[0]))
        simulated_line = simulated_line[:n]
        experimental_line = experimental_line[:n]

        mse = float(np.mean((simulated_line - experimental_line) ** 2))
        mae = float(np.mean(np.abs(simulated_line - experimental_line)))

        metrics = {
            "isotherm_temp_K": float(isotherm_temp),
            "mse": mse,
            "mae": mae,
        }

        with open(f"{run_dir}/metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] Metrics computation failed: {e}")
    
    # Save loss plot
    plt.figure(figsize=(8,5))
    plt.plot(state.history)
    plt.yscale('log')
    plt.title('Training Loss')
    plt.grid(True)
    plt.savefig(f'{run_dir}/loss.png')
    plt.close()
    
    # Save history data for interactive chart later if needed
    np.save(f'{run_dir}/loss_history.npy', np.array(state.history))

    return metrics

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/config")
def get_config():
    return config.CONFIG.config

@app.post("/config")
def update_config(new_config: LaserConfigModel):
    incoming = new_config.dict()
    # Normalize a few UI-prone fields (strings like "", "null", "1e-3")
    try:
        laser = incoming.get("laser", {}) or {}
        if "simulation_time" in laser and isinstance(laser["simulation_time"], str):
            s = laser["simulation_time"].strip()
            if s == "" or s.lower() == "null":
                laser["simulation_time"] = None
            else:
                laser["simulation_time"] = float(s)
        incoming["laser"] = laser
    except Exception:
        pass

    config.CONFIG.config = config.CONFIG.deep_update(config.CONFIG.config, incoming)
    config.CONFIG.calculate_derived_parameters()
    return {"status": "ok", "config": config.CONFIG.config}

@app.post("/train/start")
def start_training(background_tasks: BackgroundTasks):
    if state.is_running:
        return JSONResponse(status_code=400, content={"message": "Training already running"})
    
    # Unique ID for this run
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Copy current config for training
    cfg_copy = config.CONFIG.config.copy()
    
    t = threading.Thread(target=background_train, args=(cfg_copy, run_id))
    t.start()
    
    return {"status": "started", "run_id": run_id}

@app.post("/train/stop")
def stop_training():
    if not state.is_running:
        return {"status": "not running"}
    state.stop_requested = True
    return {"status": "stopping"}

@app.get("/train/status")
def get_status():
    return {
        "is_running": state.is_running,
        "epoch": state.current_epoch,
        "total_epochs": state.total_epochs,
        "loss": state.current_loss,
        "history": state.history,
        "message": state.message,
        "mode": state.mode,
        "run_id": state.run_id,
        "metrics": state.metrics,
        "learned": state.learned,
        "param_name": state.param_name,
        "param_current": state.param_current,
        "param_history": state.param_history,
    }

@app.get("/history")
def get_history():
    """List all past runs with metadata"""
    history_base = "results/runs"
    runs = []
    if not os.path.exists(history_base):
        return []
        
    for run_id in sorted(os.listdir(history_base), reverse=True):
        run_path = os.path.join(history_base, run_id)
        if not os.path.isdir(run_path):
            continue
            
        summary_path = os.path.join(run_path, "summary.json")
        summary = {}
        if os.path.exists(summary_path):
            try:
                with open(summary_path, 'r') as f:
                    summary = json.load(f)
            except:
                pass
        
        has_gif = os.path.exists(os.path.join(run_path, "solution.gif"))
        has_loss = os.path.exists(os.path.join(run_path, "loss.png"))
        
        runs.append({
            "id": run_id,
            "summary": summary,
            "has_gif": has_gif,
            "has_loss": has_loss
        })
    return runs

@app.get("/history/{run_id}/data")
def get_run_data(run_id: str):
    """Get full data for a specific run including history array and config"""
    run_dir = f"results/runs/{run_id}"
    history_file = f"{run_dir}/loss_history.npy"
    
    data = {"id": run_id, "history": []}
    if os.path.exists(history_file):
        data["history"] = np.load(history_file).tolist()
        
    summary_path = f"{run_dir}/summary.json"
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            data["summary"] = json.load(f)

    metrics_path = f"{run_dir}/metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            data["metrics"] = json.load(f)
            
    config_path = f"{run_dir}/config.json"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            data["config"] = json.load(f)
            
    return data

@app.get("/history/{run_id}/gif")
def get_run_gif(run_id: str):
    return FileResponse(f"results/runs/{run_id}/solution.gif")

@app.get("/history/{run_id}/loss_plot")
def get_run_loss_plot(run_id: str):
    return FileResponse(f"results/runs/{run_id}/loss.png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

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

app = FastAPI(title="PINN Laser Model Interface")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Ensure directories exist
os.makedirs("results/runs", exist_ok=True)
os.makedirs("static", exist_ok=True)

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
    
state = TrainingState()

# Pydantic models
class LaserConfigModel(BaseModel):
    laser: Dict[str, Any]
    material: Dict[str, Any]
    pinn: Dict[str, Any]
    training: Dict[str, Any]

def background_train(cfg_dict, run_id):
    """Background training task"""
    global state
    state.is_running = True
    state.stop_requested = False
    state.history = []
    state.message = "Initializing..."
    state.run_id = run_id
    state.mode = cfg_dict["laser"]["mode"]
    
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
        diff_coef = 1.0
        
        num_epochs = config.CONFIG.config["training"]["num_epochs"]
        state.total_epochs = num_epochs
        
        # Callback
        def progress_cb(epoch, loss):
            if state.stop_requested:
                raise InterruptedError("Training stopped by user")
            state.current_epoch = epoch
            state.current_loss = loss
            state.history.append(loss)
            
        # Train
        try:
            loss_hist = train_pinn(
                model, 
                diff_coef, 
                num_epochs=num_epochs,
                lr=config.CONFIG.config["training"]["learning_rate"],
                device=device,
                laser_mode=state.mode,
                progress_callback=progress_cb
            )
            state.message = "Finished"
        except InterruptedError:
            state.message = "Stopped by user"
            
        # Post-processing (always attempt to generate results for what we have)
        if state.current_epoch > 0:
            try:
                state.message = "Generating artifacts..."
                generate_results_to_folder(model, device, run_dir, state.mode)
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
                    "status": state.message
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

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/config")
def get_config():
    return config.CONFIG.config

@app.post("/config")
def update_config(new_config: LaserConfigModel):
    config.CONFIG.config = config.CONFIG.deep_update(config.CONFIG.config, new_config.dict())
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
        "run_id": state.run_id
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

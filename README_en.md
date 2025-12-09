# Laser Processing Prediction System

![GitHub](https://img.shields.io/badge/GitHub-ILT--ITMO-blue)
![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=FFD43B)
![C++](https://img.shields.io/badge/C++-00599C?logo=cplusplus&logoColor=white)
![Framework](https://img.shields.io/badge/Framework-PyTorch-red)

## 🌟 About the Project

This repository contains a comprehensive system for modeling, analyzing, and predicting optimal parameters for laser material structuring.

The project combines physical process modeling, algorithmic database generation, and deep learning methods (Physics-Informed Neural Networks) to solve the problem of finding ideal laser impact parameters.

---

## 🏗 Structure and Relationships

The system consists of three interconnected modules, each solving its own class of problems. The data flow in the project is organized as follows:

### System Architecture

```
Solvers (Physical Modeling)
    ↓
    ├─ Contact interaction modeling
    ├─ Temperature field calculation
    └─ Synthetic data generation
    ↓
Data Base (Dataset Formation)
    ↓
    ├─ Data aggregation from solvers
    ├─ Processing of experimental results
    └─ Structuring of the training sample
    ↓
PINN_3D_real_parms (Neural Network)
    ↓
    └─ Prediction of optimal laser impact parameters
```

### Module Interconnection in the Final Pipeline

```
Temperature Solver
    ↓
    Synthetic: T(x,y,t)
    ↓
PINN Training

Surface Optimizer
    ↓
    Calculation request
    ↓
GFMD Solver
    ↓
    Stress and strain
    ↓
Surface Optimizer
    ↓
    Optimal profile
    ↓
Recommendations for laser processing
```

---

## 📚 System Components

### 1. Physical Modeling (Solvers)
🌿 **Branch:** [`solvers`](https://github.com/ILT-ITMO/LaserProcessing/tree/main/solvers)

The foundation of the project. Here are solvers for numerical modeling of physical processes occurring during laser processing.

**Tasks:**
* Modeling contact interaction of materials
* Calculation of temperature fields and thermal patterns
* Solving heat conduction and material dynamics equations

**Role in the system:** Generation of synthetic physically correct data for subsequent model training.

---

### 2. Database Generation (Database)
🌿 **Branch:** [`data_base`](https://github.com/ILT-ITMO/LaserProcessing/tree/main/data_base)

A toolkit for aggregating and preparing data. The module is responsible for collecting, processing, and structuring information obtained from solvers and real experiments.

**Tasks:**
* Algorithms for creating and labeling a database of laser impact modes
* Data validation and cleaning
* Preparing samples for training and validation

**Role in the system:** Preparing a valid dataset for fine-tuning the neural network model.

---

### 3. Predictive Model (PINN)
🌿 **Branch:** [`PINN_3D_real_parms`](https://github.com/ILT-ITMO/LaserProcessing/tree/main/PINN_3D_real_parms)

The core of the system. Here is the implementation of a physics-informed neural network (PINN) architecture for solving the inverse problem – finding laser parameters.

**Tasks:**
* Training the neural network on prepared datasets
* Predicting optimal structuring modes based on 3D models
* Using real material and installation parameters

**Role in the system:** The final tool for the technologist and researcher to obtain optimal laser processing parameters.

---

## 🚀 Getting Started

Since each module is an independent part of the system, detailed instructions for installation and launch are located within the corresponding directories:

1. **To work with physical models**, go to [`solvers`](./solvers) – here are instructions for running solvers and generating data.

2. **To integrate datasets**, switch to the [`data_base`](https://github.com/ILT-ITMO/LaserProcessing/tree/main/data_base) branch – there are algorithms for preparing data and working with it.

3. **To run the neural network**, use the [`PINN_3D_real_parms`](https://github.com/ILT-ITMO/LaserProcessing/tree/main/PINN_3D_real_parms) branch – here are the models and instructions for training.

---

## 📖 Documentation

Each component contains its own `README.md` with detailed descriptions:

| Component | Location | Content |
|-----------|---|---|
| Solvers | `solvers/README.md` | Description of physical models, parameters, examples |
| Database | `data_base/README.md` | Dataset structure, data generation algorithms |
| PINN model | `PINN_3D_real_parms/README.md` | Network architecture, training instructions |

---

## 🤝 Authorship

The project was developed by the [ILT-ITMO](https://github.com/ILT-ITMO) team (Institute of Laser Technology, ITMO University).

---

## 💬 Contacts

Email: ilt@itmo.ru
Website: https://ilt.itmo.ru
GitHub: https://github.com/ILT-ITMO/LaserProcessing
# 🧭 Overview

This project provides an interactive Graphical User Interface (GUI) for
predicting the **Uniaxial Compressive Strength ($\sigma_{cj}$)** and
**Deformation Modulus ($E_j$)** of rock specimens containing
non-persistent joints. It utilizes high-performance hybrid Support
Vector Regression (SVR) models optimized via advanced metaheuristic
algorithms (**PSO** and **RSA**).

## 🚀 Model Information

The prediction engine is powered by two optimized machine learning
models:

-   **SVR-PSO**: Support Vector Regression optimized using Particle
    Swarm Optimization for predicting Uniaxial Compressive Strength
    ($\sigma_{cj}$).
-   **SVR-RSA**: Support Vector Regression optimized using Reptile
    Search Algorithm for predicting Deformation Modulus ($E_j$).

Both metaheuristic optimizers act as the hyperparameter search layer of
their SVR regressor. The Reptile Search Algorithm is a nature-inspired
optimizer motivated by the hunting behaviour of crocodiles: an
encircling phase performed by high walking or belly walking, followed by
a hunting phase performed by hunting coordination or hunting
cooperation.

## 🛠️ Installation & Setup

To run this project, you need to have **Python** installed on your
system.

### 1. Clone or Download the Project

Make sure all required files (`SVR_jointed_rock_mass_GUI.py`, the trained model `.pkl`
files, and `requirements.txt`) are located in the same folder.

### 2. Install Dependencies

Open your terminal or **Anaconda Prompt** and run the following command
to install the necessary libraries:

``` bash
pip install -r requirements.txt
```

## 🖥️ How to Run the Application

To launch the Graphical User Interface (GUI), follow these steps:

1.  Navigate to the project folder using the `cd` command. Wrap the path
    in double quotes if it contains spaces:

    ``` bash
    cd /d "C:\Your\Path\To\The\Project\Folder"
    ```

    *(Replace the path above with the actual path where your files are
    located.)*

2.  Execute the prediction script:

    ``` bash
    python gui_predict.py
    ```

3.  Wait a few seconds, and the interactive prediction window will
    appear on your screen.

## 📂 Project Structure

-   `gui_predict.py`: The main Python script containing the graphical
    user interface.
-   `*.pkl`: Pre-trained optimized SVR model files (SVR-PSO and
    SVR-RSA).
-   `requirements.txt`: List of Python libraries required to run the
    project.
-   `README.md`: Project documentation and instructions.

Keep `gui_predict.py` and both `.pkl` model files in the same directory,
since the script loads the trained models by relative path at startup.
The two optimizers expose the same calling convention in the interface,
so a single prediction workflow serves both the $\sigma_{cj}$ and $E_j$
models.

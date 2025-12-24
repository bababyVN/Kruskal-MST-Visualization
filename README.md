# Kruskal's Algorithm Visualization

An interactive visualization tool for Kruskal's Minimum Spanning Tree (MST) algorithm. This application features a fully functional graph editor and a GPU-accelerated simulation mode capable of handling massive graphs (up to 100,000+ vertices).

## Features

* **Interactive Graph Editor**: Draw nodes, connect edges, and arrange graph layout.
* **High-Performance Rendering**: Built with **ModernGL** (OpenGL) to render.
* **Real-Time Simulation**: Watch Kruskal's algorithm find the MST step-by-step or automatically.
* **Large Scale Testing**: Includes a "Test Mode" to instantly generate and process massive random graphs.
* **Save & Load System**: Save your graph layouts (topology + coordinates) or structure-only files.
* **Optimized Logic**: Uses **Numba** (JIT compilation) for graph processing.

## Installation

1.  **Clone the repository** (or download the files).
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Controls

### Global
* **Right Click + Drag**: Pan the camera.
* **Scroll Wheel**: Zoom in/out.

### Editor Mode
* **Left Click**: Use the active tool (Select, Create Node, Create Edge).
* **Delete**: Remove the selected node or edge.
* **Enter**: Switch to **Simulation Mode**.
* **T**: Generate a massive random graph (Stress Test).
* **Save/Load**: Use the buttons in the top-right corner.

### Simulation Mode
* **Slider**: Adjust simulation speed (Drag left for step-by-step, right for auto-speed).
* **Space**: Plause/Play simulation.
* **Left Arrow**: Step backward / Undo.
* **Right Arrow**: Step forward / Resume.
* **R**: Return to **Editor Mode**.
* **F**: Toggle **Tree View**.
* **UI Toggles**: Toggle visibility of IDs, weights, and table data.

## Project Structure

* `main.py`
* `editor.py`
* `gui.py`
* `logic.py`
* `config.py`

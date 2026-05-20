# MiniML
ML library, with its core written in C++, and a Python interface provided via `pybind11` and `scikit-build-core`.
## Disclaimer: Educational Project / Work in Progress
This library is actively under development and serves primarily as a personal learning project to explore C++ performance optimization and Python bindings via pybind11.

It is NOT ready for production use.

## Requirements

- Python 3.8 or newer
- C++ compiler supporting the C++17 standard (e.g., GCC, Clang, MSVC)
- CMake (version 3.15+)

## Installation (Recommended)

**1. Clone the repository and navigate to the directory**
```
git clone https://github.com/olafdom/miniml.git
cd miniml
```

**2. Create an isolated virtual environment**
```
python -m venv .venv
```

**3. Activate the environment**
Depending on your operating system, type the appropriate command in your terminal:

**Windows**  
```
.venv\Scripts\activate
``` 

**Linux / macOS** 
```
source .venv/bin/activate
``` 

*Once the environment is active, the `(.venv)` prefix will appear at the beginning of your terminal prompt.*

**4. Install the package**
Install the library in editable mode (this allows you to make changes to the code without having to reinstall):
```
pip install -e .
```

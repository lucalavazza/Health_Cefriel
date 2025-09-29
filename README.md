# Health_Cefriel

A research project focused on **health data analysis and causal inference**.  
The repository contains scripts for dataset management, time-series analysis, causal modeling (using [DoWhy](https://github.com/py-why/dowhy) and [CausalLearn](https://causal-learn.readthedocs.io)), persona profiling, and influence analysis.  
Outputs such as plots and reports are organized into dedicated folders.

---

## Repository Structure

```
.
├── datasets/                     # Data files (raw and processed)
├── graphs/                       # Generated plots and visualizations
├── participants/                 # Participant-related metadata
├── ppt/                          # Presentation material
├── data_analysis.py               # Exploratory data analysis
├── dataset_management.py          # Dataset preparation utilities
├── causal-learn_analysis.py       # Causal inference using CausalLearn
├── dowhy_analysis.py              # Causal inference using DoWhy
├── influence_analysis.py          # Influence modeling
├── personas_analysis.py           # Personas profiling / clustering
├── time_series_analysis.py        # Time-series analysis (aggregate)
├── time_series_analysis_single_pids.py  # Time-series analysis per participant
├── requirements.txt               # Python dependencies
└── .gitattributes
```

---

## Getting Started

### Prerequisites

- Python **3.8+**
- Virtual environment (recommended)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/lucalavazza/Health_Cefriel.git
   cd Health_Cefriel
   ```

2. (Optional) Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS/Linux
   .\venv\Scripts\activate    # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

Run analysis scripts individually, for example:

```bash
python data_analysis.py
python time_series_analysis.py
python causal-learn_analysis.py
python dowhy_analysis.py
```

- Outputs such as plots and figures are saved under `graphs/`.  
- Modify dataset paths in scripts if needed (see `dataset_management.py`).  
- For participant-level analysis, use:

  ```bash
  python time_series_analysis_single_pids.py
  ```

---

## Modules Overview

- **`data_analysis.py`** → Exploratory data analysis and descriptive statistics  
- **`dataset_management.py`** → Data loading, cleaning, preprocessing  
- **`causal-learn_analysis.py`** → Causal discovery using CausalLearn  
- **`dowhy_analysis.py`** → Causal inference with DoWhy  
- **`influence_analysis.py`** → Influence modeling between variables  
- **`personas_analysis.py`** → Persona creation / clustering from datasets  
- **`time_series_analysis.py`** → General time-series modeling  
- **`time_series_analysis_single_pids.py`** → Time-series modeling for each participant  

---

## Datasets

- Place raw and processed data files inside `datasets/`.  
- Use `dataset_management.py` for dataset preparation.  
- **Note:** Some datasets may be restricted or private—ensure you comply with data usage policies.

---

## Graphs & Outputs

All generated plots and analysis results are stored in the `graphs/` directory.  
Use these in reports or presentations (see `ppt/` for prepared slides).

---

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository  
2. Create a feature branch (`git checkout -b feature/your-feature`)  
3. Commit your changes (`git commit -m "Add new feature"`)  
4. Push to your branch (`git push origin feature/your-feature`)  
5. Open a Pull Request  

Please follow **PEP 8** style guidelines, include **docstrings**, and test before submitting.

---

## License

The MIT License (MIT)

Copyright (c) 2015 Chris Kibble

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

## Acknowledgements

- [DoWhy](https://github.com/py-why/dowhy) – Causal inference framework  
- [CausalLearn](https://github.com/py-why/causal-learn) – Causal discovery algorithms  
- [Python Scientific Stack](https://scipy.org/) – NumPy, pandas, matplotlib, etc.

---

## Credits

Luca Lavazza (University la Sapienza, University of Brescia)

---
# SpikF-GO (Spiking Fourier Graph Operators)

This repository contains the implementation of **SpikF-GO**.

---

## 1. Environment Setup

Create and activate a virtual environment.

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## 2. Dataset

Download the dataset from Figshare:

https://figshare.com/s/7617530bce306584fe95?file=62576929

After downloading, place the dataset files **directly** inside the existing `data/` folder.

**Important**

* Do **not** create subfolders inside `data/`.
* Place each dataset file individually in `data/`.

### Expected structure

```
SpikF-GO/
│── data/
│   │── dataset_file_1
│   │── dataset_file_2
│   │── ...
|── model/
|── utils/
│── scripts/
│   │── ecl.sh
│── requirements.txt
│── train.py
│── README.md
```

---

## 3. Run Experiments

Run scripts are located in the `scripts/` folder.

Example:

```bash
bash scripts/ecl.sh
```

If needed, make the script executable:

```bash
chmod +x scripts/ecl.sh
./scripts/ecl.sh
```

```markdown
# 🚀 Installation Guide — Goodbooks Recommendation System

This guide provides step-by-step instructions to install, set up, and run the **Goodbooks Recommendation System** locally for development, experimentation, or demo purposes.

---

## 📋 System Requirements

### Minimum Requirements
* **Operating System**: macOS 12+, Ubuntu 20.04+, or Windows 10+
* **Python**: 3.10 or higher (3.11 recommended)
* **Memory**: 8 GB RAM
* **Storage**: ~5 GB free space
* **Internet**: Required to download the dataset and Python dependencies

### Recommended Requirements
* **Memory**: 16 GB RAM
* **Storage**: 10 GB+ SSD
* **CPU**: Modern multi-core CPU
* **GPU**: *Not required* (all models run on CPU)

> [!INFO]
> This project is designed to run fully on CPU and is compatible with Streamlit Community Cloud.

---

## 🛠️ Prerequisites Installation

### 1. Python Installation

**macOS (Homebrew)**
```bash
brew install python@3.11

```

**Ubuntu / Debian**

```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-pip

```

**Windows**

1. Download Python 3.11 from [python.org](https://www.python.org/downloads/).
2. Ensure **"Add Python to PATH"** is checked during installation.

**Verify installation:**

```bash
python --version

```

### 2. Git Installation

**macOS**

```bash
brew install git

```

**Ubuntu / Debian**

```bash
sudo apt install git

```

**Windows**

```bash
# Using Chocolatey
choco install git

```

---

## 📦 Project Installation

### Step 1: Clone the Repository

```bash
git clone [https://github.com/](https://github.com/)<your-username>/goodbooks-rec.git
cd goodbooks-rec

```

### Step 2: Create and Activate Virtual Environment

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate

```

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate

```

**Upgrade pip:**

```bash
pip install --upgrade pip

```

### Step 3: Install Python Dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt

```

---

## 📊 Dataset Setup (Required)

This repository does not version the Goodbooks dataset. You must download it locally before running the app or tests.

### Download the Dataset

```bash
bash scripts/download_data.sh

```

**This script performs the following:**

1. Downloads the **Goodbooks-10k** dataset.
2. Extracts CSV files into `./data`.
3. Skips re-downloads if files already exist.

**Verify files:**

```bash
ls data/

```

---

## 🧪 Running Tests

**Run All Tests**

```bash
pytest

```

**Run with Coverage Report**

```bash
pytest --cov=goodbooks_rec --cov-report=term-missing

```

---

## 🚀 Running the Streamlit App

### Start the Application

```bash
streamlit run apps/streamlit_app.py

```

### Access the App

Open your browser at: `http://localhost:8501`

**Features available in the UI:**

* Explore multiple recommendation strategies (Collaborative Filtering, RL-based).
* Switch between user personas.
* Compare model outputs interactively.

---

## 🧩 Troubleshooting

| Error | Fix |
| --- | --- |
| **FileNotFoundError**: `data/books.csv` | Run `bash scripts/download_data.sh` |
| **ModuleNotFoundError** | Ensure your virtual environment is active: `source .venv/bin/activate` |
| **Streamlit Command Not Found** | Run `pip install streamlit` within your virtual environment |

---

✅ **Installation Complete!**
You’re now ready to run experiments, explore hybrid CF + RL recommenders, or deploy the Streamlit demo.

For architecture and model details, see:

* `README.md`
* `docs/Goodbooks_Rec_Documentation.pdf`


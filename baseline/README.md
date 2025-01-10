# Baseline Experiments

```
.
├── LICENSE
├── README.md
└── baseline
    ├── 01_build_dataset.py
    ├── 02_build_tokenizer.py
    ├── 03_training.py
    ├── README.md            # You're reading this!
    ├── config.py
    └── requirements.txt
```

### **`baseline/` Directory**
The `baseline` folder contains the initial implementation of the Transformer model. This version closely follows the original paper and is built with float32 precision.

#### **Files Explained:**
- `01_build_dataset.py` - Script to build the dataset for training.
- `02_build_tokenizer.py` - Script to create a tokenizer compatible with the dataset.
- `03_training.py` - Main script to train the baseline Transformer model.
- `config.py` - Contains configuration details such as model hyperparameters and paths.
- `requirements.txt` - Lists the Python dependencies required to run the code.
- `README.md` - Documentation specific to the baseline implementation.


## 🚀 **Getting Started**
### **1. Clone the Repository**
```bash
git clone https://github.com/your-username/attention-is-all-you-need.git
cd attention-is-all-you-need/baseline
```

### **2. Install Dependencies**
Make sure you have Python 3.8+ installed. Then, install the required packages using:
```bash
pip install -r requirements.txt
```

### **3. Run the Baseline Model**
Start by building the dataset and tokenizer before initiating the training process:
```bash
python 01_build_dataset.py
python 02_build_tokenizer.py
python 03_training.py
```

# **HW3 Handout**

In this assignment, we are introducing a new format for assignment delivery, designed to enhance your development workflow. The key motivations for this change are:

- **Test Suite Integration**: Your code will be tested in a manner similar to `HWP1's`.
- **Local Development**: You will be able to perform most of your initial development locally, reducing the need for compute resources.
- **Hands-on Experience**: This assignment provides an opportunity to build an end-to-end deep learning pipeline from scratch. We will be substantially reducing the amount of abstractions compared to previous assignments.

For our provided notebook's to work, your notebook's current working directory must be the same as the handout.
This is important because the relative imports in the notebook's depend on the current working directory.
This can be achieved by:

1. Physically moving the notebook's into the handout directory.
2. Changing the notebook's current working directory to the handout directory using the `os.chdir()` function.

Your current working directory should have the following files for this assignment:

```
.
├── README.md
├── hw3lib/
├── tests/
├── hw3_data_subset/
└── requirements.txt
```

## 📊 Dataset Structure

We have provided a subset of the dataset for you to use. This subset has been provided with the intention of allowing you to implement and test your code locally. The full dataset is the Kaggle dataset [`cmu11785/f26-hw3p2-data`](https://www.kaggle.com/datasets/cmu11785/f26-hw3p2-data), and the setup steps below show how to get it in each environment. The subset follows the same structure as the original dataset and is organized as follows:

```
hw3_data_subset/
└── hw3p2_data/ # For end-to-end speech recognition
├── dev-clean/
│ ├── fbank/
│ └── text/
├── test-clean/
│ └── fbank/
└── train-clean-100/
├── fbank/
└── text/

```

## 🔧 Implementation Files

### Main Library (`hw3lib/`)

For `HW3P2`, you will incrementally implement components of `hw3lib` to build and train an _Encoder-Decoder Transformer_ for end-to-end speech recognition.

The components are built up one at a time, each reusing the ones before it, so that the pieces stay small enough to test in isolation. You should see the following files in the `hw3lib/` directory (`__init__.py`'s are not shown):

```
hw3lib/
├── data/
│ ├── tokenizer_jsons/
│ ├── asr_dataset.py
│ ├── tokenizer.py
│ └── verify_dataloader.py
├── decoding/
│ └── sequence_generator.py
├── model/
│ ├── masks.py
│ ├── positional_encoding.py
│ ├── norms.py
│ ├── attention.py
│ ├── speech_embedding.py
│ ├── sublayers.py
│ ├── decoder_layers.py
│ ├── encoder_layers.py
│ └── transformers.py
├── trainers/
│  ├── base_trainer.py
│  └── asr_trainer.py
└── utils/
   ├── create_lr_scheduler.py
   └── create_optimizer.py
```

You will build these Transformer models in PyTorch, including your own multi-head attention in `hw3lib/model/attention.py`. It is the PyTorch version of the `mytorch` multi-head attention you implemented from scratch in `HW3P1`, so the interface should already be familiar.

### Test Suite (`tests/`)

In `HW3P2`, you will be provided with a test suite that will be used to test your implementation. You should see the following files in the `tests/` directory:

```

tests/
├── testing_framework.py
├── test_dataset_asr.py
├── test_mask_padding.py
├── test_mask_causal.py
├── test_positional_encoding.py
├── test_rope.py
├── test_norms.py
├── test_attention.py
├── test_sublayer_selfattention.py
├── test_sublayer_feedforward.py
├── test_sublayer_crossattention.py
├── test_decoderlayer_crossattention.py
├── test_encoderlayer_selfattention.py
├── test_transformer_encoder_decoder.py
└── test_decoding.py

```

## Setup

Follow the setup instructions based on your preferred environment!

### Local

One of our key goals in designing this assignment is to allow you to complete most of the preliminary implementation work locally.  
We highly recommend that you **pass all tests locally** using the provided `hw3_data_subset` before moving to a GPU runtime.  
To do this, simply:

#### Step 1: Create a new conda environment

```bash
# Be sure to deactivate any active environments first
conda create -n hw3 python=3.12.4
```

#### Step 2: Activate the conda environment

```bash
conda activate hw3
```

#### Step 3: Install the dependencies using the provided `requirements.txt`

```bash
pip install --no-cache-dir --ignore-installed -r requirements.txt
```

#### Step 4: Ensure that your notebook is in the same working directory as the `Handout`

This can be achieved by:

1. Physically moving the notebook into the handout directory.
2. Changing the notebook’s current working directory to the handout directory using the os.chdir() function.

#### Step 5: Open the notebook and select the newly created environment from the kernel selector.

If everything was done correctly, You should see atleast the following files in your current working directory after running `!ls`:

```
.
├── README.md
├── requirements.txt
├── hw3lib/
├── tests/
└── hw3_data_subset/
```

### Colab

#### Step 1: Get your handout

- See writeup for recommended approaches.

##### Example: My preferred approach

```python
import os

# Settings -> Developer Settings -> Personal Access Tokens -> Token (classic)
os.environ['GITHUB_TOKEN'] = "your-token"

GITHUB_USERNAME = "your-username"
REPO_NAME = "IDL-HW3" # Your repo's name. The steps below use this variable
TOKEN = os.environ.get("GITHUB_TOKEN")
repo_url = f"https://{TOKEN}@github.com/{GITHUB_USERNAME}/{REPO_NAME}.git"
!git clone {repo_url}

## -------------

# To pull latest changes (Must be in the repo dir, use pwd/ls to verify)
!cd {REPO_NAME} && git pull
```

#### Step 2: Install Dependencies

- `NOTE`: Colab already has PyTorch and most of what this homework needs, so this only installs the packages it is missing.
- `NOTE`: Do not install `requirements.txt` on Colab. It pins versions for the local setup, and Colab's Python cannot install some of them.

```python
!pip install wandb==0.23.1 torchinfo torchmetrics kaggle==1.8.0 "kagglehub>=1.0.0" --quiet
```

#### Step 3: Obtain Data

- `NOTE`: This downloads the `HW3P2` dataset from Kaggle and unzips it to `/content/hw3p2-data/hw3p2_data`.

```bash
!curl -L -o /content/f26-hw3p2-data.zip https://www.kaggle.com/api/v1/datasets/download/cmu11785/f26-hw3p2-data
!unzip -q -o /content/f26-hw3p2-data.zip -d /content
!rm -rf /content/f26-hw3p2-data.zip
!du -h --max-depth=2 /content/hw3p2-data/hw3p2_data
```

#### Step 4: Move to Handout Directory

```python
import os
os.chdir(REPO_NAME)
!ls
```

You must be within the handout directory for the library imports to work!

- `NOTE`: You may have to repeat running this command anytime you restart your runtime.
- `NOTE`: You can do a `pwd` to check if you are in the right directory.
- `NOTE`: Your data is in `/content/hw3p2-data/hw3p2_data`. Use this path for `root` in the config file.

If everything was done correctly, You should see atleast the following files in your current working directory after running `!ls`:

```
.
├── README.md
├── requirements.txt
├── hw3lib/
├── tests/
└── hw3_data_subset/
```

### Kaggle

While it is possible to run the notebook on Kaggle, we would recommend against it. This assignment is more resource intensive and may run slower on Kaggle.

#### Step 1: Get your handout

- See writeup for recommended approaches.

##### Example: My preferred approach

```python
import os

# Settings -> Developer Settings -> Personal Access Tokens -> Token (classic)
os.environ['GITHUB_TOKEN'] = "your-token"

GITHUB_USERNAME = "your-username"
REPO_NAME = "IDL-HW3" # Your repo's name. The steps below use this variable
TOKEN = os.environ.get("GITHUB_TOKEN")
repo_url = f"https://{TOKEN}@github.com/{GITHUB_USERNAME}/{REPO_NAME}.git"
!git clone {repo_url}

## -------------

# To pull latest changes (Must be in the repo dir, use pwd/ls to verify)
!cd {REPO_NAME} && git pull
```

#### Step 2: Install Dependencies

Turn on `Internet` in the notebook settings, then install the extra packages:

```python
!pip install wandb==0.23.1 torchinfo torchmetrics kaggle==1.8.0 "kagglehub>=1.0.0" --quiet
```

#### Step 3: Obtain Data

##### ⚠️ Important: Kaggle Users

If you are using Kaggle, **do not manually download the data!** The dataset is large and may exceed your available disk space. Instead, follow these steps to add the dataset directly to your notebook:

1. Open your **Kaggle Notebook**.
2. Navigate to **Notebook → Input**.
3. Click **Add Input**.
4. In the search bar, paste the following URL:

👉 [https://www.kaggle.com/datasets/cmu11785/f26-hw3p2-data](https://www.kaggle.com/datasets/cmu11785/f26-hw3p2-data)

5. Click the **➕ (plus sign)** to add the dataset to your notebook.

##### 📌 Note:

Run the following to print where Kaggle put the data, for example `/kaggle/input/datasets/cmu11785/f26-hw3p2-data/hw3p2-data/hw3p2_data`, and use that path for `root` in the config file:

```python
import glob
# The handout ships a small hw3_data_subset/hw3p2_data of its own, which is not the dataset
print([p for p in glob.glob('/kaggle/input/**/hw3p2_data', recursive=True) if 'hw3_data_subset' not in p])
```

#### Step 4: Move to Handout Directory

```python
import os
os.chdir(REPO_NAME)
!ls
```

You must be within the handout directory for the library imports to work!

- `NOTE`: You may have to repeat running this command anytime you restart your runtime.
- `NOTE`: You can do a `pwd` to check if you are in the right directory.
- `NOTE`: Use the path printed in Step 3 for `root` in the config file.

If everything was done correctly, You should see atleast the following files in your current working directory after running `!ls`:

```
.
├── README.md
├── requirements.txt
├── hw3lib/
├── tests/
└── hw3_data_subset/
```

### PSC

#### Step 1: Setting Up Your Environment on Bridges2

For this homework, we are providing a shared Conda environment for the entire class. Follow these steps to set up the environment and start a Jupyter notebook on Bridges2:

##### 1. SSH into Bridges2

```bash
ssh username@bridges2.psc.edu
```

##### 2. Navigate to your Jet Home Directory

```bash
cd /jet/home/<your_username>
```

##### 3. Load the Anaconda Module

```bash
module load anaconda3
```

##### 4. Activate the provided HW3 Environment

```bash
# First, deactivate any existing Conda environment
conda deactivate
conda activate /ocean/projects/cis250019p/mzhang23/TA/envs/IDLF26 && export PYTHONNOUSERSITE=1
```

##### 5. Request a Compute Node

```bash
interact -p GPU-shared --gres=gpu:v100-32:1 -t 8:00:00 -A cis250019p
```

##### 6. Re-activate Environment

If your Conda environment was deactivated due to node allocation:

```bash
# First, deactivate any existing Conda environment
conda deactivate
conda activate /ocean/projects/cis250019p/mzhang23/TA/envs/IDLF26 && export PYTHONNOUSERSITE=1
```

##### 7. Start Jupyter Notebook

```bash
jupyter notebook --no-browser --ip=0.0.0.0
```

##### 8. Connect to Jupyter Server

You can now connect to the Jupyter Server in whichever way you prefer.

The following is my preferred way of connecting to the Jupyter Server:

###### 8.1 Connect in VSCode

I prefer uploading the notebook to my Jet home directory (`/jet/home/<your_username>`) on Bridges2 and then connecting to the Jupyter Server from there.

1. Use Remote - SSH to connect to Bridges2 and navigate to your Jet home directory.

2. Upload the notebook to that directory.

3. Open the notebook in VSCode.

4. Go to **Kernel** → **Select Another Kernel** → **Existing Jupyter Server**

5. Enter the URL of the Jupyter Server:`http://{hostname}:{port}/tree?token={token}`

- eg: `http://v011.ib.bridges2.psc.edu:8888/tree?token=e4b302434e68990f28bc2b4ae8d216eb87eecb7090526249`

> **Note**: Replace `{hostname}`, `{port}` and `{token}` with your actual values from the Jupyter output.

#### Step 2: Get your handout

- See writeup for recommended approaches.
- If you use Remote - SSH to connect to Bridges2, you can upload the handout to your Jet home directory (`/jet/home/<your_username>`) and work from there.

##### Example: My preferred approach

```python
import os

# Settings -> Developer Settings -> Personal Access Tokens -> Token (classic)
os.environ['GITHUB_TOKEN'] = "your-token"

GITHUB_USERNAME = "your-username"
REPO_NAME = "IDL-HW3" # Your repo's name. The steps below use this variable
TOKEN = os.environ.get("GITHUB_TOKEN")
repo_url = f"https://{TOKEN}@github.com/{GITHUB_USERNAME}/{REPO_NAME}.git"
!git clone {repo_url}

## -------------

# To pull latest changes (Must be in the repo dir, use pwd/ls to verify)
!cd {REPO_NAME} && git pull
```

#### Step 3: Get Data

- `NOTE`: There is no shared copy of the dataset on PSC. This downloads the `HW3P2` dataset from Kaggle and unzips it to `$LOCAL/hw3p2-data/hw3p2_data`.
- `NOTE`: We are using `$LOCAL`: the scratch storage on local disk on the node running a job to store out data.
  - Disk accesses are much faster than what you would get from `$PROJECT` storage
  - `IT IS NOT PERSISTENT`, so you have to download the data again every time you move to a new node
- `NOTE`: Make sure you have completed the previous steps before running this cell.
- Read more about it PSC File Spaces [here](https://www.psc.edu/resources/bridges-2/user-guide#file-spaces).

```python
import os
os.environ['KAGGLE_USERNAME'] = "<your-username>" # TODO: Verify in Settings
os.environ['KAGGLE_API_TOKEN'] = "<your-key>" # TODO: Add your API token, KGAT_ followed by 32 letters and digits

# The new API token starts with KGAT_. The 32-character key in an older kaggle.json does not work here.
os.environ['KAGGLE_API_TOKEN'] = os.environ['KAGGLE_API_TOKEN'].strip()
assert os.environ['KAGGLE_API_TOKEN'].startswith("KGAT_"), "Use the API token that starts with KGAT_ (Kaggle Profile → Settings → Generate New Token)"

import kaggle
api = kaggle.api
api.dataset_download_files("cmu11785/f26-hw3p2-data", path=os.environ["LOCAL"], unzip=True)
!du -h --max-depth=2 $LOCAL/hw3p2-data/hw3p2_data
```

#### Step 4: Move to Handout Directory

Depending on the way you are running your notebook, you may or may not need to run this cell. As long as you are within the handout directory for the library imports to work!

```python
# Move to the handout directory if you are not there already
import os
os.chdir(REPO_NAME)
!ls
```

- `NOTE`: You may have to repeat running this command anytime you restart your runtime.
- `NOTE`: You can do a `pwd` to check if you are in the right directory.
- `NOTE`: Your data is in `$LOCAL/hw3p2-data/hw3p2_data`, which is `/local/hw3p2-data/hw3p2_data`. Use this path for `root` in the config file.

If everything was done correctly, You should see atleast the following files in your current working directory after running `!ls`:

```
.
├── README.md
├── requirements.txt
├── hw3lib/
├── tests/
└── hw3_data_subset/
```

---

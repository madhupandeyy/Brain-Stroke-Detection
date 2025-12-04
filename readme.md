

````markdown
# Brain Stroke Detection Web App

> **Short description:** A Flask-based web application that loads trained ML/DL models to predict brain stroke from uploaded images. It stores patient records in CSV and uses MySQL for user authentication, providing a web UI for users and admins.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Prerequisites](#prerequisites)
3. [Detailed Installation Steps](#detailed-installation-steps)
   - Environment Setup (Anaconda)
   - Install Dependencies
   - MySQL Setup (XAMPP)
4. [Files & Folders the App Expects](#files--folders-the-app-expects)
5. [Run the Project](#run-the-project)
6. [Project Folder Structure](#project-folder-structure)

---

## Project Overview
This Flask web application:  
- Accepts brain scan image uploads through a web interface.  
- Uses prediction functions (`pred_skin_disease`, `pred_skin_disease3`) from model modules (`model_predict2.py`, `model_predict2un.py`).  
- Stores prediction results and patient information in `patient_data.csv` and MySQL (`ddbb`) for user authentication.  
- Provides registration/login pages and pages to display prediction results.

> **Note:** The main Flask file is `app.py` and the app starts with `python app.py`.

---

## Prerequisites
- Anaconda (recommended) or Python 3.10+ installed.  
- XAMPP (for MySQL + Apache) installed.  
- Basic knowledge of terminal / Anaconda Prompt.

---

## Detailed Installation Steps

### 1) Environment Setup (Anaconda)
```bash
conda create -n leaf_disease python=3.10.12 -y
conda activate leaf_disease

# Optional: install Jupyter kernel
pip install ipykernel
python -m ipykernel install --user --name leaf_disease --display-name "leaf_disease"
conda install jupyter -y
````

### 2) Install Dependencies

Create a `requirements.txt` file with:

```
Flask==2.0.2
Werkzeug==2.3.7
numpy==1.26.4
pandas==2.1.1
requests
Pillow==11.1.0
reportlab==4.0.6
matplotlib==3.10.0
tensorflow==2.13.0
opencv-python==4.8.1.78
xgboost==1.5.2
pydub==0.25.1
pygame==2.1.2
gTTS==2.2.3
scikit-learn==1.3.2
editdistance==0.6.0
lmdb==1.3.0
PyMySQL==0.10.0
scikit-image==0.19.1
fuzzywuzzy==0.18.0
path==16.2.0
seaborn==0.13.2
typing_extensions==4.12.2
efficientnet==1.1.1
tqdm
ipykernel
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 3) MySQL Setup (XAMPP)

1. Start **Apache** and **MySQL** in XAMPP.
2. Open phpMyAdmin at `http://localhost/phpmyadmin`.
3. Create a database named `ddbb`.
4. Create a `user_register` table:

```sql
CREATE TABLE user_register (
  id INT AUTO_INCREMENT PRIMARY KEY,
  user VARCHAR(255) NOT NULL,
  password VARCHAR(255) NOT NULL
);

-- Optional test user
INSERT INTO user_register (user, password) VALUES ('test@example.com','password123');
```

> The app will automatically create/update `patient_data.csv` when it runs, so no manual CSV creation is needed.

---

## Files & Folders the App Expects

* `app.py` (main Flask file)
* `model_predict2.py` (`pred_skin_disease` function)
* `model_predict2un.py` (`pred_skin_disease3` function)
* `templates/` (HTML files like `home1.html`, `login44.html`, `register44.html`, `index.html`, `patient_info.html`, `rust-result.html`, etc.)
* `static/` (CSS, JS, images)
* `patient_data.csv` (auto-created by the app)
* `name.pkl` (stores current patient ID/name)
* Saved ML/DL models that `model_predict2.py` / `model_predict2un.py` load

---

## Run the Project

1. Activate the environment:

```bash
conda activate leaf_disease
```

2. Start XAMPP and ensure `ddbb` database exists.
3. Run Flask:

```bash
python app.py
```

4. Open `http://127.0.0.1:5000/` in your browser.

---

## Project Folder Structure

```
APPLICATION_FINALISED_BRAIN_SRTOKE_UNKNOWN/
│
├── notebooks/
│   ├── brain_stroke_model.h5
│   ├── efficientnet_brain_stroke_unknown2.weights.h5
│   ├── final_brain_stroke_model.ipynb
│   └── unknown-main-code.ipynb
│
├── static/
│   └── (CSS, JS, images)
│
├── templates/
│   └── (HTML templates)
│
├── app.py
├── config.py
├── model_predict2.py
├── model_predict2un.py
├── brain_stroke_model.h5
├── efficientnet_brain_stroke_unknown2.weights.h5
├── efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5
├── name.pkl
├── output.png
├── output2.png
├── patient_data.csv
├── patient_data - Copy.csv
├── patient_predictions.csv
├── readme.md
└── unknown-main-code.ipynb
```

---


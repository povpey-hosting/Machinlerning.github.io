# 🛍️ Customer Segmentation – K-Means (Flask)

## របៀបដំណើរការក្នុង VS Code
1. បើក folder នេះក្នុង VS Code (`File > Open Folder`)
2. បើក Terminal (`Ctrl + ~`) បង្កើត virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # macOS / Linux
   ```
3. ដំឡើង library: `pip install -r requirements.txt`
4. (ជម្រើស) ដាក់ `Mall_Customers.csv` ក្នុង folder `data/`
5. ដំណើរការ: `python app.py`
6. បើក browser ទៅ **http://127.0.0.1:5000**

> ត្រូវការ Internet ដើម្បីផ្ទុក Chart.js និង Google Fonts។

## រចនាសម្ព័ន្ធ
```
app.py             Flask routes (API)
segmentation.py    K-Means logic
templates/         HTML
static/css/        CSS
static/js/         JavaScript (charts)
data/              Mall_Customers.csv
```

  
**COMPAS Dashboard**

Implementation Guide

Streamlit Dashboard for Interpretable & Fair Recidivism Prediction

Prepared for: Data and Cloud

Cloud Deployment Demo: March 30, 2026

NOTE: The figures in this document might not be accurate for our project, change as needed. I just needed some example numbers to get points across.

# **Table of contents**

1\. Design System & Color Palette  
2\. Page Structure & Navigation  
3\. Page 1: Home Dashboard  
4\. Page 2: Prediction Tool  
5\. Page 3: Fairness Analysis  
6\. Page 4: Model Comparison  
7\. Page 5: Geographic Generalization  
8\. Page 6: About  
9\. Technical Implementation Notes  
10\. File Structure & Dependencies  
11\. Picture mockups 

# **1\. Design system & color palette**

## **1.1 Color palette (Ocean Gradient theme)**

Use these exact hex values for consistency across all pages:

| Color Name | Hex Code | Usage |
| :---- | :---- | :---- |
| Primary Navy | \#0F172A | Sidebar background, dark headers |
| Secondary Navy | \#1E3A5F | Sidebar gradient end, secondary elements |
| Accent Blue | \#0EA5E9 | Primary accent, active states, links, highlights |
| Cyan | \#06B6D4 | Secondary accent, subtitles |
| Success Green | \#10B981 | Fair status, positive metrics, Georgia data |
| Warning Amber | \#F59E0B | Moderate status, warnings |
| Danger Red | \#EF4444 | Unfair status, high risk, errors |
| Text Primary | \#1E293B | Main body text |
| Text Muted | \#64748B | Secondary text, labels, captions |
| Light Gray | \#F8FAFC | Page background |
| Border Gray | \#E2E8F0 | Card borders, dividers |

## **1.2 Typography**

| Element | Size | Style |
| :---- | :---- | :---- |
| Page Title | 20-22px | Font-weight: 500, color: text-primary |
| Section Header | 14-16px | Font-weight: 500, color: text-primary |
| Body Text | 13-14px | Font-weight: 400, color: text-primary |
| Labels/Captions | 11-12px | Font-weight: 400, color: text-muted (\#64748B) |
| Metric Values | 24-28px | Font-weight: 500, color varies by context |

## **1.3 Component styles**

### **Cards**

* Background: white (\#FFFFFF)  
* Border: 0.5px solid \#E2E8F0  
* Border-radius: 12px  
* Padding: 16-20px  
* Shadow (optional): 0 2px 8px rgba(0,0,0,0.08)

### **Metric cards**

* Background: \#F8FAFC (light gray)  
* Border-radius: 8px  
* Padding: 16px  
* Label: 12px muted text above  
* Value: 24px bold below

### **Status badges**

* Fair/Good: background \#DCFCE7, text \#166534  
* Warning/Moderate: background \#FEF3C7, text \#92400E  
* Unfair/Bad: background \#FEE2E2, text \#991B1B  
* Info: background \#DBEAFE, text \#1E40AF  
* Padding: 4px 12px, border-radius: 12px (pill shape)

# **2\. Page structure & navigation**

## **2.1 Sidebar navigation**

The sidebar should have two states: expanded (220px) and collapsed (56px). Use Streamlit's native sidebar with custom CSS styling.

### **Navigation items**

| Icon | Page Name | File Name |
| :---- | :---- | :---- |
| 🏠 | Home | pages/1\_🏠\_Home.py |
| 🎯 | Prediction tool | pages/2\_🎯\_Prediction.py |
| ⚖️ | Fairness analysis | pages/3\_⚖️\_Fairness.py |
| 📊 | Model comparison | pages/4\_📊\_Models.py |
| 🗺️ | Generalization | pages/5\_🗺️\_Generalization.py |
| ℹ️ | About | pages/6\_ℹ️\_About.py |

### **Sidebar CSS styling**

*Add this to .streamlit/config.toml or inject via st.markdown:*

\[theme\]primaryColor \= '\#0EA5E9'backgroundColor \= '\#F8FAFC'secondaryBackgroundColor \= '\#FFFFFF'textColor \= '\#1E293B'

# **3\. Page 1: Home dashboard**

## **3.1 Layout overview**

The home page provides a high-level overview of the dataset and model performance. It should load quickly and give users immediate insights.

### **Components (top to bottom)**

1. **Header:** Page title \+ subtitle \+ dataset selector dropdown  
2. **Metric cards row:** 4 cards in a row (use st.columns(4))  
3. **Charts row:** 2 charts side by side (use st.columns(2))  
4. **CTA banner:** Call-to-action linking to prediction tool

## **3.2 Metric cards**

| Metric | Value | Subtext | Color |
| :---- | :---- | :---- | :---- |
| Total records | 7,214 | Florida dataset | Green (\#10B981) |
| Recidivism rate | 45.1% | 2-year window | Amber (\#F59E0B) |
| Best model AUC | 0.74 | Random Forest | Blue (\#0EA5E9) |
| Disparate impact | 0.58 | Below 0.8 threshold | Red (\#EF4444) |

## **3.3 Charts**

### **Chart 1: Recidivism rate by race (horizontal bar)**

* African American: 52.3% (color: \#0EA5E9)  
* Caucasian: 39.2% (color: \#1E3A5F)  
* Hispanic: 35.4% (color: \#06B6D4)  
* Other: 41.1% (color: \#64748B)  
* **Use:** st.plotly\_chart() with go.Bar(orientation='h')

### **Chart 2: Model performance comparison (horizontal bar)**

* COMPAS: 65.4% (color: \#64748B)  
* Logistic Regression: 67.2% (color: \#0EA5E9)  
* Random Forest: 69.1% (color: \#10B981)  
* Decision Tree: 66.3% (color: \#8B5CF6)

# **4\. Page 2: Prediction tool**

## **4.1 Layout**

Two-column layout: input form on left (340px), results on right (flex).

## **4.2 Input form fields**

| Field | Type | Options/Range |
| :---- | :---- | :---- |
| Age | st.number\_input | min=18, max=80, default=34 |
| Sex | st.selectbox | \['Male', 'Female'\] |
| Race | st.selectbox | \['African American', 'Caucasian', 'Hispanic', 'Other'\] (we can choose not to include Hispanic or Other) |
| Prior crimes count | st.number\_input | min=0, max=40, default=3 |
| Juvenile felonies | st.number\_input | min=0, max=10, default=0 |
| Charge degree | st.selectbox | \['Felony', 'Misdemeanor'\] |
| Model | st.selectbox | \['Logistic Regression', 'Random Forest', 'Decision Tree', 'XGBoost \+ Debiasing'\] |

## **4.3 Results display**

### **Risk gauge**

* Circular gauge showing prediction probability (0-100%)  
* Use Plotly gauge chart: go.Indicator(mode='gauge+number')  
* Color ranges: 0-30% green, 30-60% amber, 60-100% red  
* Below gauge: status badge ('Low risk', 'Medium risk', 'High risk')

### **Multi-model comparison**

* Run same input through all models  
* Show horizontal bar for each model's prediction  
* Helps user understand model variance

### **SHAP waterfall explanation**

* Show feature contributions to prediction  
* Use shap.plots.waterfall() or custom Plotly implementation  
* Red bars \= increases risk, Green bars \= decreases risk  
* Start from base value (population average), end at prediction  
* Include interpretation text box below explaining key drivers

# **5\. Page 3: Fairness analysis**

## **5.1 Controls**

* Protected attribute selector: Race (default), Sex  
* Model selector: Random Forest, Logistic Regression, Decision Tree

## **5.2 Fairness metrics cards (4 columns)**

| Metric | Value | Status | Threshold |
| :---- | :---- | :---- | :---- |
| Disparate Impact (DI) | 0.58 | Unfair (red) | ≥0.8 is fair |
| Statistical Parity Diff (SPD) | \-0.18 | Unfair (red) | ±0.1 is fair |
| Equal Opportunity Diff (EOD) | \-0.12 | Moderate (amber) | ±0.1 is fair |
| Equalized Odds Diff | \-0.09 | Moderate (amber) | ±0.1 is fair |

## **5.3 Charts**

### **Chart 1: Error rates by demographic group**

* Side-by-side bar charts for FPR and FNR  
* FPR: African American 44.8%, Caucasian 23.5%  
* FNR: African American 28.0%, Caucasian 47.7%  
* Highlight: African Americans have 1.9x higher FPR

### **Chart 2: Calibration plot**

* Scatter plot: predicted probability vs actual recidivism rate  
* Separate series for each demographic group  
* Include ideal diagonal line (perfect calibration)

## **5.4 Debiasing comparison table**

| Method | Accuracy | AUC | DI | SPD | Status |
| :---- | :---- | :---- | :---- | :---- | :---- |
| Baseline | 69.1% | 0.74 | 0.58 | \-0.18 | Unfair |
| Reweighing | 66.8% | 0.71 | 0.72 | \-0.11 | Improved |
| Equalized Odds | 64.2% | 0.68 | 0.85 | \-0.05 | Fair |
| Reject Option (recommended) | 65.5% | 0.70 | 0.81 | \-0.07 | Fair |

# **6\. Page 4: Model comparison**

## **6.1 Model cards (4 columns)**

Display 4 model cards side by side. Highlight 'Recommended' model (Logistic Regression) with blue border.

| Model | Accuracy | AUC | Precision | Recall | Tags |
| :---- | :---- | :---- | :---- | :---- | :---- |
| COMPAS | 65.4% | 0.68 | 0.63 | 0.59 | Black-box, Unfair |
| Logistic Reg. | 67.2% | 0.72 | 0.66 | 0.64 | Interpretable |
| Random Forest | 69.1% | 0.74 | 0.68 | 0.66 | Best accuracy |
| Decision Tree | 66.3% | 0.70 | 0.64 | 0.62 | Interpretable |

## **6.2 ROC curves**

* Use sklearn.metrics.roc\_curve() for each model  
* Plot all 4 curves on same chart with legend showing AUC  
* Include diagonal reference line (random classifier)  
* Colors: RF=\#10B981, LR=\#0EA5E9, DT=\#8B5CF6, COMPAS=\#64748B

## **6.3 Confusion matrix**

* Dropdown to select model  
* 2x2 grid with colored cells: TN (green), TP (blue), FP (red), FN (amber)  
* Show count in each cell  
* Use Plotly heatmap or custom HTML/CSS

# **7\. Page 5: Geographic generalization**

## **7.1 Research question banner**

Dark gradient banner (\#0F172A to \#1E3A5F) displaying: "Do recidivism models trained in one jurisdiction generalize to another?"

## **7.2 Transfer performance cards (3 columns)**

| Scenario | Accuracy | AUC |
| :---- | :---- | :---- |
| Florida → Florida (baseline) | 69.1% | 0.74 |
| Florida → Georgia (transfer) | 61.3% | 0.65 (-7.8% drop) |
| Georgia → Georgia (retrained) | 67.8% | 0.72 |

## **7.3 Feature importance comparison chart**

* Grouped horizontal bar chart showing importance in FL vs GA  
* Features: priors\_count, age, juv\_fel\_count, offense\_type, sex  
* Florida bars in blue (\#0EA5E9), Georgia bars in green (\#10B981)

## **7.4 Dataset comparison table**

| Attribute | Florida | Georgia |
| :---- | :---- | :---- |
| Records | 7,214 | 25,835 |
| Recidivism rate | 45.1% | 52.3% |
| Mean age | 34.8 | 31.2 |
| % African American | 51% | 62% |
| Avg prior crimes | 2.7 | 3.4 |

## **7.5 Key findings section**

Display 3 colored info boxes:

5. **Performance gap (amber):** 7.8% accuracy drop when transferring  
6. **Transferable features (green):** priors\_count and age remain strong in both  
7. **Recommendation (blue):** Retrain models on local data

# **8\. Page 6: About**

## **8.1 Project overview banner**

Dark gradient banner with project title, description, and 4 stat counters (4 RQs, 33K+ records, 5 models, 2 states).

## **8.2 Sections to include**

8. **Research questions:** List all 4 RQs with badges (RQ1-RQ4)  
9. **Team members:** Avatar \+ name \+ role for each of 6 team members  
10. **Data sources:** COMPAS (Florida) and NIJ (Georgia) with record counts  
11. **Technology stack:** Pill badges for Python, Scikit-learn, XGBoost, AIF360, SHAP, Streamlit, Plotly, Pandas  
12. **Key references:** Dressel & Farid (2018), Rudin et al. (2020), Wang et al. (2022)  
13. **Footer:** University of Alberta, INT D 491 B1, Winter 2026

# **9\. Technical implementation notes**

## **9.1 Streamlit configuration**

\# .streamlit/config.toml\[theme\]primaryColor \= '\#0EA5E9'backgroundColor \= '\#F8FAFC'secondaryBackgroundColor \= '\#FFFFFF'textColor \= '\#1E293B'font \= 'sans serif'\[server\]headless \= trueport \= 8501

## **9.2 Custom CSS injection**

Add at top of each page for consistent styling:

st.markdown('''\<style\>    .stMetric { background: \#F8FAFC; padding: 16px; border-radius: 8px; }    .stMetric label { color: \#64748B; font-size: 12px; }    .stMetric \[data-testid="metric-value"\] { font-size: 24px; font-weight: 500; }    div\[data-testid="stSidebar"\] { background: linear-gradient(180deg, \#0F172A, \#1E3A5F); }\</style\>''', unsafe\_allow\_html=True)

## **9.3 Data loading pattern**

@st.cache\_datadef load\_data():    florida\_df \= pd.read\_csv('data/compas\_florida.csv')    georgia\_df \= pd.read\_csv('data/nij\_georgia.csv')    return florida\_df, georgia\_df@st.cache\_resourcedef load\_models():    models \= {        'Logistic Regression': joblib.load('models/logistic\_reg.pkl'),        'Random Forest': joblib.load('models/random\_forest.pkl'),        'Decision Tree': joblib.load('models/decision\_tree.pkl')    }    return models

## **9.4 Key libraries to install**

\# requirements.txtstreamlit\>=1.28.0pandas\>=2.0.0numpy\>=1.24.0scikit-learn\>=1.3.0xgboost\>=2.0.0shap\>=0.43.0aif360\>=0.5.0plotly\>=5.17.0joblib\>=1.3.0

# **10\. File structure & dependencies**

## **10.1 Recommended project structure**

compas-dashboard/├── app.py                          \# Main entry point├── requirements.txt├── .streamlit/│   └── config.toml                 \# Theme configuration├── pages/│   ├── 1\_🏠\_Home.py│   ├── 2\_🎯\_Prediction.py│   ├── 3\_⚖️\_Fairness.py│   ├── 4\_📊\_Models.py│   ├── 5\_🗺️\_Generalization.py│   └── 6\_ℹ️\_About.py├── utils/│   ├── \_\_init\_\_.py│   ├── data\_loader.py              \# Data loading functions│   ├── models.py                   \# Model loading & prediction│   ├── fairness.py                 \# AIF360 fairness metrics│   ├── shap\_utils.py               \# SHAP explanation helpers│   └── charts.py                   \# Plotly chart functions├── data/│   ├── compas\_florida.csv│   ├── nij\_georgia.csv│   └── processed/                  \# Preprocessed data├── models/│   ├── logistic\_reg.pkl│   ├── random\_forest.pkl│   ├── decision\_tree.pkl│   └── xgboost\_debiased.pkl└── assets/    └── style.css                   \# Custom CSS

## **10.2 Deployment checklist**

* Streamlit Cloud: Connect GitHub repo, set Python version to 3.10+  
* Ensure all data files are under 200MB for Streamlit Cloud  
* Add .gitignore for \_\_pycache\_\_, .pyc, .env files  
* Test locally with: streamlit run app.py  
* Verify all pages load without errors before demo

## **10.3 Demo day reminders**

* **Cloud Deployment Demo: March 30, 2026 (20% of grade)**  
* Prepare 2-3 demo scenarios with pre-filled inputs  
* Have backup local deployment ready in case of internet issues  
* Test on multiple browsers (Chrome, Firefox, Safari)  
* Ensure mobile responsiveness (instructor may view on phone)

# **11\. Picture Mockups**

## **11.1 Home Page**

![][image1]

## **11.2 Prediction Tool Page**

![][image2]

## **11.3 Fairness Analysis Page**

![][image3]

## **11.4  Geographic Generalization Page**

![][image4]

## **11.5 Model Comparison Page**

![][image5]

## **11.6  About Page**


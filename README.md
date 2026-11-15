# Global Supply Chain Analytics & Delivery Risk Dashboard

CEN445 Introduction to Data Visualization – Team Project (2026–27)

An interactive Streamlit dashboard that explores 20,000 DataCo supply-chain order lines (January 2015 – January 2018) to show where deliveries run late, where profit is lost, and whether late deliveries can be predicted.

## Dataset

- **Source:** [DataCo Smart Supply Chain for Big Data Analysis](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis) (Kaggle), file `DataCoSupplyChainDataset.csv`
- **Raw size:** ~180,000 rows × 53 columns
- **Cleaned file in this repo:** `cleaned_supply_chain.csv` – 20,000 rows × 27 columns (24 original + 3 derived), 5 markets, 154 countries, 50 categories, 117 products

### Preprocessing (`preprocess.py`)

1. Kept 24 relevant columns (delivery, geography, product, sales and profit fields).
2. Removed duplicates and rows with missing values.
3. Normalised accented characters and translated Spanish country names to English (e.g. *Estados Unidos* → United States).
4. Parsed order/shipping dates and derived `Order Year`, `Order Month` and `Shipping Delay` (real − scheduled shipping days).
5. Kept rows with positive sales and quantity, then sampled 20,000 rows (`random_state=42`).

To rebuild the cleaned file, download the raw CSV from Kaggle into the project folder and run `python preprocess.py`.

## How to run

```bash
git clone https://github.com/Candincer95/supply-chain-vis-dashboard.git
cd supply-chain-vis-dashboard
pip install -r requirements.txt
streamlit run app.py
```

The dashboard opens at http://localhost:8501.

## Dashboard structure

- **Sidebar filters** (apply to all chart tabs): market multiselect, sales range slider, "show only late deliveries" checkbox
- **KPI cards:** filtered orders, total sales, total profit, late delivery rate
- **Tabs:**

| Tab | Visualizations |
| --- | --- |
| Geospatial & Logistics Flow | Geographic bubble map, Sankey diagram, sunburst |
| Temporal & Product Performance | Connected scatterplot (brushing & linking), treemap, parallel coordinates |
| Sales, Shipping & Order Status | Heatmap, dumbbell chart, mosaic (Marimekko) chart |
| Machine Learning Models | Decision tree and k-nearest-neighbours classifiers for late-delivery risk |

## Project structure

```
app.py                    # main Streamlit app: data loading, filters, KPIs, tabs
preprocess.py             # data cleaning script
cleaned_supply_chain.csv  # cleaned dataset used by the app
charts/can.py             # map, Sankey, sunburst
charts/umran.py           # connected scatterplot, treemap, parallel coordinates
charts/berkay.py          # heatmap, dumbbell, mosaic chart
charts/ml_models.py       # decision tree and KNN models
requirements.txt
```

## Team contributions

| Member | Contribution |
| --- | --- |
| İbrahim Can Dinçer | Data cleaning and preprocessing, GitHub repository, dashboard layout and sidebar filters (`app.py`), map, Sankey and sunburst charts |
| Ümran Er | Connected scatterplot, treemap and parallel coordinates with linked views (`charts/umran.py`), project report |
| Berkay Doyuran | Heatmap, dumbbell and mosaic charts (`charts/berkay.py`), decision-tree and KNN models (`charts/ml_models.py`) |

## AI assistance

Gemini was used for parts of the data cleaning. Claude Opus 5.5 (Anthropic) was used on 14 November 2026 to help develop and test `charts/umran.py`.

# NvidiaEarningsDashboard
This repo consist of all the source code for the Nvidia Earning Analysis Dashboard. the code is in Python and the dashboard will be rendered on streamlit.
The Dashboard is used to give an Analysis of Earnings of NVIDIA and how it behaves during Quarterly results, how are the Factor Decomposition and how does the volatility behave.
The dashboard has extensive summary in details explanaing the numbers and the statistics.
The main module is the app.py module which calls functions created in different modules to generate the dshboard. The other modules are constants.py which has all the constant defined in the variables assigned, ui.py this file has the styling elements for the dashboard design, data.py this module has function read in the input data and handle the input data processing, analysis.py module which has the initial model code and different statistical functions that will be called in the app.py, plots.py this has the plotly helper functions which help in generation of the visuals in the dashboard. The main module is the app.py module which has the code to generate all the different tabs of the dashboard.
The Dashboard has filters on the left to select a date range, factors, Pre and Post Earning window, Volatility slider and the Return components like Total, Systematic and Idiosyncratic.
There are 2 documents **Summary of NVIDIA Returns Analysis** placed to give Summary at a high level, on in PDF Format and the other in Word format. 
# ⭐ High-Level Explanation of All 7 Tabs
**1)Overview**
Purpose :Gives a **top-level summary** of NVIDIA’s stock behavior across the chosen date range, focusing on returns, cumulative performance, and how earnings events affect price movement.
### **Key Features:**
* High-level KPIs such as:
  * total number of trading days
  * number of factors used
  * earnings dates covered
* Daily and cumulative return charts with earnings-day vertical markers
* A combined chart showing:
  * **Total NVIDIA return**
  * **Factor-based (systematic) return**
  * **Idiosyncratic return**
* Quick intro to methodology and event alignment logic.
### **Takeaway:**
This tab gives you a **bird’s-eye view** of NVIDIA’s performance and how day-0 earnings reactions compare to normal trading days.

# **2) Data & EDA (Exploratory Data Analysis)**
Purpose:Shows the **data quality**, structure, and initial exploratory insights for NVIDIA returns and factor returns.
### **Key Features:**
* Missing values
* Duplicates
* Outlier detection (z-score based)
* Summary statistics for:
  * NVIDIA total returns
  * factor-predicted returns
  * idiosyncratic returns
* Time-series charts for NVIDIA vs selected factors
* Correlation heatmap for factor returns
* Explanation boxes that describe:
  * What the data contains
  * Why factor correlations matter
  * How factor loadings and returns interact
### **Takeaway:**
Ensures the input data is clean and gives users an understanding of how factors behave relative to NVIDIA.

# **3) Earnings Event Analysis**
Purpose:Analyzes NVIDIA’s **behavior around earnings**, decomposing returns **before**, **on**, and **after** earnings.
### **Key Features:**
* A KPI grid comparing:
  * Average total return
  * Average factor return
  * Average idiosyncratic return
  * For three windows:
    * **Pre** (−N day window)
    * **Day 0** (earnings reaction day)
    * **Post** (+N day window)
* Narrative explanations automatically updated based on user selections
* Event-aligned return curves (Total, Systematic, Idiosyncratic)
* Boxplot comparing:
  * Earnings-day return distributions
  * Non-earnings-day distributions
### **Takeaway:**
Shows *how NVIDIA typically behaves around earnings announcements* and whether the reaction is systematic or company-specific.

# **4) Factor-Based Return Decomposition**
Purpose:Decomposes daily returns into:
* **Systematic (factor-based)**
* **Idiosyncratic**
And analyzes how much each contributes.
### **Key Features:**
* KPIs:
  * R² (variance explained by factors)
  * Correlation (Total vs Factor)
  * Hit rate (same sign frequency)
  * Average absolute idiosyncratic vs factor return
* Summary statistics: mean, volatility, skew, kurtosis
* Cumulative return curves around earnings for each component
* **Top 5 factors by contribution**:
  * Total contribution
  * Absolute contribution
  * Average daily contribution
  * Contribution volatility
  * Percentage impact
### **Takeaway:**
Helps understand whether NVIDIA’s movements are **market-driven** or **driven by stock-specific news**, and which factors matter the most.

# **5) Volatility Around Earnings Events**
Purpose:Quantifies how NVIDIA’s **volatility** behaves **before**, **during**, and **after** earnings.
### **Key Features:**
* Rolling volatility (annualized) for:
  * Total returns
  * Idiosyncratic returns
* Pre vs post-earnings volatility analysis:
  * Average absolute returns pre-window vs post-window
  * Per-event differences
  * Visual bar chart for change in volatility
* Narrative describing how volatility usually spikes on day-0
* Insights into calming vs heating-up events
### **Takeaway:**
Helps users understand the **risk environment** around earnings and whether NVIDIA experiences volatility shocks.

# **6) Statistical Reliability & Limitations**
Purpose:Validates the reliability of the results using proper **statistical tests**.
### **Key Features:**
* t-tests for:
  * Day-0 idiosyncratic mean ≠ 0
  * Day-0 total return mean ≠ 0
  * Post vs pre volatility difference
* p-values presented clearly
* Explanation of what t-tests mean
* List of well-stated limitations such as:
  * Few events
  * Non-normal returns
  * After-hours effects
  * Factor model sensitivity
### **Takeaway:**
Shows whether the insights are **statistically meaningful** or could be due to chance.

# **7) Summary & Key Insights**
Purpose:A **final, human-readable interpretation** of everything discovered across the analysis.
### **Key Features:**
* Complete summary of:
  1. **Performance around earnings**
  2. **Factor vs idiosyncratic behavior**
  3. **Volatility patterns**
  4. **Strategy ideas**
  5. **Statistical reliability**
* Includes medians, averages, factor shares, volatility comparisons
* Clear narratives such as:
  * “Most of the day-0 move is idiosyncratic.”
  * “Volatility often calms down after earnings.”
  * “Hedging strategies based on factor exposure.”
### **Takeaway:**
This is the **executive summary** that distills the entire dashboard into actionable insights.

# ✅ Final Summary
The 7-tab dashboard progresses **logically** from raw data → cleaning → modeling → event analysis → volatility → stats → insights.
It is essentially a **complete quant research framework** packaged into an interactive Streamlit application.

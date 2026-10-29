"""
Damodaran Cost of Capital — Streamlit App
==========================================
Deploy to Streamlit Cloud via GitHub.
Students open the URL, type a ticker, see all results instantly.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
from datetime import date

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Damodaran Cost of Capital",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# DAMODARAN CONSTANTS (2024)
# ─────────────────────────────────────────────
RATING_TABLE = [
    (8.50,  float('inf'), "Aaa/AAA", 0.0063),
    (6.50,  8.499,        "Aa2/AA",  0.0078),
    (5.50,  6.499,        "A1/A+",   0.0098),
    (4.25,  5.499,        "A2/A",    0.0108),
    (3.00,  4.249,        "A3/A-",   0.0122),
    (2.50,  2.999,        "Baa2/BBB",0.0156),
    (2.25,  2.499,        "Ba1/BB+", 0.0200),
    (2.00,  2.249,        "Ba2/BB",  0.0240),
    (1.75,  1.999,        "B1/B+",   0.0275),
    (1.50,  1.749,        "B2/B",    0.0325),
    (1.25,  1.499,        "B3/B-",   0.0425),
    (0.80,  1.249,        "Caa/CCC", 0.0575),
    (0.65,  0.799,        "Ca2/CC",  0.0825),
    (0.20,  0.649,        "C2/C",    0.1200),
    (-999,  0.199,        "D2/D",    0.1500),
]

COUNTRY_ERP = {
    "United States": 0.0472, "Canada": 0.0497, "United Kingdom": 0.0522,
    "Germany": 0.0497, "France": 0.0522, "Japan": 0.0572,
    "China": 0.0722, "India": 0.0872, "Brazil": 0.0972, "Australia": 0.0497,
}

def get_rating(cov):
    for mn, mx, rat, sp in RATING_TABLE:
        if mn <= cov <= mx:
            return rat, sp
    return "D2/D", 0.15

# ─────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────
def safe_val(df, keys, default=None):
    if df is None or df.empty: return default
    col = df.columns[0]
    for k in keys:
        if k in df.index:
            v = df.loc[k, col]
            if pd.notna(v) and v != 0: return float(v)
    return default

def safe_info(info, keys, default=None):
    for k in keys:
        v = info.get(k)
        if v is not None and v != 0 and str(v) not in ('nan','inf','-inf'):
            try: return float(v)
            except: pass
    return default

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(ticker):
    t = yf.Ticker(ticker.upper())
    info = t.info or {}
    inc = t.income_stmt
    bs  = t.balance_sheet
    cf  = t.cashflow

    price   = safe_info(info, ['currentPrice','regularMarketPrice','previousClose'])
    mktcap  = safe_info(info, ['marketCap'])
    shares  = safe_info(info, ['sharesOutstanding','impliedSharesOutstanding'])
    beta    = safe_info(info, ['beta']) or 1.0
    ev      = safe_info(info, ['enterpriseValue'])
    div_yld = safe_info(info, ['dividendYield','trailingAnnualDividendYield']) or 0
    payout  = safe_info(info, ['payoutRatio']) or 0

    revenue  = safe_val(inc, ['Total Revenue','Operating Revenue'])
    ebit     = safe_val(inc, ['EBIT','Operating Income','Total Operating Income As Reported'])
    ebitda   = safe_val(inc, ['EBITDA','Normalized EBITDA']) or safe_info(info, ['ebitda'])
    int_exp  = safe_val(inc, ['Interest Expense','Interest Expense Non Operating'])
    if int_exp: int_exp = abs(int_exp)
    pretax   = safe_val(inc, ['Pretax Income','Income Before Tax'])
    tax_prov = safe_val(inc, ['Tax Provision','Income Tax Expense'])
    net_inc  = safe_val(inc, ['Net Income','Net Income Common Stockholders',
                               'Net Income From Continuing Operation Net Minority Interest'])

    if pretax and tax_prov and pretax != 0:
        tax_rate = max(0.0, min(tax_prov/pretax, 0.50))
    else:
        tax_rate = safe_info(info, ['effectiveTaxRate']) or 0.21

    total_debt   = safe_val(bs, ['Total Debt','Long Term Debt And Capital Lease Obligation'])
    if total_debt is None:
        ltd = safe_val(bs, ['Long Term Debt']) or 0
        std = safe_val(bs, ['Current Debt','Current Debt And Capital Lease Obligation']) or 0
        total_debt = ltd + std
    equity_bv    = safe_val(bs, ['Common Stock Equity','Stockholders Equity',
                                   'Total Equity Gross Minority Interest'])
    cash         = safe_val(bs, ['Cash And Cash Equivalents',
                                   'Cash Cash Equivalents And Short Term Investments','Cash Financial'])
    total_assets = safe_val(bs, ['Total Assets'])

    if ev is None and mktcap and total_debt is not None and cash is not None:
        ev = mktcap + total_debt - cash

    capex = safe_val(cf, ['Capital Expenditure','Capital Expenditures','Purchase Of PPE'])
    if capex: capex = -abs(capex)
    da = safe_val(cf, ['Depreciation And Amortization','Depreciation Amortization Depletion',
                        'Reconciled Depreciation'])
    if da is None: da = safe_val(inc, ['Reconciled Depreciation'])

    try:
        tnx = yf.Ticker("^TNX")
        hist = tnx.history(period="5d")
        rf = hist["Close"].iloc[-1]/100 if not hist.empty else 0.0499
    except: rf = 0.0499

    country = info.get('country','United States')
    erp = COUNTRY_ERP.get(country, 0.0572)
    growth = safe_info(info, ['earningsGrowth','revenueGrowth']) or 0.05

    return {
        'ticker': ticker.upper(),
        'company': info.get('longName') or info.get('shortName') or ticker.upper(),
        'sector': info.get('sector',''),
        'industry': info.get('industry',''),
        'country': country,
        'exchange': info.get('exchange',''),
        'currency': info.get('currency','USD'),
        'price': price, 'mktcap': mktcap, 'shares': shares, 'beta': beta,
        'ev': ev, 'div_yld': div_yld, 'payout': payout,
        'revenue': revenue, 'ebit': ebit, 'ebitda': ebitda, 'int_exp': int_exp,
        'pretax': pretax, 'tax_prov': tax_prov, 'net_inc': net_inc, 'tax_rate': tax_rate,
        'total_debt': total_debt, 'equity_bv': equity_bv, 'cash': cash,
        'total_assets': total_assets, 'capex': capex, 'da': da,
        'rf': rf, 'erp': erp, 'growth': growth,
    }

# ─────────────────────────────────────────────
# CALCULATIONS
# ─────────────────────────────────────────────
def compute(d, erp_override=None, blume_weight=0.67):
    rf = d['rf']
    erp = erp_override if erp_override else d['erp']
    t = d['tax_rate']
    beta_lev = d['beta']
    mktcap = d['mktcap'] or 1
    total_debt = d['total_debt'] or 0
    ebit = d['ebit'] or 0
    int_exp = d['int_exp'] or 0

    # Coverage & rating
    coverage = ebit / int_exp if int_exp and int_exp > 0 else None
    if coverage is not None:
        syn_rating, syn_spread = get_rating(coverage)
    else:
        syn_rating, syn_spread = "N/A", 0.0156

    # Cost of Debt
    kd_pretax = rf + syn_spread
    kd_aftertax = kd_pretax * (1 - t)

    # Hamada unlevering
    de_ratio = total_debt / mktcap if mktcap > 0 else 0
    beta_unlev = beta_lev / (1 + (1 - t) * de_ratio)

    # Blume adjustment
    beta_adj = blume_weight * beta_lev + (1 - blume_weight) * 1.0
    beta_final = beta_adj

    # Cost of Equity (CAPM)
    ke = rf + beta_final * erp

    # Current WACC
    cur_dr = total_debt / (mktcap + total_debt) if (mktcap + total_debt) > 0 else 0
    cur_er = 1 - cur_dr
    cur_wacc = ke * cur_er + kd_aftertax * cur_dr

    # Optimal Capital Structure
    ebit_m = ebit / 1e6
    total_cap = (mktcap + total_debt) / 1e6
    ocs_rows = []
    for dr in [i/10 for i in range(10)]:
        er = 1 - dr
        if er <= 0: continue
        de = dr / er
        debt_est = dr * total_cap
        kd_init = rf + 0.0063
        int_est = debt_est * kd_init
        if int_est > 0 and ebit_m > 0:
            cov_est = ebit_m / int_est
            _, sp = get_rating(cov_est)
            rat, _ = get_rating(cov_est)
        else:
            sp = 0.0063; rat = "Aaa/AAA"
        kd_p = rf + sp
        kd_a = kd_p * (1 - t)
        bl = beta_unlev * (1 + (1 - t) * de)
        bl_adj = blume_weight * bl + (1 - blume_weight) * 1.0
        ke_i = rf + bl_adj * erp
        wacc_i = ke_i * er + kd_a * dr
        ocs_rows.append({
            'Debt Ratio': f"{dr:.0%}",
            'D/E': f"{de:.2f}x",
            'Rating': rat,
            'Spread': sp,
            'Pre-Tax Kd': kd_p,
            'After-Tax Kd': kd_a,
            'Re-levered β': bl_adj,
            'Cost of Equity': ke_i,
            'WACC': wacc_i,
            '_dr': dr,
            '_wacc': wacc_i,
        })

    opt = min(ocs_rows, key=lambda x: x['_wacc'])

    return {
        'coverage': coverage,
        'syn_rating': syn_rating,
        'syn_spread': syn_spread,
        'kd_pretax': kd_pretax,
        'kd_aftertax': kd_aftertax,
        'beta_unlev': beta_unlev,
        'beta_adj': beta_adj,
        'ke': ke,
        'cur_dr': cur_dr,
        'cur_wacc': cur_wacc,
        'ocs_rows': ocs_rows,
        'opt': opt,
        'rf': rf,
        'erp': erp,
        't': t,
    }

# ─────────────────────────────────────────────
# FORMATTING HELPERS
# ─────────────────────────────────────────────
def pct(v, dec=2): return f"{v:.{dec}%}" if v is not None else "N/A"
def num(v, dec=0): return f"${v/1e6:,.{dec}f}M" if v is not None else "N/A"
def val(v, dec=2): return f"{v:.{dec}f}" if v is not None else "N/A"

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3b/NYU_Stern_School_of_Business.svg/320px-NYU_Stern_School_of_Business.svg.png", width=180)
    st.title("📊 Damodaran\nCost of Capital")
    st.markdown("---")

    ticker_input = st.text_input(
        "🔍 Enter Ticker Symbol",
        value="AAPL",
        placeholder="e.g. AAPL, TSLA, MSFT",
        help="Enter any valid stock ticker from Yahoo Finance"
    ).upper().strip()

    st.markdown("---")
    st.subheader("⚙️ Assumptions")

    erp_override = st.slider(
        "Equity Risk Premium (ERP)",
        min_value=0.02, max_value=0.12, value=0.0472, step=0.001,
        format="%.2f%%",
        help="Damodaran US ERP 2024 = 4.72%"
    )

    blume_weight = st.slider(
        "Blume Weight (β adjustment)",
        min_value=0.0, max_value=1.0, value=0.67, step=0.01,
        help="0.67 = standard Blume (2/3 raw + 1/3 market)"
    )

    st.markdown("---")
    st.markdown("**📚 Methodology**")
    st.markdown("""
- **Cost of Debt**: Synthetic rating via interest coverage ratio (Damodaran 2024)
- **Cost of Equity**: CAPM with Hamada unlevering + Blume adjustment
- **Optimal Structure**: WACC minimization across 0–90% debt ratios
- **Data**: Yahoo Finance via yfinance
- **ERP**: Damodaran Online 2024
    """)
    st.markdown("---")
    st.caption(f"Data as of {date.today()}")

# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────
st.title("📊 Damodaran Cost of Capital Model")
st.markdown("*Dynamic valuation tool — type any ticker to compute Cost of Debt, Cost of Equity, and Optimal Capital Structure*")

if not ticker_input:
    st.info("👈 Enter a ticker symbol in the sidebar to get started.")
    st.stop()

# Fetch data
with st.spinner(f"Fetching live data for **{ticker_input}** from Yahoo Finance..."):
    try:
        d = fetch_data(ticker_input)
        c = compute(d, erp_override=erp_override, blume_weight=blume_weight)
    except Exception as e:
        st.error(f"❌ Could not fetch data for **{ticker_input}**. Please check the ticker symbol.")
        st.exception(e)
        st.stop()

if not d['price']:
    st.error(f"❌ No data found for **{ticker_input}**. Please verify the ticker at finance.yahoo.com")
    st.stop()

# ── Company Header ──
col1, col2, col3 = st.columns([3, 1, 1])
with col1:
    st.header(f"{d['company']} ({d['ticker']})")
    st.caption(f"{d['sector']} · {d['industry']} · {d['country']} · {d['exchange']} · {d['currency']}")
with col2:
    st.metric("Stock Price", f"${d['price']:,.2f}" if d['price'] else "N/A")
with col3:
    st.metric("Market Cap", f"${d['mktcap']/1e9:.1f}B" if d['mktcap'] else "N/A")

st.markdown("---")

# ── KEY RESULTS DASHBOARD ──
st.subheader("🎯 Key Results Summary")
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Pre-Tax Kd", pct(c['kd_pretax']), help="Cost of Debt before tax shield")
k2.metric("After-Tax Kd", pct(c['kd_aftertax']), help="Kd × (1 − Tax Rate)")
k3.metric("Cost of Equity", pct(c['ke']), help="CAPM: Rf + β × ERP")
k4.metric("Current WACC", pct(c['cur_wacc']), help="At current capital structure")
k5.metric("Optimal WACC", pct(c['opt']['_wacc']), help="Minimum achievable WACC")
k6.metric("Optimal D/V", c['opt']['Debt Ratio'], help="Debt ratio that minimizes WACC")

st.markdown("---")

# ── TABS ──
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💰 Cost of Debt",
    "📈 Cost of Equity",
    "⚖️ Optimal Structure",
    "📋 Inputs",
    "📚 Methodology"
])

# ════════════════════════════════════════════════════════════════
# TAB 1: COST OF DEBT
# ════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("💰 Cost of Debt — Damodaran Synthetic Rating Method")
    st.markdown("*EBIT ÷ Interest Expense → Coverage Ratio → Synthetic Rating → Default Spread → Kd*")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Step 1–4: Coverage → Rating → Spread")
        cod_data = {
            "Metric": [
                "EBIT ($M)",
                "Interest Expense ($M)",
                "Interest Coverage Ratio",
                "Coverage Interpretation",
                "Synthetic Credit Rating",
                "Default Spread",
            ],
            "Value": [
                num(d['ebit']),
                num(d['int_exp']),
                f"{c['coverage']:.2f}x" if c['coverage'] else "N/A",
                ("Very Strong (≥8.5x)" if c['coverage'] and c['coverage'] >= 8.5 else
                 "Strong (6.5-8.5x)" if c['coverage'] and c['coverage'] >= 6.5 else
                 "Good (5.5-6.5x)" if c['coverage'] and c['coverage'] >= 5.5 else
                 "Adequate (4.25-5.5x)" if c['coverage'] and c['coverage'] >= 4.25 else
                 "Moderate (3.0-4.25x)" if c['coverage'] and c['coverage'] >= 3.0 else
                 "Below Average (2.5-3.0x)" if c['coverage'] and c['coverage'] >= 2.5 else
                 "Weak (2.0-2.5x)" if c['coverage'] and c['coverage'] >= 2.0 else
                 "Distressed (<2.0x)" if c['coverage'] else "N/A"),
                c['syn_rating'],
                pct(c['syn_spread']),
            ]
        }
        st.dataframe(pd.DataFrame(cod_data), hide_index=True, use_container_width=True)

    with col_b:
        st.markdown("#### Step 5: Cost of Debt Calculation")
        cod_calc = {
            "Component": [
                "Risk-Free Rate (Rf)",
                "Default Spread",
                "Pre-Tax Cost of Debt [Kd]",
                "Tax Rate",
                "After-Tax Cost of Debt [Kd×(1-t)]",
            ],
            "Formula": [
                "10Y US Treasury",
                "Rf + Spread",
                "Rf + Default Spread",
                "Tax Provision / Pre-Tax Income",
                "Kd × (1 − t)",
            ],
            "Value": [
                pct(c['rf']),
                pct(c['syn_spread']),
                pct(c['kd_pretax']),
                pct(c['t']),
                pct(c['kd_aftertax']),
            ]
        }
        df_cod = pd.DataFrame(cod_calc)
        st.dataframe(df_cod, hide_index=True, use_container_width=True)

        st.success(f"**After-Tax Cost of Debt = {pct(c['kd_aftertax'])}**")

    st.markdown("---")
    st.markdown("#### Damodaran Rating Table (2024)")
    rating_df = pd.DataFrame([
        {"Coverage Min": mn if mn > -900 else "< 0",
         "Coverage Max": mx if mx < 1e8 else "∞",
         "Synthetic Rating": rat,
         "Default Spread": f"{sp:.2%}",
         "Pre-Tax Kd (Rf=5%)": f"{0.05+sp:.2%}"}
        for mn, mx, rat, sp in RATING_TABLE
    ])
    # Highlight current rating
    def highlight_rating(row):
        if row['Synthetic Rating'] == c['syn_rating']:
            return ['background-color: #C6EFCE; color: #276221; font-weight: bold'] * len(row)
        return [''] * len(row)
    st.dataframe(rating_df.style.apply(highlight_rating, axis=1),
                 hide_index=True, use_container_width=True)
    st.caption(f"🟢 Green row = {d['ticker']}'s current synthetic rating ({c['syn_rating']})")

# ════════════════════════════════════════════════════════════════
# TAB 2: COST OF EQUITY
# ════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📈 Cost of Equity — CAPM + Hamada Equation")
    st.markdown("*Ke = Rf + β_levered × ERP  |  β via Hamada unlevering + Blume adjustment*")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Step 1–2: Hamada Equation (Unlever Beta)")
        st.latex(r"\beta_{unlevered} = \frac{\beta_{levered}}{1 + (1-t) \times \frac{D}{E}}")
        hamada_data = {
            "Variable": ["Levered Beta (Raw)", "D/E Ratio (Market)", "Tax Rate", "Unlevered Beta [βu]"],
            "Value": [
                f"{d['beta']:.3f}",
                f"{(d['total_debt'] or 0)/(d['mktcap'] or 1):.3f}x",
                pct(c['t']),
                f"{c['beta_unlev']:.3f}",
            ],
            "Note": [
                "5-year monthly vs S&P 500",
                "Book Debt / Market Cap",
                "Effective tax rate",
                "Strips out financial risk",
            ]
        }
        st.dataframe(pd.DataFrame(hamada_data), hide_index=True, use_container_width=True)

        st.markdown("#### Step 3: Blume Adjustment")
        st.latex(r"\beta_{adjusted} = 0.67 \times \beta_{raw} + 0.33 \times 1.0")
        blume_data = {
            "Variable": ["Raw Beta", "Blume Weight", "Adjusted Beta (used in CAPM)"],
            "Value": [f"{d['beta']:.3f}", f"{blume_weight:.2f}", f"{c['beta_adj']:.3f}"],
        }
        st.dataframe(pd.DataFrame(blume_data), hide_index=True, use_container_width=True)

    with col_b:
        st.markdown("#### Step 4: CAPM Cost of Equity")
        st.latex(r"K_e = R_f + \beta_{levered} \times ERP")
        capm_data = {
            "Component": ["Risk-Free Rate (Rf)", "Beta (Blume-adjusted)", "Equity Risk Premium (ERP)",
                          "Beta × ERP", "Cost of Equity [Ke]"],
            "Value": [pct(c['rf']), f"{c['beta_adj']:.3f}", pct(c['erp']),
                      pct(c['beta_adj'] * c['erp']), pct(c['ke'])],
        }
        st.dataframe(pd.DataFrame(capm_data), hide_index=True, use_container_width=True)
        st.success(f"**Cost of Equity (CAPM) = {pct(c['ke'])}**")

        st.markdown("#### Alternative Estimates")
        ddm_ke = (d['div_yld'] or 0) + (d['growth'] or 0)
        ep_val = None
        if d['net_inc'] and d['shares'] and d['price'] and d['shares'] > 0 and d['price'] > 0:
            ep_val = (d['net_inc'] / d['shares']) / d['price']
        alt_data = {
            "Method": ["DDM (Div Yield + g)", "Earnings Yield (E/P)", "Average CAPM & DDM"],
            "Value": [pct(ddm_ke), pct(ep_val), pct((c['ke'] + ddm_ke)/2)],
            "Formula": ["Div Yield + Growth", "EPS / Price", "(CAPM + DDM) / 2"],
        }
        st.dataframe(pd.DataFrame(alt_data), hide_index=True, use_container_width=True)

# ════════════════════════════════════════════════════════════════
# TAB 3: OPTIMAL CAPITAL STRUCTURE
# ════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("⚖️ Optimal Capital Structure — WACC Minimization")
    st.markdown("*Vary D/(D+E) from 0% to 90% → Re-lever beta → Find debt ratio that minimizes WACC*")

    # Summary metrics
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Optimal Debt Ratio", c['opt']['Debt Ratio'], help="Minimizes WACC")
    s2.metric("Optimal WACC", pct(c['opt']['_wacc']), help="Minimum cost of capital")
    s3.metric("Current Debt Ratio", pct(c['cur_dr']), help="Actual D/(D+E)")
    s4.metric("Current WACC", pct(c['cur_wacc']),
              delta=f"{(c['cur_wacc']-c['opt']['_wacc']):.2%} above optimal",
              delta_color="inverse")

    # OCS Table
    ocs_df = pd.DataFrame(c['ocs_rows'])
    display_df = ocs_df[['Debt Ratio','D/E','Rating','Pre-Tax Kd','After-Tax Kd',
                          'Re-levered β','Cost of Equity','WACC']].copy()
    display_df['Pre-Tax Kd']    = display_df['Pre-Tax Kd'].map(lambda x: f"{x:.2%}")
    display_df['After-Tax Kd']  = display_df['After-Tax Kd'].map(lambda x: f"{x:.2%}")
    display_df['Re-levered β']  = display_df['Re-levered β'].map(lambda x: f"{x:.3f}")
    display_df['Cost of Equity']= display_df['Cost of Equity'].map(lambda x: f"{x:.2%}")
    display_df['WACC']          = display_df['WACC'].map(lambda x: f"{x:.2%}")

    opt_dr = c['opt']['_dr']
    cur_dr = c['cur_dr']

    def style_ocs(row):
        dr_str = row['Debt Ratio'].replace('%','')
        try: dr_val = float(dr_str)/100
        except: return [''] * len(row)
        if abs(dr_val - opt_dr) < 0.001:
            return ['background-color: #C6EFCE; color: #276221; font-weight: bold'] * len(row)
        if abs(dr_val - cur_dr) < 0.05:
            return ['background-color: #FFEB9C; color: #0000FF'] * len(row)
        return [''] * len(row)

    st.dataframe(display_df.style.apply(style_ocs, axis=1),
                 hide_index=True, use_container_width=True)
    st.caption("🟢 Green = Optimal (minimizes WACC)  |  🟡 Yellow = Current structure")

    # WACC Chart
    st.markdown("#### WACC vs Debt Ratio")
    chart_data = pd.DataFrame({
        'Debt Ratio': [r['_dr'] for r in c['ocs_rows']],
        'WACC': [r['_wacc'] for r in c['ocs_rows']],
        'Cost of Equity': [r['Cost of Equity'] for r in c['ocs_rows']],
        'After-Tax Kd': [r['After-Tax Kd'] for r in c['ocs_rows']],
    }).set_index('Debt Ratio')
    st.line_chart(chart_data, use_container_width=True)

# ════════════════════════════════════════════════════════════════
# TAB 4: INPUTS
# ════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("📋 Raw Financial Data")
    st.caption(f"Source: Yahoo Finance via yfinance  |  Data as of {date.today()}")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Market Data")
        mkt = {
            "Field": ["Stock Price","Market Cap","Shares Outstanding","Levered Beta",
                      "Enterprise Value","Dividend Yield","Payout Ratio"],
            "Value": [
                f"${d['price']:,.2f}" if d['price'] else "N/A",
                num(d['mktcap']),
                f"{d['mktcap']/1e6/d['price']:.1f}M shares" if d['mktcap'] and d['price'] else "N/A",
                f"{d['beta']:.3f}",
                num(d['ev']),
                pct(d['div_yld']),
                pct(d['payout']),
            ]
        }
        st.dataframe(pd.DataFrame(mkt), hide_index=True, use_container_width=True)

        st.markdown("#### Balance Sheet ($M)")
        bs_data = {
            "Field": ["Book Value of Debt","Book Value of Equity","Cash & Equivalents","Total Assets"],
            "Value": [num(d['total_debt']), num(d['equity_bv']), num(d['cash']), num(d['total_assets'])]
        }
        st.dataframe(pd.DataFrame(bs_data), hide_index=True, use_container_width=True)

    with col_b:
        st.markdown("#### Income Statement ($M)")
        inc_data = {
            "Field": ["Revenue","EBIT","EBITDA","Interest Expense","Pre-Tax Income",
                      "Tax Provision","Net Income","Effective Tax Rate"],
            "Value": [num(d['revenue']), num(d['ebit']), num(d['ebitda']), num(d['int_exp']),
                      num(d['pretax']), num(d['tax_prov']), num(d['net_inc']), pct(d['tax_rate'])]
        }
        st.dataframe(pd.DataFrame(inc_data), hide_index=True, use_container_width=True)

        st.markdown("#### Macro Assumptions")
        macro_data = {
            "Field": ["Risk-Free Rate (10Y UST)","Equity Risk Premium (ERP)","Growth Estimate"],
            "Value": [pct(d['rf']), pct(d['erp']), pct(d['growth'])],
            "Source": ["Yahoo Finance (^TNX)","Damodaran Online 2024","Yahoo Finance consensus"]
        }
        st.dataframe(pd.DataFrame(macro_data), hide_index=True, use_container_width=True)

# ════════════════════════════════════════════════════════════════
# TAB 5: METHODOLOGY
# ════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("📚 Damodaran Methodology")

    with st.expander("📌 Cost of Debt — Synthetic Rating Method", expanded=True):
        st.markdown("""
**Step 1:** Compute Interest Coverage Ratio
$$\\text{Coverage} = \\frac{\\text{EBIT}}{\\text{Interest Expense}}$$

**Step 2:** Map coverage to synthetic rating using Damodaran's table (2024)

**Step 3:** Look up default spread for that rating

**Step 4:** Pre-Tax Cost of Debt
$$K_d = R_f + \\text{Default Spread}$$

**Step 5:** After-Tax Cost of Debt (interest tax shield)
$$K_d^{\\text{after-tax}} = K_d \\times (1 - t)$$

*Source: Damodaran, A. (2024). "Ratings, Interest Coverage Ratios and Default Spreads." Damodaran Online.*
        """)

    with st.expander("📌 Cost of Equity — CAPM + Hamada Equation"):
        st.markdown("""
**Step 1:** Unlever the market beta using Hamada Equation
$$\\beta_u = \\frac{\\beta_L}{1 + (1-t) \\times \\frac{D}{E}}$$

**Step 2:** Blume Adjustment (mean-reversion toward 1.0)
$$\\beta_{adj} = 0.67 \\times \\beta_{raw} + 0.33 \\times 1.0$$

**Step 3:** CAPM Cost of Equity
$$K_e = R_f + \\beta_{levered} \\times ERP$$

*Sources: Hamada (1972) Journal of Finance; Blume (1975) Journal of Finance; Damodaran (2024) ERP.*
        """)

    with st.expander("📌 Optimal Capital Structure — WACC Minimization"):
        st.markdown("""
**Step 1:** For each debt ratio D/V (0% to 90%):
- Estimate interest expense at that leverage
- Compute coverage ratio → synthetic rating → spread → Kd
- Re-lever beta: $\\beta_L = \\beta_u \\times [1 + (1-t) \\times D/E]$
- Compute Ke via CAPM
- Compute WACC: $WACC = K_e \\times \\frac{E}{V} + K_d(1-t) \\times \\frac{D}{V}$

**Step 2:** Find the debt ratio that **minimizes WACC**

**Why it works:** More debt → lower Kd (tax shield) but higher Ke (financial risk) and higher Kd (distress risk). The optimal point balances these forces.

*Source: Damodaran, A. (2024). Applied Corporate Finance. Wiley.*
        """)

    with st.expander("📌 Data Sources"):
        st.markdown("""
| Data | Source |
|------|--------|
| Financial statements | Yahoo Finance via yfinance Python library |
| Stock price, beta, market cap | Yahoo Finance |
| Risk-free rate | 10-Year US Treasury (^TNX) — live |
| Equity Risk Premium | Damodaran Online 2024 — US base ERP = 4.72% |
| Rating spreads | Damodaran Online 2024 |
| Beta | Yahoo Finance 5-year monthly regression vs S&P 500 |
        """)

# ── Footer ──
st.markdown("---")
st.caption(
    "📊 Damodaran Cost of Capital Model | "
    "Data: Yahoo Finance | "
    "Methodology: Damodaran Online (2024) | "
    f"Updated: {date.today()}"
)
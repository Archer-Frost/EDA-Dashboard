import streamlit as st
import pandas as pd
import re

st.title("Of Tar and Taxes : Tobacco Prices and Inflation in India")
# --------------------------------------------------
# Data loading
# --------------------------------------------------
@st.cache_data
def load_all_india():
    m = pd.read_csv("data/all_india_m.csv")
    q = pd.read_csv("data/all_india_q.csv")
    return m, q

all_india_m, all_india_q = load_all_india()

# Create month-start date column (date only, no time)
all_india_m["date"] = pd.to_datetime(
    dict(year=all_india_m["year"], month=all_india_m["month"],day=1)
)

def load_village_data():
    alrl = pd.read_csv("data/cpi_al-rl_vDec2018_1.csv")
    iw = pd.read_csv("data/cpi_iw_Dec2018_1.csv")
    
    # Clean column names
    alrl.columns = alrl.columns.str.strip()
    iw.columns = iw.columns.str.strip()
    
    return alrl, iw

alrl, iw = load_village_data()



# --------------------------------------------------
# Tabs
# --------------------------------------------------
tab0, tab1, tab2, tab3 = st.tabs([
    "Introduction",
    "National Price Trends",
    "Statewise Comparison",
    "Interpretation"
])

#TAB 0================================================================================================================================
with tab0:
    st.subheader("A brief synopsis")
    st.markdown("""
    Tobacco products such as bidis and cigarettes hold a complex position in Indian society. They are
not only commodities of consumption but also central to debates around public health, taxation,
and affordability. The government has attempted to reduce consumption through fiscal policy,
most notably excise taxes and, later, the implementation of the Goods and Services Tax (GST)
in 2017. Understanding how tobacco prices evolve over time and how they compare with general
cost-of-living indices is essential for assessing the effectiveness of these measures.

This report examines the time evolution of tobacco price indices in India using Consumer Price
Index (CPI) data for both urban (Industrial Workers, CPI-IW) and rural (Agricultural and Rural Labourers, CPI-AL/RL) contexts. We supplement these indices with micro-level retail price
quotes that feed into the CPI construction, thereby grounding the aggregate trends in the observed
heterogeneity of retail markets across states.
The analysis is descriptive and exploratory in nature. Our primary questions are:

- How have tobacco product prices (especially bidis and cigarettes) evolved relative to the base
       year 2000?
- Has the affordability of tobacco decreased relative to general consumption baskets?
- Are there differences between urban and rural populations, and across regions?
- Do key policy interventions (e.g., GST) align with visible structural shifts in these trends?
    
We pursue these questions by conducting exploratory data analysis (EDA), presenting univariate,
bivariate, and multivariate visuals, and then consolidating the results into policy-relevant conclusions
""")

#TAB 1================================================================================================================================
with tab1:
    st.markdown("""
   The table below presents the monthly Consumer Price Index (CPI) and Wholesale Price Index (WPI) series for tobacco products and the corresponding all-items index for the two population groups under consideration: Industrial Workers (IW) and Agricultural/Rural Labourers (AL/RL). All indices are rebased to 2000 = 100, ensuring comparability across time and product categories.

The dataset includes item-level indices for bidis and cigarettes, along with the aggregate all-items index for each population group. These indices capture the evolution of consumer-facing price movements and allow for both product-level and basket-level inflation comparisons.
""")
    st.dataframe(
        all_india_m.assign(date=all_india_m["date"].dt.strftime("%Y-%m"))
    )
    st.markdown("""
    From the table, we observe the monthly progression of item-level CPI indices for bidis and cigarettes under both CPI-IW and CPI-AL/RL frameworks. The national index series (_n_00) reflects the overall price movement at the aggregate level, while the retail index series (_r_00) captures price behavior derived directly from retail price quotes used in CPI construction.

The parallel presentation of these series enables direct comparison across:

- Population groups (Industrial Workers vs Agricultural/Rural Labourers)

- Product categories (Bidi vs Cigarette vs All-items)

- Index construction type (National vs Retail)

This structure allows us to assess whether tobacco prices have risen faster than the general consumption basket and whether price dynamics differ across population segments. Subsequent visualizations will illustrate these trends more clearly over the selected time range.
""")

    st.subheader("CPI Analysis")
    container = st.container()

    with container:
        controls_col,plot_col = st.columns([1,3],gap="small")

        with controls_col:
            all = st.checkbox("Select all",key="pop select all")
            if all:
                selected_options1 = st.multiselect("Population Types:",
                                                         ['Industrial Workers','Agricultural/Rural Labourers'],['Industrial Workers','Agricultural/Rural Labourers'])
            else:
                selected_options1 =  st.multiselect("Population Types:",
                                                          ['Industrial Workers','Agricultural/Rural Labourers'],
                                                   default = ["Industrial Workers"])

            all = st.checkbox("Select all",key="prod select all")
            if all:
                selected_options2 = st.multiselect("Product Types:",
                                                         ['Bidi','Cigarette','All items'],['Bidi','Cigarette','All items'])
            else:
                selected_options2 =  st.multiselect("Product Types:",
                                                          ['Bidi','Cigarette','All items'],
                                                   default = ["Bidi"])

            all = st.checkbox("Select all",key="type select all")
            if all:
                selected_options3 = st.multiselect("Index Series Type",
                                                         ['National','Retail'],['National','Retail'])
            else:
                selected_options3 =  st.multiselect("Index Series Type",
                                                          ['National','Retail'],
                                                   default = ["National"])

            min_date = all_india_m["date"].min().to_pydatetime()
            max_date = all_india_m["date"].max().to_pydatetime()
            
            start_date,end_date = st.slider(
                "Time range",
                min_value=min_date,
                max_value=max_date,
                value=(min_date, max_date),
                format="YYYY-MM"
            )

        with plot_col:
            df_f = all_india_m[(all_india_m["date"] >= start_date) & (all_india_m["date"] <= end_date)].copy()
            
            # ---------- Normalize columns (CRITICAL) ----------
            # Keep original -> normalized mapping so selection works even if CSV has weird spaces/case
            orig_cols = list(df_f.columns)
            norm_cols = [c.strip().lower() for c in orig_cols]
            rename_to_norm = dict(zip(orig_cols, norm_cols))
            rename_to_orig = dict(zip(norm_cols, orig_cols))
            
            df_f = df_f.rename(columns=rename_to_norm)
            
            # ignore WPI
            usable_cols = [c for c in df_f.columns if "wpi" not in c]
            
            # ---------- Widget selections (your variables) ----------
            selected_pops = selected_options1          # ["Industrial Workers", "Agricultural/Rural Labourers"]
            selected_prods = selected_options2         # ["Bidi", "Cigarette", "All items"]
            selected_types = selected_options3         # ["National", "Retail"]
            
            pop_prefix = {
                "Industrial Workers": "iw",
                "Agricultural/Rural Labourers": "alrl",
            }
            
            prod_tokens = {
                "Bidi": ["bidi"],
                "Cigarette": ["cig", "cigarette"],
                "All items": ["all-items", "all_items", "allitems", "all"],
            }
            
            type_tokens = {
                "National": "n",
                "Retail": "r",
            }
            
            def find_col(prefix: str, prod_alts: list[str], nr: str) -> str | None:
                """
                Match columns like:
                  alrl_bidi_n_00, alrl_cig_r_00, alrl_all-items_n_00, iw_bidi_r_00, etc.
                We require:
                  - starts with prefix + "_"
                  - contains any product token
                  - contains "_" + nr + "_" OR ends with "_" + nr (rare)
                """
                candidates = []
                for c in usable_cols:
                    if not c.startswith(prefix + "_"):
                        continue
                    if not any(tok in c for tok in prod_alts):
                        continue
                    if f"_{nr}_" in c or c.endswith(f"_{nr}"):
                        candidates.append(c)
            
                if not candidates:
                    return None
            
                # Prefer ones that look like the rebased series (often ending with _00)
                candidates.sort(key=lambda x: ("_00" not in x, len(x)))
                return candidates[0]
            
            # ---------- Build series list ----------
            series_norm_cols = []
            series_labels = []
            
            for pop in selected_pops:
                prefix = pop_prefix.get(pop)
                if not prefix:
                    continue
            
                for prod in selected_prods:
                    prod_alts = prod_tokens.get(prod, [])
                    if not prod_alts:
                        continue
            
                    for t in selected_types:
                        nr = type_tokens.get(t)
                        if not nr:
                            continue
            
                        col = find_col(prefix, prod_alts, nr)
                        if col:
                            series_norm_cols.append(col)
                            series_labels.append(f"{pop} | {prod} | {t}")
            
            # ---------- Plot ----------
            if not series_norm_cols:
                st.warning("No matching columns found. Your selection may not exist in this CSV (e.g., IW may only have Retail).")
                # Optional: show available columns to debug quickly
                # st.write(sorted([c for c in usable_cols if c.startswith("iw_") or c.startswith("alrl_")])[:80])
            else:
                plot_df = df_f[["date"] + series_norm_cols].copy()
                plot_df = plot_df.rename(columns=dict(zip(series_norm_cols, series_labels)))
                plot_df = plot_df.set_index("date")
                st.line_chart(plot_df)

    st.subheader("WPI Analysis")
    container1 = st.container()

    with container1:
        controls_col,plot_col = st.columns([1,3],gap="small")

        with controls_col:
            product_options = ['Bidi', 'Cigarette', 'All items']
            default_products = ['Bidi']
            
            # Initialize session state once
            if "wpi_prods" not in st.session_state:
                st.session_state.wpi_prods = default_products
            
            if "prod_select_all" not in st.session_state:
                st.session_state.prod_select_all = False
            
            def toggle_products():
                if st.session_state.prod_select_all:
                    # Select everything
                    st.session_state.wpi_prods = product_options
                else:
                    # Revert to default
                    st.session_state.wpi_prods = default_products
            
            st.checkbox(
                "Select all",
                key="prod_select_all",
                on_change=toggle_products
            )
            
            selected_options_wpi0 = st.multiselect(
                "Product Types:",
                product_options,
                key="wpi_prods"
            )

            index_options = ["National", "Retail"]
            default_index = ["National"]
            
            # Initialize once
            if "wpi_index_series" not in st.session_state:
                st.session_state.wpi_index_series = default_index
            
            if "type_select_all_wpi" not in st.session_state:
                st.session_state.type_select_all_wpi = False
            
            def toggle_index_series():
                if st.session_state.type_select_all_wpi:
                    st.session_state.wpi_index_series = index_options
                else:
                    st.session_state.wpi_index_series = default_index
            
            st.checkbox(
                "Select all",
                key="type_select_all_wpi",
                on_change=toggle_index_series
            )
            
            selected_options_wpi1 = st.multiselect(
                "Index Series Type",
                index_options,
                key="wpi_index_series",
            )

            min_date = all_india_m["date"].min().to_pydatetime()
            max_date = all_india_m["date"].max().to_pydatetime()
            
            start_date,end_date = st.slider(
                "Time range",
                min_value=min_date,
                max_value=max_date,
                value=(min_date, max_date),
                format="YYYY-MM",
                key = "slider wpi"
            )

        with plot_col:
            # -----------------------------
            # Filter by time (WPI slider)
            # -----------------------------
            df_wpi = all_india_m[
                (all_india_m["date"] >= start_date) &
                (all_india_m["date"] <= end_date)
            ].copy()
        
            # -----------------------------
            # Map widget selections -> column names
            # -----------------------------
            wpi_product_token = {
                "Bidi": "bidi",
                "Cigarette": "cig",
                "All items": "all-items",
            }
        
            wpi_series_token = {
                "National": "_n_",
                "Retail": "_r_",
            }
        
            series_cols = []
            series_labels = []
        
            for prod in selected_options_wpi0:
                p_tok = wpi_product_token.get(prod)
                if not p_tok:
                    continue
        
                for s in selected_options_wpi1:
                    sr_tok = wpi_series_token.get(s)
                    if not sr_tok:
                        continue
        
                    # find matching WPI column
                    for col in df_wpi.columns:
                        if (
                            col.lower().startswith("wpi_")
                            and p_tok in col.lower()
                            and sr_tok in col.lower()
                        ):
                            series_cols.append(col)
                            series_labels.append(f"WPI | {prod} | {s}")
        
            # -----------------------------
            # Plot
            # -----------------------------
            if not series_cols:
                st.warning("No matching WPI series found for your selections.")
                # Optional debug:
                # st.write([c for c in df_wpi.columns if c.lower().startswith("wpi_")])
            else:
                plot_df = df_wpi[["date"] + series_cols].copy()
                plot_df = plot_df.rename(columns=dict(zip(series_cols, series_labels)))
                plot_df = plot_df.set_index("date")
        
                st.line_chart(plot_df)

#TAB 2================================================================================================================================
with tab2:
    st.subheader("About state coverage and missing values")
    st.markdown("""
The state-wise analysis presented here is based on retail price quotes collected under CPI-AL/RL (village-level) and CPI-IW (centre-level) datasets. It is important to note the following:

- Incomplete State Coverage:
  Not all states report price quotations for every product, unit, and month. As a result, the number of states shown in the ranking    table may vary across selections.

- Unit-Specific Reporting:
Certain products are quoted only in specific units (e.g., bidis typically in bundles of 25, cigarettes often in packs of 10 or 20). When a particular unit is selected, only states reporting that exact unit are included.

- Overlap Requirement Between CPI Series:
For comparability, the dashboard displays only those months where both CPI-AL/RL and CPI-IW report data for the selected product and unit. This may reduce the number of available states in some periods.

- Missing or Invalid Quotes:
Observations with missing or non-numeric price values are excluded from aggregation. Therefore, state counts reflect valid price entries only.

Consequently, the “Top N States” slider represents an upper bound. If fewer states are available for the selected filters, the table will display only the states for which valid data exists.
""")
    st.subheader("State-wise Comparison (Retail Price Quotes: CPI-AL/RL villages vs CPI-IW centres)")

    # ---- Prep (robust cleaning) ----
    def _prep_micro(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = df.columns.str.strip()

        # Normalize price column
        price_like = [c for c in df.columns if c.strip().lower() == "price"]
        if price_like and price_like[0] != "price":
            df = df.rename(columns={price_like[0]: "price"})

        # Clean fields
        if "state" in df.columns:
            df["state"] = df["state"].astype(str).str.strip()
        if "item" in df.columns:
            df["item"] = df["item"].astype(str).str.strip().str.lower()

        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df["unit"] = pd.to_numeric(df["unit"], errors="coerce")
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df["month"] = pd.to_numeric(df["month"], errors="coerce")

        df["date"] = pd.to_datetime(
            dict(year=df["year"], month=df["month"], day=1),
            errors="coerce"
        )

        return df

    alrl_p = _prep_micro(alrl)
    iw_p   = _prep_micro(iw)

    # ---- Common items ----
    items_common = sorted(
        list(
            set(alrl_p["item"].dropna().unique()).intersection(
                set(iw_p["item"].dropna().unique())
            )
        )
    )

    if not items_common:
        st.error("No common tobacco items found between AL/RL and IW.")
        st.stop()

    controls_col, viz_col = st.columns([1, 3], gap="small")

    with controls_col:
        item_sel = st.selectbox("Product (item)", items_common)

        # Item-specific common units
        alrl_units = sorted(alrl_p.loc[alrl_p["item"] == item_sel, "unit"].dropna().unique())
        iw_units   = sorted(iw_p.loc[iw_p["item"] == item_sel, "unit"].dropna().unique())
        common_units = sorted(list(set(alrl_units).intersection(set(iw_units))))

        if not common_units:
            st.warning("No common units for this item across AL/RL and IW.")
            st.stop()

        default_unit = 25.0 if 25.0 in common_units else common_units[0]
        unit_sel = st.selectbox("Unit", common_units, index=common_units.index(default_unit))

        agg_sel = st.selectbox("Aggregation across quotes", ["Median", "Mean"])
        top_n = st.slider("Show top N States", 5, 30, 15)

    # ---- Filter ----
    def _filter(df):
        return df[
            (df["item"] == item_sel) &
            (df["unit"] == float(unit_sel))
        ].dropna(subset=["state", "date", "price"])

    alrl_f = _filter(alrl_p)
    iw_f   = _filter(iw_p)

    # ---- Common months ----
    months_alrl = set(alrl_f["date"].unique())
    months_iw   = set(iw_f["date"].unique())
    common_months = sorted(list(months_alrl.intersection(months_iw)))

    if not common_months:
        st.warning("No overlapping months for this item + unit.")
        st.stop()

    with controls_col:
        month_labels = [d.strftime("%Y-%m") for d in common_months]

        # Map label back to datetime
        month_map = dict(zip(month_labels, common_months))
        
        snap_label = st.selectbox(
            "Snapshot month",
            options=month_labels,
            index=len(month_labels) - 1
        )

        snap_date = month_map[snap_label]

    # ---- Snapshot aggregation ----
    def _agg_snapshot(df, label):
        d = df[df["date"] == pd.to_datetime(snap_date)]
        if d.empty:
            return pd.DataFrame(columns=["state", label])

        if agg_sel == "Median":
            g = d.groupby("state", as_index=False)["price"].median()
        else:
            g = d.groupby("state", as_index=False)["price"].mean()

        return g.rename(columns={"price": label})

    alrl_state = _agg_snapshot(alrl_f, "ALRL_price")
    iw_state   = _agg_snapshot(iw_f,   "IW_price")

    merged = pd.merge(alrl_state, iw_state, on="state", how="outer")
    merged["Gap_IW_minus_ALRL"] = merged["IW_price"] - merged["ALRL_price"]

    rank_basis = "IW_price" if merged["IW_price"].notna().sum() else "ALRL_price"
    merged_ranked = merged.sort_values(rank_basis, ascending=False).head(top_n)

    with viz_col:
        st.markdown(
            f"**Snapshot:** {pd.to_datetime(snap_date).strftime('%Y-%m')} | "
            f"**Item:** {item_sel} | **Unit:** {int(unit_sel)} | **Agg:** {agg_sel}"
        )

        # Level comparison
        level_plot = merged_ranked.set_index("state")[["ALRL_price", "IW_price"]]
        if not level_plot.dropna(how="all").empty:
            st.bar_chart(level_plot)

        # Gap chart
        gap_df = merged.dropna(subset=["Gap_IW_minus_ALRL"])
        if not gap_df.empty:
            gap_df["abs_gap"] = gap_df["Gap_IW_minus_ALRL"].abs()
            gap_top = gap_df.sort_values("abs_gap", ascending=False).head(top_n)
            st.caption("Gap = IW − AL/RL")
            st.bar_chart(gap_top.set_index("state")[["Gap_IW_minus_ALRL"]])

        # Table
        st.subheader("State details table (snapshot)")
        st.dataframe(
            merged.sort_values(rank_basis, ascending=False).reset_index(drop=True),
            use_container_width=True
        )

        # -------------------------
        # Time Trends (always shown)
        # -------------------------
        st.subheader("Time trends")

        min_date = min(common_months)
        max_date = max(common_months)

        start_date, end_date = st.slider(
            "Time range",
            min_value=min_date.to_pydatetime(),
            max_value=max_date.to_pydatetime(),
            value=(min_date.to_pydatetime(), max_date.to_pydatetime()),
            format="YYYY-MM",
            key="statewise_timerange"
        )

        all_states = sorted(list(set(alrl_f["state"]).union(set(iw_f["state"]))))
        default_states = merged_ranked["state"].dropna().tolist()
        if not default_states:
            default_states = all_states[:min(5, len(all_states))]

        states_sel = st.multiselect(
            "Select states",
            all_states,
            default=default_states,
            key="statewise_states_sel"
        )

        def _agg_ts(df, label):
            d = df[
                (df["date"] >= pd.to_datetime(start_date)) &
                (df["date"] <= pd.to_datetime(end_date)) &
                (df["state"].isin(states_sel))
            ]
            if d.empty:
                return pd.DataFrame()

            if agg_sel == "Median":
                g = d.groupby(["date", "state"], as_index=False)["price"].median()
            else:
                g = d.groupby(["date", "state"], as_index=False)["price"].mean()

            return g.rename(columns={"price": label})

        ts_alrl = _agg_ts(alrl_f, "ALRL_price")
        ts_iw   = _agg_ts(iw_f,   "IW_price")

        def normalize_keys(df):
    # Normalize column names
            df = df.copy()
            df.columns = df.columns.astype(str).str.strip().str.lower()
        
            # If keys are in the index, bring them back as columns
            if "date" not in df.columns or "state" not in df.columns:
                df = df.reset_index()
        
                # Re-normalize after reset_index (new columns may appear)
                df.columns = df.columns.astype(str).str.strip().str.lower()
        
            return df

        ts_alrl = normalize_keys(ts_alrl)
        ts_iw   = normalize_keys(ts_iw)
        
        # Hard fail with clear diagnostics (so you know which one is missing what)
        missing_alrl = {"date", "state"} - set(ts_alrl.columns)
        missing_iw   = {"date", "state"} - set(ts_iw.columns)

        if missing_alrl or missing_iw:
            import streamlit as st
            st.error(f"Missing keys. ts_alrl missing: {missing_alrl}, ts_iw missing: {missing_iw}")
            st.write("ts_alrl columns:", list(ts_alrl.columns))
            st.write("ts_iw columns:", list(ts_iw.columns))
            st.stop()

# Optional: ensure date types match
        ts_alrl["date"] = pd.to_datetime(ts_alrl["date"], errors="coerce")
        ts_iw["date"]   = pd.to_datetime(ts_iw["date"], errors="coerce")
        
        ts = pd.merge(ts_alrl, ts_iw, on=["date", "state"], how="outer")

        ts = pd.merge(ts_alrl, ts_iw, on=["date", "state"], how="outer")

        if ts.empty:
            st.warning("No time-series data available for selected filters.")
        else:
            st.caption("AL/RL (villages)")
            st.line_chart(ts.pivot(index="date", columns="state", values="ALRL_price"))

            st.caption("IW (centres)")
            st.line_chart(ts.pivot(index="date", columns="state", values="IW_price"))
    
            

            

            
    
    
        



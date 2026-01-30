import pandas as pd
import plotly.express as px
import json
import numpy as np
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output, dash_table, no_update, ctx, State

dataset = pd.read_csv("world_economics.csv")
dataset.drop(columns=['borders'], inplace=True)
dataset['GDP_log'] = np.log10(dataset['GDP'])

world_map = json.load(open('world.geojson'))

countries = []

for feature in world_map["features"]:
    countries.append(feature["properties"]["NAME"])

countries = list(set(countries))
countries.sort()

country_df = pd.DataFrame(countries, columns =['Country'])

metrics = {
    "GDP (in billions USD)": "GDP",
    "Interest Rate (in %)": "Interest Rate",
    "Inflation Rate (in %)": "Inflation Rate",
    "Jobless Rate (in %)": "Jobless Rate",
    "Debt/GDP (in %)": "Debt/GDP",
    "Population (in millions)": "Population"
}

app = Dash(__name__)

# ---------- UI styles  ----------
CARD = {
    "background": "white",
    "border": "1px solid #e5e7eb",
    "borderRadius": "14px",
    "boxShadow": "0 1px 6px rgba(0,0,0,0.06)",
    "padding": "12px",
}

CONTROL_ROW = {
    "display": "flex",
    "gap": "12px",
    "alignItems": "center",
    "flexWrap": "wrap",
    "marginBottom": "14px",
}

DROPDOWN_STYLE = {
    "width": "250px",
    "borderRadius": "10px",
}

BUTTON_STYLE = {
    "height": "40px",
    "padding": "0 14px",
    "borderRadius": "10px",
    "border": "1px solid #d1d5db",
    "background": "#111827",
    "color": "white",
    "cursor": "pointer",
    "fontWeight": "600",
}


# ---------- FULL LAYOUT ----------
app.layout = html.Div(
    style={
        "backgroundColor": "#f7f7fb",
        "minHeight": "100vh",
        "color": "black",
        "padding": "16px",
    },
    children=[
        dcc.Store(id="selected_country", data=None),

        # Header
        html.Div(
            style={

                "alignItems": "baseline",
                "justifyContent": "space-between",
                "marginBottom": "8px",
            },
            children=[
                html.Div(
                    children=[
                        html.H1(
                            "World Economic Visualization",
                            style={
                                "margin": "0 0 18px 0",
                                "padding": "18px 0",
                                "width": "100%",
                                "textAlign": "center",
                                "fontSize": "32px",
                                "fontWeight": "800",
                                "background": "linear-gradient(90deg, #111827, #374151)",
                                "color": "white",
                                "borderRadius": "12px",
                                "letterSpacing": "0.6px",
                                "boxShadow": "0 4px 10px rgba(0,0,0,0.08)",
                            },
                        ),
                        html.Div(
                            "Select a metric + region, then click a country.",
                            style={"fontSize": "13px", "color": "#6b7280"},
                        ),
                    ]
                ),
            ],
        ),

        # Controls row
        html.Div(
            style=CONTROL_ROW,
            children=[
                dcc.Dropdown(
                    id="metric",
                    options=[{"label": k, "value": v} for k, v in metrics.items()],
                    value="GDP",
                    clearable=False,
                    style=DROPDOWN_STYLE,
                ),
                dcc.Dropdown(
                    id="region",
                    options=[
                        {"label": "Europe", "value": "Europe"},
                        {"label": "Asia", "value": "Asia"},
                        {"label": "Africa", "value": "Africa"},
                        {"label": "Americas", "value": "Americas"},
                        {"label": "Oceania", "value": "Oceania"},
                    ],
                    value=[],
                    multi=True,
                    clearable=True,
                    style=DROPDOWN_STYLE,
                ),
                html.Button("Reset", id="reset", n_clicks=0, style=BUTTON_STYLE),
            ],
        ),

        # Main 2-column area
        html.Div(
            style={"display": "flex", "gap": "18px"},
            children=[
                # -------- LEFT COLUMN: MAP + BAR + LOLLIPOP --------
                html.Div(
                    style={"display": "flex", "flexDirection": "column", "gap": "16px", "flex": "2"},
                    children=[
                        # Map card
                        html.Div(
                            style=CARD,
                            children=[
                                dcc.Graph(id="map", style={"height": "45vh"}),
                            ],
                        ),

                        # Bar card
                        html.Div(
                            style={**CARD, "height": "40vh", "display": "flex", "flexDirection": "column", "minHeight": 0},
                            children=[
                                html.Div(
                                    id="bar_window_label",
                                    style={"fontSize": "13px", "color": "#374151", "marginBottom": "6px"},
                                ),
                                html.Div(
                                    style={"padding": "0 8px 6px 8px"},
                                    children=[
                                        dcc.Slider(
                                            id="bar_window_start",
                                            min=0,
                                            max=1,
                                            step=1,
                                            value=0,
                                            updatemode="drag",
                                            included=False,
                                            tooltip={"placement": "bottom", "always_visible": True},
                                        ),
                                    ],
                                ),
                                dcc.Graph(
                                    id="bar",
                                    style={"flex": "1", "minHeight": 0},
                                    config={"displayModeBar": False},
                                ),
                            ],
                        ),

                        # Lollipop card
                        html.Div(
                            style=CARD,
                            children=[
                                dcc.Graph(id="lollipop", style={"height": "35vh"}),
                            ],
                        ),
                    ],
                ),

                # -------- RIGHT COLUMN: SPIDER + HEATMAP --------
                html.Div(
                    style={"display": "flex", "flexDirection": "column", "gap": "16px", "flex": "1"},
                    children=[
                        html.Div(
                            style=CARD,
                            children=[dcc.Graph(id="spider", style={"height": "38vh"})],
                        ),
                        html.Div(
                            style=CARD,
                            children=[dcc.Graph(id="heatmap", style={"height": "38vh"})],
                        ),
                    ],
                ),
            ],
        ),
    ],
)

def get_map(data, metric, regions=None, selected_country=None, current_view=None, uirevision="map"):
    data = data.copy()
    data[metric] = pd.to_numeric(data[metric], errors="coerce")
    data = data[data[metric] > 0].copy()

    if data.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No data for this selection",
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin=dict(t=50, b=0, l=0, r=0),
            font_color="black",
            uirevision=uirevision,
        )
        return fig

    data["value"] = data[metric]
    data["value_log"] = np.log10(data["value"])

    fig = px.choropleth(
        data,
        geojson=world_map,
        featureidkey="properties.NAME",
        locations="name",
        color="value_log",
        hover_data={metric: ":,.0f", "value_log": False},
        hover_name="name",
        custom_data=["name"],
        color_continuous_scale="Viridis",
        projection="orthographic",
        title=f"{metric} by Country",
    )

    # Highlight selected country in red
    if selected_country:
        sel = str(selected_country).strip()
        all_rows = data.copy()   # BEFORE filtering by metric
        sel_df = all_rows[data["name"].astype(str).str.strip() == sel]
        if not sel_df.empty:
            red_trace = px.choropleth(
                sel_df,
                geojson=world_map,
                featureidkey="properties.NAME",
                locations="name",
                color_discrete_sequence=["#6A0DAD"],
                hover_name="name",
            ).data[0]
            red_trace.name = sel
            red_trace.showlegend = True
            red_trace.legendgroup = "selected"
            red_trace.showscale = False
            fig.add_trace(red_trace)

    # --- Region view presets (rotation + zoom) ---
    region_view = {
        "Europe":   {"rot": {"lon": 15,   "lat": 54},  "scale": 1.85},
        "Asia":     {"rot": {"lon": 90,   "lat": 35},  "scale": 1.55},
        "Africa":   {"rot": {"lon": 20,   "lat": 2},   "scale": 1.75},
        "Americas": {"rot": {"lon": -75,  "lat": 10},  "scale": 1.55},
        "Oceania":  {"rot": {"lon": 135,  "lat": -20}, "scale": 2.10},
    }

    # defaults
    rot = {"lon": 0, "lat": 0}
    scale = 1.0

    # normalize regions
    if regions is None:
        regions = []
    elif isinstance(regions, str):
        regions = [regions]

    # ✅ if we have a current view from relayoutData, use it (prevents snapping back)
    if current_view and "rotation" in current_view and "scale" in current_view:
        rot = current_view["rotation"]
        scale = current_view["scale"]

    # ✅ otherwise, use region preset if exactly one region selected
    elif len(regions) == 1 and regions[0] in region_view:
        rot = region_view[regions[0]]["rot"]
        scale = region_view[regions[0]]["scale"]

    fig.update_layout(
        uirevision=uirevision,  # ✅ key to preserve view across updates
        geo=dict(
            visible=False,
            showcountries=True,
            domain=dict(x=[0.05, 0.95], y=[0.05, 0.95]),
            projection=dict(type="orthographic", rotation=rot, scale=scale),
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        geo_bgcolor="white",
        margin=dict(t=50, b=0, l=0, r=0),
        font_color="black",
    )

    vmin, vmax = data["value_log"].min(), data["value_log"].max()
    ticks = np.linspace(vmin, vmax, 6)
    fig.update_coloraxes(
        colorbar=dict(
            title=f"{metric} (log scale)",
            tickvals=ticks,
            ticktext=[f"{10**v:,.0f}" for v in ticks],
            thickness=18,
            len=0.45,
            x=0.93,
            y=0.5,
            outlinewidth=0,
        )
    )

    return fig

def get_barchart(data, metric, window_start=0, window_size=20, highlight_country=None):
    data = data.copy()
    data[metric] = pd.to_numeric(data[metric], errors="coerce")
    data = data[data[metric] > 0].copy()

    # stable sort
    data = data.sort_values([metric, "name"], ascending=[False, True]).reset_index(drop=True)
    data["rank"] = data.index + 1  # 1-based

    n = len(data)
    if n == 0:
        return px.bar(title="No data available")

    max_start = max(0, n - window_size)
    window_start = int(min(max(window_start or 0, 0), max_start))

    window_df = data.iloc[window_start:window_start + window_size].copy()

    # Highest values should be on top
    window_df = window_df.sort_values([metric, "name"], ascending=[False, True])


    # Highlight
    if highlight_country:
        window_df["highlight"] = window_df["name"].apply(
            lambda x: "Selected" if str(x).strip() == str(highlight_country).strip() else "Other"
        )
    else:
        window_df["highlight"] = "Other"

    a = window_start + 1
    b = min(window_start + window_size, n)

    fig = px.bar(
        window_df,
        x=metric,
        y="name",
        orientation="h",
        text=metric,
        color="highlight",
        color_discrete_map={"Selected": "#6A0DAD", "Other": "#1f77b4"},
        category_orders={"highlight": ["Other", "Selected"]},
        hover_data={"rank": True, metric: ":,.0f", "name": True},
        labels={"name": "Country", metric: metric},
        title=f"{metric} by Country — ranks {a}–{b} of {n}"
    )
    # Force the category order + put the biggest on top
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=window_df["name"].tolist(),
        autorange="reversed"
    )


    fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig.update_layout(
        showlegend=False,
        margin=dict(t=55, b=35, l=150, r=30)
    )
    return fig

def get_heatmap(data, regions=None):
    # Use numeric metric columns only (from dropdown metrics)
    metric_cols = list(metrics.values())
    metric_cols = [m for m in metric_cols if m in data.columns]

    # Keep only numeric data
    corr_data = data[metric_cols].apply(pd.to_numeric, errors="coerce")

    # If not enough data, return empty figure
    if corr_data.shape[1] < 2:
        return px.imshow(title="Not enough data for correlation heatmap")

    # Compute correlation matrix
    corr = corr_data.corr(method="pearson")

    # ----- Title: Global vs Regions -----
    if regions and len(regions) > 0:
        scope = " / ".join(regions)
    else:
        scope = "Global"

    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="Viridis",
        zmin=-1,
        zmax=1,
        title=f"Correlation Heatmap (Pearson) — {scope}"
    )

    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font_color="black",
        margin=dict(t=60, b=40, l=60, r=40)
    )

    return fig

def get_lollipop(data, metric, highlight_country=None):
    df = data.copy()
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df[df[metric].notna()].copy()

    if df.empty:
        return go.Figure().update_layout(title="No data available")

    #  log scale for very skewed metrics like GDP
    use_logx = (str(metric).upper() == "GDP")

    # If we use log scale, we MUST keep only positive values
    if use_logx:
        df = df[df[metric] > 0].copy()
        if df.empty:
            return go.Figure().update_layout(title="No positive values for log scale")

    # Robust outliers via IQR
    q1 = df[metric].quantile(0.25)
    q3 = df[metric].quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr

    df["is_outlier"] = (df[metric] < low) | (df[metric] > high)

    # ----------------- FOCUS MODE : bottom + top + outliers + selected -----------------
    N = 10
    df_sorted = df.sort_values(metric, ascending=True).reset_index(drop=True)

    bottom = df_sorted.head(N)
    top = df_sorted.tail(N)
    outliers = df_sorted[df_sorted["is_outlier"]]

    keep = pd.concat([bottom, top, outliers], ignore_index=True).drop_duplicates(subset=["name"])

    if highlight_country:
        sel = str(highlight_country).strip()
        sel_row = df_sorted[df_sorted["name"].astype(str).str.strip() == sel]
        if not sel_row.empty:
            keep = pd.concat([keep, sel_row], ignore_index=True).drop_duplicates(subset=["name"])

    df = keep.copy()

    # Sort AFTER focusing
    df = df.sort_values(metric, ascending=True).reset_index(drop=True)

    # Color groups
    def label_row(name, is_out):
        if highlight_country and str(name).strip() == str(highlight_country).strip():
            return "Selected"
        return "Outlier" if is_out else "Normal"

    df["group"] = [label_row(n, o) for n, o in zip(df["name"], df["is_outlier"])]

    # Build lollipop: stems + dots
    fig = go.Figure()

    # Stems: softer (less "carpet")
    x0 = min(0, df[metric].min())  # baseline, works also if metric can be negative
    for _, r in df.iterrows():
        fig.add_trace(go.Scatter(
            x=[x0, r[metric]],
            y=[r["name"], r["name"]],
            mode="lines",
            line=dict(width=1),
            showlegend=False,
            hoverinfo="skip",
            opacity=0.18
        ))

    # Dots
    color_map = {"Normal": "#1f77b4", "Outlier": "#ff7f0e", "Selected": "#d62728"}
    size_map = {"Normal": 8, "Outlier": 12, "Selected": 12}

    for gname in ["Normal", "Outlier", "Selected"]:
        gdf = df[df["group"] == gname]
        if gdf.empty:
            continue

        fig.add_trace(go.Scatter(
            x=gdf[metric],
            y=gdf["name"],
            mode="markers",
            name=gname,
            marker=dict(size=size_map[gname], color=color_map[gname]),
            hovertemplate=(
                "<b>%{y}</b><br>"
                f"{metric}: %{{x:,.0f}}<extra></extra>"
            )
        ))

    # Trend markers: median + IQR band
    median = df_sorted[metric].median()  # median over ALL (not only focused)
    fig.add_vrect(x0=low, x1=high, opacity=0.10, line_width=0)
    fig.add_vline(
        x=median,
        line_width=3,
        line_dash="dash",
        annotation_text="Median",
        annotation_position="top"
    )

    # Layout: avoid title clipping and legend collision
    title = f"Lollipop — {metric} (outliers by IQR)"
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", y=0.98, yanchor="top"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font_color="black",
        margin=dict(t=90, b=55, l=140, r=30),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,  # legend below plot => no cut
            xanchor="left",
            x=0
        ),
    )

    if use_logx:
        fig.update_xaxes(type="log", title=f"{metric} (log scale)")
    else:
        fig.update_xaxes(title=metric)

    fig.update_yaxes(title="Country")

    return fig




WINDOW_SIZE = 20

from dash import ctx, State  # make sure you have State imported

@app.callback(
    Output("map", "figure"),
    Output("bar", "figure"),
    Input("metric", "value"),
    Input("region", "value"),
    Input("bar_window_start", "value"),
    Input("selected_country", "data"),
    State("map", "relayoutData"),   # ✅ add this
)
def update_figures(metric, regions, bar_window_start, selected_country, relayoutData):
    try:
        data = dataset.copy()
        if regions:
            data = data[data["region"].isin(regions)].copy()

        # ✅ Extract current view (rotation + scale) from relayoutData
        current_view = None
        if relayoutData:
            lon = relayoutData.get("geo.projection.rotation.lon")
            lat = relayoutData.get("geo.projection.rotation.lat")
            scale = relayoutData.get("geo.projection.scale")
            if lon is not None and lat is not None and scale is not None:
                current_view = {"rotation": {"lon": lon, "lat": lat}, "scale": scale}

        # ✅ Use presets only when region actually triggered the callback (or metric if you want)
        use_current_view = (ctx.triggered_id == "selected_country")

        # ✅ uirevision changes when region selection changes -> allows preset reset on region change,
        # but preserves view for country clicks/metric changes
        uirevision = f"map|regions={','.join(sorted(regions or []))}"

        map_fig = get_map(
            data,
            metric,
            regions,
            selected_country,
            current_view=current_view if use_current_view else None,
            uirevision=uirevision
        )

        bar_fig = get_barchart(
            data,
            metric,
            window_start=bar_window_start,
            window_size=WINDOW_SIZE,
            highlight_country=selected_country
        )

        return map_fig, bar_fig

    except Exception as e:
        fig1 = go.Figure()
        fig1.update_layout(title=f"Map callback error: {type(e).__name__}: {e}")
        fig2 = go.Figure()
        fig2.update_layout(title=f"Bar callback error: {type(e).__name__}: {e}")
        return fig1, fig2



@app.callback(
    Output("metric", "value"),
    Output("region", "value"),
    Input("reset", "n_clicks"),
    prevent_initial_call=True
)
def reset_all(n):
    return "GDP", []



@app.callback(
    Output("bar_window_start", "max"),
    Output("bar_window_start", "marks"),
    Output("bar_window_start", "disabled"),
    Output("bar_window_label", "children"),
    Input("metric", "value"),
    Input("region", "value"),
    Input("bar_window_start", "value"),
)
def update_bar_slider(metric, regions, current_start):
    data = dataset.copy()
    if regions:
        data = data[data["region"].isin(regions)].copy()

    # numeric + positive only
    s = pd.to_numeric(data[metric], errors="coerce")
    data = data[s > 0].copy()
    n = len(data)

    max_start = max(0, n - WINDOW_SIZE)
    disabled = (max_start == 0)

    # light marks: every 20 ranks
    marks = {0: "1"}
    for m in range(20, max_start + 1, 20):
        marks[m] = str(m + 1)

    start = int(current_start or 0)
    start = min(max(start, 0), max_start)
    a = start + 1
    b = min(start + WINDOW_SIZE, n)

    if n == 0:
        label = "No positive values for this metric/region."
    elif disabled:
        label = f"Showing ranks {a}–{b} of {n} (nothing to slide — ≤ {WINDOW_SIZE} countries)."
    else:
        label = f"Showing ranks {a}–{b} of {n} (slide to move the 20-country window)."

    return max_start, marks, disabled, label

@app.callback(
    Output("spider", "figure"),
    Input("map", "clickData"),
    Input("metric", "value"),
    Input("region", "value"),
    Input("reset", "n_clicks"),
)

def update_spider(clickData, metric, regions, n_clicks):

    if ctx.triggered_id == "reset":
        return px.line_polar(
            title="Click a country on the map to see the spider chart"
        )

    if not clickData:
        return px.line_polar(
            title="Click a country on the map to see the spider chart"
        )

    point = clickData["points"][0]
    country = (
        point.get("location")
        or point.get("hovertext")
        or (point.get("customdata")[0] if point.get("customdata") else None)
    )

    if not country:
        return px.line_polar(title="Could not read country from click")

    data = dataset.copy()
    if regions and len(regions) > 0:
        data = data[data["region"].isin(regions)].copy()

    return get_spider(data, str(country).strip())

@app.callback(
    Output("heatmap", "figure"),
    Input("metric", "value"),
    Input("region", "value"),
)
def update_heatmap(metric, regions):
    data = dataset.copy()

    if regions and len(regions) > 0:
        data = data[data["region"].isin(regions)].copy()

    return get_heatmap(data)

@app.callback(
    Output("bar_window_start", "value"),
    Input("selected_country", "data"),
    Input("reset", "n_clicks"),
    Input("metric", "value"),
    Input("region", "value"),
    State("bar_window_start", "value"),
)
def move_bar_window(selected_country, n_clicks, metric, regions, current_start):
    # Reset -> go back to start
    if ctx.triggered_id == "reset":
        return 0

    # No selection -> keep slider where user left it
    if not selected_country:
        return current_start if current_start is not None else 0

    data = dataset.copy()
    if regions:
        data = data[data["region"].isin(regions)].copy()

    data[metric] = pd.to_numeric(data[metric], errors="coerce")
    data = data[data[metric] > 0].copy()
    data = data.sort_values([metric, "name"], ascending=[False, True]).reset_index(drop=True)

    matches = data.index[data["name"].astype(str).str.strip() == str(selected_country).strip()].tolist()
    if not matches:
        return current_start if current_start is not None else 0

    idx = matches[0]
    start = idx - (WINDOW_SIZE // 2)

    max_start = max(0, len(data) - WINDOW_SIZE)
    start = min(max(start, 0), max_start)
    return int(start)


@app.callback(
    Output("selected_country", "data"),
    Input("map", "clickData"),
    Input("reset", "n_clicks"),
    prevent_initial_call=True
)
def set_selected_country(clickData, n_clicks):
    if ctx.triggered_id == "reset":
        return None

    if not clickData or not clickData.get("points"):
        return no_update

    p = clickData["points"][0]
    country = (
        p.get("location")
        or p.get("hovertext")
        or (p.get("customdata")[0] if p.get("customdata") else None)
    )
    return str(country).strip() if country else None

@app.callback(
    Output("lollipop", "figure"),
    Input("metric", "value"),
    Input("region", "value"),
    Input("selected_country", "data"),
)
def update_lollipop(metric, regions, selected_country):
    data = dataset.copy()
    if regions:
        data = data[data["region"].isin(regions)].copy()

    return get_lollipop(data, metric, highlight_country=selected_country)



app.run(jupyter_mode="external", debug=False)


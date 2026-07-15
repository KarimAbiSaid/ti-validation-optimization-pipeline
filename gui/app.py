"""
app.py — entry point for the BIDS_TI_Toolbox GUI.

Multi-page Dash app (dash.register_page): each phase of the toolbox lives in
its own file under pages/, registered there and picked up automatically
here. Pages declare a `category` (and optional `order`) kwarg on their own
dash.register_page() call — extra kwargs Dash stores as-is in the page
registry — and the nav groups by that.

Run with the cloned GUI environment (D:/envs/bids_ti_gui_env), not the
SimNIBS install's own env directly — that one doesn't have Dash installed.
"""
import dash
from dash import Dash, html, dcc

app = Dash(__name__, use_pages=True, suppress_callback_exceptions=True)

CATEGORY_ORDER = ["Preprocessing", "Simulation", "Optimization"]


def _build_nav():
    by_category = {}
    for page in dash.page_registry.values():
        by_category.setdefault(page.get("category", "Other"), []).append(page)
    for pages in by_category.values():
        pages.sort(key=lambda p: p.get("order", 0))

    ordered_categories = [c for c in CATEGORY_ORDER if c in by_category]
    ordered_categories += [c for c in by_category if c not in CATEGORY_ORDER]

    groups = []
    for cat in ordered_categories:
        groups.append(html.Div([
            html.Span(cat, style={"fontWeight": "bold", "marginRight": "0.75rem", "color": "#666"}),
            *[dcc.Link(p["name"], href=p["path"], style={"marginRight": "1.25rem"})
              for p in by_category[cat]],
        ], style={"marginBottom": "0.4rem"}))
    return groups


app.layout = html.Div([
    html.Nav(_build_nav(), style={"padding": "1rem", "borderBottom": "1px solid #ccc",
                                   "marginBottom": "1rem"}),
    dash.page_container,
])

if __name__ == "__main__":
    app.run(debug=True)

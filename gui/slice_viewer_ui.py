"""
slice_viewer_ui.py — reusable 3-plane MRI Slice Viewer UI component:
layout panels (image + crosshair overlay + slider) and the clientside
callbacks that drive slice-scrubbing, crosshair-position sync, and
click/drag crosshair navigation (like a standard MRI viewer — click or
drag on any plane and the other two jump to that point).

No server-side build/poll logic here — actually building the frames
(viz_discovery.build_slice_frames) and wiring whatever button/job/interval
triggers that is left to each page, since that part legitimately differs
per page (FEM Validation's own region/roi-only picker vs. e.g. Analysis
page's already-resolved subject/run/region). Everything here is
prefix-parameterized so more than one page can host its own independent
slice viewer instance without ID collisions — see fem_validation.py's
"Slice Viewer" section and analysis.py's "ROI Preview" section for the two
current call sites.
"""
import dash
from dash import html, dcc, Output, Input, State

CROSSHAIR_COLOR = "rgba(0,255,255,0.85)"


def _crosshair_h_style():
    return {"position": "absolute", "left": 0, "top": "50%", "width": "100%", "height": "0",
           "borderTop": f"1px solid {CROSSHAIR_COLOR}", "pointerEvents": "none"}


def _crosshair_v_style():
    return {"position": "absolute", "top": 0, "left": "50%", "height": "100%", "width": "0",
           "borderLeft": f"1px solid {CROSSHAIR_COLOR}", "pointerEvents": "none"}


def slice_panel(prefix: str, plane: str, title: str):
    """One Slice Viewer panel (image + crosshair overlay + slider). IDs are
    f"{prefix}-{plane}-img"/"-slider"/"-crosshair-h"/"-crosshair-v". The
    image sits in a position:relative wrapper so the two crosshair lines
    (position:absolute, driven by the OTHER two planes' slider values — see
    register_clientside_callbacks() below) draw directly on top of it
    without a canvas or extra image re-render."""
    return html.Div([
        html.H5(title),
        html.Div([
            html.Img(id=f"{prefix}-{plane}-img", style={"width": "100%", "display": "block"}),
            html.Div(id=f"{prefix}-{plane}-crosshair-h", style=_crosshair_h_style()),
            html.Div(id=f"{prefix}-{plane}-crosshair-v", style=_crosshair_v_style()),
        ], style={"position": "relative", "lineHeight": 0}),
        dcc.Slider(id=f"{prefix}-{plane}-slider", min=0, max=0, step=1, value=0),
    ], style={"flex": "1", "marginRight": "0.75rem", "minWidth": "200px"})


def viewer_container(prefix: str, style: dict | None = None):
    """The full 3-panel row (Axial/Coronal/Sagittal) plus the per-plane
    frame Stores and the click-init Store this prefix's clientside
    callbacks need — everything a page drops straight into its layout.
    Container id is f"{prefix}-viewer-container" (register_clientside_
    callbacks()'s click-listener trigger fires off this id existing) —
    style defaults to display:none, flip it to flex/etc. once a build
    finishes, same as every other job-store-driven section in this app."""
    return html.Div([
        html.Div([
            slice_panel(prefix, "axial", "Axial"),
            slice_panel(prefix, "coronal", "Coronal"),
            slice_panel(prefix, "sagittal", "Sagittal"),
        ], id=f"{prefix}-viewer-container", style=style or {"display": "none", "flexWrap": "wrap",
                                                             "marginTop": "1rem"}),
        dcc.Store(id=f"{prefix}-axial-frames"),
        dcc.Store(id=f"{prefix}-coronal-frames"),
        dcc.Store(id=f"{prefix}-sagittal-frames"),
        dcc.Store(id=f"{prefix}-click-init"),
    ])


# Clientside: swap the shown slice image when a slider moves, reading from
# the already-downloaded frame list — no server round-trip, so scrubbing
# stays smooth even on a slow connection. Same function body for every
# prefix (only the Output/Input/State component ids differ, not the logic).
_SLICE_CLIENTSIDE_JS = """
function(idx, frames) {
    if (!frames || idx === undefined || idx === null || !frames[idx]) {
        return window.dash_clientside.no_update;
    }
    return "data:image/png;base64," + frames[idx];
}
"""


def _crosshair_js(img_id: str) -> str:
    """Crosshair overlay, like a standard MRI viewer — each plane's two
    guide lines mark where the OTHER two planes' sliders currently point.
    Purely a coordinate transform off values already in the browser (each
    image's own natural pixel size, which is exactly the downsampled
    volume's in-plane shape — see viz_discovery._oriented_slice()) — no new
    data, no server round-trip."""
    return (
        'function(colVal, rowVal) {\n'
        f'    var img = document.getElementById("{img_id}");\n'
        '    if (!img || !img.naturalWidth || !img.naturalHeight || colVal === undefined ||\n'
        '        colVal === null || rowVal === undefined || rowVal === null) {\n'
        '        return [window.dash_clientside.no_update, window.dash_clientside.no_update];\n'
        '    }\n'
        '    var n0 = img.naturalWidth, n1 = img.naturalHeight;\n'
        '    var leftPct = (n0 > 1) ? (colVal / (n0 - 1)) * 100 : 50;\n'
        '    var row = (n1 - 1) - rowVal;\n'
        '    var topPct = (n1 > 1) ? (row / (n1 - 1)) * 100 : 50;\n'
        '    var hStyle = {position: "absolute", left: 0, width: "100%", height: "0",\n'
        '                  borderTop: "1px solid rgba(0,255,255,0.85)", pointerEvents: "none",\n'
        '                  top: topPct + "%"};\n'
        '    var vStyle = {position: "absolute", top: 0, height: "100%", width: "0",\n'
        '                  borderLeft: "1px solid rgba(0,255,255,0.85)", pointerEvents: "none",\n'
        '                  left: leftPct + "%"};\n'
        '    return [hStyle, vStyle];\n'
        '}'
    )


def _click_init_js(prefix: str) -> str:
    """One-time: attach click/drag listeners directly to the three slice
    <img> elements (they exist in the DOM from initial page load, just
    display:none until a build finishes) so clicking/dragging on one plane
    moves the OTHER two planes' sliders — same as a standard MRI viewer's
    crosshair navigation. Uses dash_clientside.set_props (Dash >=2.9) to
    push the new slider values exactly as if the user had dragged them,
    which in turn drives both the image-swap callback and the crosshair-
    redraw callbacks — no extra wiring needed for those. Reads each image's
    own naturalWidth/naturalHeight at click time (not at attach time), so
    there's no dependency on frames having already loaded when this runs."""
    lines = [
        'function(_container_id) {',
        '    function attach(imgId, onPick) {',
        '        var img = document.getElementById(imgId);',
        '        if (!img || img._sliceClickAttached) { return; }',
        '        img._sliceClickAttached = true;',
        '        img.style.cursor = "crosshair";',
        '        var dragging = false;',
        '        function handle(e) {',
        '            var rect = img.getBoundingClientRect();',
        '            if (rect.width === 0 || rect.height === 0) { return; }',
        '            var xFrac = (e.clientX - rect.left) / rect.width;',
        '            var yFrac = (e.clientY - rect.top) / rect.height;',
        '            if (xFrac < 0 || xFrac > 1 || yFrac < 0 || yFrac > 1) { return; }',
        '            var col = Math.round(xFrac * (img.naturalWidth - 1));',
        '            var row = Math.round(yFrac * (img.naturalHeight - 1));',
        '            onPick(row, col, img.naturalWidth, img.naturalHeight);',
        '        }',
        '        img.addEventListener("mousedown", function(e) { dragging = true; handle(e); '
        'e.preventDefault(); });',
        '        window.addEventListener("mousemove", function(e) { if (dragging) { handle(e); } });',
        '        window.addEventListener("mouseup", function() { dragging = false; });',
        '    }',
        '',
        f'    attach("{prefix}-axial-img", function(row, col, w, h) {{',
        f'        window.dash_clientside.set_props("{prefix}-sagittal-slider", {{value: col}});',
        f'        window.dash_clientside.set_props("{prefix}-coronal-slider", {{value: (h - 1) - row}});',
        '    });',
        f'    attach("{prefix}-coronal-img", function(row, col, w, h) {{',
        f'        window.dash_clientside.set_props("{prefix}-sagittal-slider", {{value: col}});',
        f'        window.dash_clientside.set_props("{prefix}-axial-slider", {{value: (h - 1) - row}});',
        '    });',
        f'    attach("{prefix}-sagittal-img", function(row, col, w, h) {{',
        f'        window.dash_clientside.set_props("{prefix}-coronal-slider", {{value: col}});',
        f'        window.dash_clientside.set_props("{prefix}-axial-slider", {{value: (h - 1) - row}});',
        '    });',
        '',
        '    return true;',
        '}',
    ]
    return '\n'.join(lines)


def register_clientside_callbacks(prefix: str) -> None:
    """Call once per prefix (module level, right after the page that hosts
    viewer_container(prefix) is defined) to wire slice-scrubbing, crosshair
    redraw, and click/drag crosshair navigation for that prefix's panels.
    Safe to call for several DIFFERENT prefixes (each page gets its own
    independent callback set) — calling it twice for the SAME prefix would
    register duplicate Dash callbacks against the same Output, don't."""
    for plane in ("axial", "coronal", "sagittal"):
        dash.clientside_callback(
            _SLICE_CLIENTSIDE_JS,
            Output(f"{prefix}-{plane}-img", "src"),
            Input(f"{prefix}-{plane}-slider", "value"),
            State(f"{prefix}-{plane}-frames", "data"),
        )

    # For each plane: which slider drives the vertical (left%) line, and
    # which drives the horizontal (top%) line — see _oriented_slice()'s
    # row/col derivation in viz_discovery.py for why it's always "Sagittal
    # -> column" and "the remaining one of {Coronal, Axial} -> row".
    slider_inputs = {
        "axial":    (f"{prefix}-sagittal-slider", f"{prefix}-coronal-slider"),
        "coronal":  (f"{prefix}-sagittal-slider", f"{prefix}-axial-slider"),
        "sagittal": (f"{prefix}-coronal-slider",  f"{prefix}-axial-slider"),
    }
    for plane, (col_slider, row_slider) in slider_inputs.items():
        dash.clientside_callback(
            _crosshair_js(f"{prefix}-{plane}-img"),
            Output(f"{prefix}-{plane}-crosshair-h", "style"),
            Output(f"{prefix}-{plane}-crosshair-v", "style"),
            Input(col_slider, "value"),
            Input(row_slider, "value"),
        )

    dash.clientside_callback(
        _click_init_js(prefix),
        Output(f"{prefix}-click-init", "data"),
        Input(f"{prefix}-viewer-container", "id"),
    )

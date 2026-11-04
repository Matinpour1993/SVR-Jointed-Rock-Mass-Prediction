# -*- coding: utf-8 -*-
"""
=====================================================================
Non-persistent Jointed Rock Predictor — GUI
Two independent SVR models optimized by PSO and RSA
---------------------------------------------------------------------
Four inputs: JRC, L, γ, θ
Outputs: σ_cj (MPa) and E_j (GPa)
---------------------------------------------------------------------
Professional Light Slate & Teal theme · Times New Roman
=====================================================================
"""

import os
import joblib
import tkinter as tk
from tkinter import font, messagebox
import numpy as np


# ============================================================
# 1) MODEL BUNDLE CLASS
# ============================================================

class ModelBundle:
    """
    Wrapper for a trained SVR model together with its
    input and output scalers.

    Input order: JRC, L, γ, θ

    Raw inputs are scaled using scaler_x. Model predictions
    are transformed back to the original output scale using scaler_y.
    """

    def __init__(self, model, scaler_x, scaler_y):
        self.model = model
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y

    def predict(self, raw_input):

        raw_input = np.asarray(
            raw_input,
            dtype=float
        )

        if raw_input.ndim == 1:
            raw_input = raw_input.reshape(1, -1)

        if hasattr(
            self.scaler_x,
            "n_features_in_"
        ):

            expected_features = (
                self.scaler_x.n_features_in_
            )

            supplied_features = (
                raw_input.shape[1]
            )

            if supplied_features != expected_features:

                raise ValueError(
                    f"Input feature mismatch: "
                    f"the saved model expects "
                    f"{expected_features} features, "
                    f"but the GUI supplied "
                    f"{supplied_features}."
                )

        X_scaled = self.scaler_x.transform(
            raw_input
        )

        pred_scaled = np.asarray(
            self.model.predict(X_scaled),
            dtype=float
        ).reshape(-1, 1)

        pred_original = (
            self.scaler_y.inverse_transform(
                pred_scaled
            )
        )

        return pred_original.ravel()


# ============================================================
# 2) BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# 3) MODEL FILES
# ============================================================

MODEL_PATHS = {
    "SVR-PSO_UCSj": "SVR-PSO_UCSj.pkl",
    "SVR-RSA_Ej": "SVR-RSA_Ej.pkl",
}


# ============================================================
# 4) LOAD MODELS
# ============================================================

models = {}
model_errors = {}


for model_name, filename in MODEL_PATHS.items():

    model_path = os.path.join(
        BASE_DIR,
        filename
    )

    try:

        if not os.path.exists(model_path):

            model_errors[model_name] = (
                f"Model file not found: {filename}"
            )

            print(
                f"[WARNING] {model_errors[model_name]}"
            )

            continue

        loaded_object = joblib.load(
            model_path
        )

        if isinstance(
            loaded_object,
            ModelBundle
        ):

            models[model_name] = loaded_object

        elif isinstance(
            loaded_object,
            dict
        ):

            required_keys = [
                "model",
                "scaler_x",
                "scaler_y"
            ]

            if all(
                key in loaded_object
                for key in required_keys
            ):

                models[model_name] = ModelBundle(
                    loaded_object["model"],
                    loaded_object["scaler_x"],
                    loaded_object["scaler_y"]
                )

            else:

                raise ValueError(
                    "Dictionary does not contain "
                    "'model', 'scaler_x', and 'scaler_y'."
                )

        elif isinstance(
            loaded_object,
            (tuple, list)
        ):

            if len(loaded_object) == 3:

                models[model_name] = ModelBundle(
                    loaded_object[0],
                    loaded_object[1],
                    loaded_object[2]
                )

            else:

                raise ValueError(
                    "Tuple/list model file must contain "
                    "(model, scaler_x, scaler_y)."
                )

        else:

            raise ValueError(
                "Unsupported .pkl format. Expected "
                "ModelBundle, dictionary, or "
                "(model, scaler_x, scaler_y)."
            )

        print(
            f"[OK] Model loaded successfully: "
            f"{model_name}"
        )

    except Exception as exc:

        model_errors[model_name] = str(exc)

        print(
            f"[ERROR] Failed to load "
            f"{model_name}: {exc}"
        )


# ============================================================
# 5) INPUT / OUTPUT SPECIFICATIONS
# ============================================================

INPUT_LABELS = [
    ("JRC", "Joint roughness coefficient", "1 – 19"),
    ("L", "Bridge length (mm)", "10 – 40"),
    ("γ", "Bridge angle (degree)", "90 – 180"),
    ("θ", "Joint inclination (degree)", "0 – 90"),
]

# Minimum and maximum values are inclusive and follow the input order above.
INPUT_RANGES = [
    (1.0, 19.0),
    (10.0, 40.0),
    (90.0, 180.0),
    (0.0, 90.0),
]

OUTPUT_LABELS = {
    "SVR-PSO_UCSj": ("Jointed-slab compressive strength", "MPa", "σ", "cj"),
    "SVR-RSA_Ej": ("Deformation modulus", "GPa", "E", "j"),
}


# ============================================================
# 6) INPUT VALIDATION
# ============================================================

# ============================================================
# 7) COLOR PALETTE
# ============================================================

BG_COLOR = "#eaf0f6"

HEADER_BG = "#26384d"

# Brighter header text for clear contrast on the dark header background.
HEADER_TEXT = "#f4f7fb"

FRAME_COLOR = "#ffffff"

CARD_BG = "#f4f7fb"

ENTRY_BG = "#ffffff"

ENTRY_BORDER = "#b8c7d8"

GOLD = "#16827f"          # Teal accent

GOLD_HOVER = "#20a19b"

STEEL_BLUE = "#3f789f"

TEXT_PRIMARY = "#263548"

TEXT_SECONDARY = "#61748a"

TEXT_DIM = "#8796a8"

ERROR_RED = "#c44750"

SUCCESS_GREEN = "#27845b"


# ============================================================
# 8) VALIDATION MESSAGE FUNCTION
# ============================================================

def clear_validation_message():

    validation_label.config(
        text="",
        fg=TEXT_SECONDARY
    )

    for frame in input_entry_frames:

        frame.configure(
            highlightbackground=ENTRY_BORDER,
            highlightcolor=STEEL_BLUE,
            highlightthickness=1
        )


def show_validation_error(
    message,
    invalid_indices=None
):

    if invalid_indices is None:
        invalid_indices = []

    validation_label.config(
        text=message,
        fg=ERROR_RED
    )

    for i, frame in enumerate(
        input_entry_frames
    ):

        if i in invalid_indices:

            frame.configure(
                highlightbackground=ERROR_RED,
                highlightcolor=ERROR_RED,
                highlightthickness=2
            )

        else:

            frame.configure(
                highlightbackground=ENTRY_BORDER,
                highlightcolor=STEEL_BLUE,
                highlightthickness=1
            )


# ============================================================
# 9) INPUT VALIDATION
# ============================================================

def validate_inputs():

    clear_validation_message()

    try:
        input_values = [float(entry.get().strip()) for entry in entries]
    except ValueError:
        show_validation_error(
            "Please enter valid numeric values for all four inputs.",
            []
        )
        return False, None

    invalid_indices = [
        i for i, value in enumerate(input_values)
        if not np.isfinite(value)
    ]

    if invalid_indices:
        show_validation_error(
            "All input values must be finite numbers.",
            invalid_indices
        )
        entries[invalid_indices[0]].focus_set()
        return False, None

    out_of_range = [
        i for i, (value, (minimum, maximum))
        in enumerate(zip(input_values, INPUT_RANGES))
        if value < minimum or value > maximum
    ]

    if out_of_range:
        messages = [
            f"{INPUT_LABELS[i][0]} must be between "
            f"{INPUT_RANGES[i][0]:g} and {INPUT_RANGES[i][1]:g}."
            for i in out_of_range
        ]
        show_validation_error(
            "Input value outside the permitted range: " + " ".join(messages),
            out_of_range
        )
        entries[out_of_range[0]].focus_set()
        return False, None

    return True, input_values


# ============================================================
# 11) PREDICTION FUNCTION
# ============================================================

def run_prediction():

    valid, input_values = validate_inputs()

    if not valid:
        return

    X = np.asarray(input_values, dtype=float).reshape(1, 4)

    try:
        for model_name, model in models.items():
            prediction = np.asarray(model.predict(X), dtype=float).ravel()

            if len(prediction) == 0 or not np.isfinite(prediction[0]):
                raise ValueError(
                    f"{model_name} returned an invalid prediction."
                )

            value_labels[model_name].config(
                text=f"{float(prediction[0]):.4f}",
                fg=GOLD
            )

            unit_labels[model_name].config(
                text=OUTPUT_LABELS[model_name][1],
                fg=TEXT_SECONDARY
            )

        validation_label.config(
            text="Prediction completed successfully.",
            fg=SUCCESS_GREEN
        )

        for frame in input_entry_frames:
            frame.configure(
                highlightbackground=ENTRY_BORDER,
                highlightcolor=STEEL_BLUE,
                highlightthickness=1
            )

    except Exception as exc:
        messagebox.showerror(
            "Prediction Error",
            f"An error occurred:\n\n{exc}"
        )


# ============================================================
# 12) MODEL SENSITIVITY DIAGNOSTIC
# ============================================================

def run_model_diagnostic():

    print()
    print("=" * 72)
    print("SVR GUI MODEL DIAGNOSTIC")
    print("=" * 72)
    print("Expected input order: JRC -> L -> γ -> θ")
    print()

    for model_name, model in models.items():
        try:
            scaler_x = model.scaler_x

            if (
                hasattr(scaler_x, "data_min_")
                and hasattr(scaler_x, "data_max_")
                and len(scaler_x.data_min_) == 4
            ):
                x_min = np.asarray(scaler_x.data_min_, dtype=float)
                x_max = np.asarray(scaler_x.data_max_, dtype=float)
                X_probe = np.vstack([x_min, (x_min + x_max) / 2.0, x_max])
            else:
                X_probe = np.array(
                    [[0.0, 0.0, 0.0, 0.0],
                     [0.5, 0.5, 0.5, 0.5],
                     [1.0, 1.0, 1.0, 1.0]], dtype=float
                )

            predictions = np.asarray(model.predict(X_probe), dtype=float).ravel()
            output_unit = OUTPUT_LABELS[model_name][1]

            print(f"\n{model_name}")
            print("-" * 45)
            for i, pred in enumerate(predictions, start=1):
                print(f"  Probe {i}: {pred:.8f} {output_unit}")

            prediction_range = np.max(predictions) - np.min(predictions)
            print(f"Prediction range: {prediction_range:.10f}")

            if np.isclose(prediction_range, 0.0, atol=1e-10):
                print("[WARNING] The saved model produces essentially the same prediction for different inputs.")
            else:
                print("[OK] The saved model responds to changes in input values.")

        except Exception as exc:
            print(f"[ERROR] Diagnostic failed for {model_name}: {exc}")

    print("\n" + "=" * 72)
    print("Diagnostic completed.")
    print("=" * 72 + "\n")


# ============================================================
# 13) ENTRY FIELD FOCUS EFFECTS
# ============================================================

def on_focus_in(
    _event,
    frame
):

    frame.configure(
        highlightbackground=STEEL_BLUE,
        highlightcolor=STEEL_BLUE,
        highlightthickness=2
    )


def on_focus_out(
    _event,
    frame
):

    frame.configure(
        highlightbackground=ENTRY_BORDER,
        highlightcolor=STEEL_BLUE,
        highlightthickness=1
    )


# ============================================================
# 14) MAIN WINDOW
# ============================================================

root = tk.Tk()

root.title(
    "Non-Persistent Jointed Rock Strength and Deformability Prediction Tool"
)

root.configure(
    bg=BG_COLOR
)

root.resizable(
    False,
    False
)


# ============================================================
# 15) FONTS
# ============================================================

f_title = font.Font(
    family="Times New Roman",
    size=21,
    weight="bold"
)

f_subtitle = font.Font(
    family="Times New Roman",
    size=14,
    slant="italic"
)

f_section = font.Font(
    family="Times New Roman",
    size=16,
    weight="bold"
)

f_label = font.Font(
    family="Times New Roman",
    size=14,
    weight="bold"
)

f_symbol = font.Font(
    family="Times New Roman",
    size=13,
    weight="bold"
)

# ------------------------------------------------------------
# UPDATED:
# Allowed-range font increased from size 10 to size 12
# ------------------------------------------------------------

f_range = font.Font(
    family="Times New Roman",
    size=12,
    slant="italic"
)

f_entry = font.Font(
    family="Times New Roman",
    size=14
)

f_button = font.Font(
    family="Times New Roman",
    size=15,
    weight="bold"
)

f_cardname = font.Font(
    family="Times New Roman",
    size=14,
    weight="bold"
)

f_value = font.Font(
    family="Times New Roman",
    size=16,
    weight="bold"
)

f_unit = font.Font(
    family="Times New Roman",
    size=12
)

f_hint = font.Font(
    family="Times New Roman",
    size=11,
    slant="italic"
)

f_validation = font.Font(
    family="Times New Roman",
    size=10,
    weight="bold"
)

f_subscript_title = font.Font(
    family="Times New Roman",
    size=12,
    weight="bold"
)

f_subscript_card = font.Font(
    family="Times New Roman",
    size=9,
    weight="bold"
)


def add_subscript_symbol(parent, base_text, sub_text, base_font, sub_font,
                         bg, fg):
    """Build a symbol from two labels so the subscript is truly lowered."""
    symbol_frame = tk.Frame(parent, bg=bg, bd=0, highlightthickness=0)
    base_label = tk.Label(symbol_frame, text=base_text, font=base_font,
                          bg=bg, fg=fg, bd=0)
    sub_label = tk.Label(symbol_frame, text=sub_text, font=sub_font,
                         bg=bg, fg=fg, bd=0)

    base_label.update_idletasks()
    sub_label.update_idletasks()
    base_width = base_label.winfo_reqwidth()
    base_height = base_label.winfo_reqheight()
    sub_width = sub_label.winfo_reqwidth()
    frame_width = max(1, base_width + sub_width - 1)
    frame_height = max(base_height + 2, sub_label.winfo_reqheight() + 5)

    symbol_frame.configure(width=frame_width, height=frame_height)
    symbol_frame.pack_propagate(False)
    base_label.place(x=0, y=0)
    sub_label.place(x=max(0, base_width - 2), y=max(4, base_height // 2 - 1))
    return symbol_frame


# ============================================================
# 16) HEADER
# ============================================================

header = tk.Frame(
    root,
    bg=HEADER_BG
)

header.pack(
    fill="x"
)

header_title = tk.Frame(header, bg=HEADER_BG)
header_title.pack(pady=(12, 2))

tk.Label(
    header_title, text="Prediction of", font=f_title,
    bg=HEADER_BG, fg=HEADER_TEXT
).pack(side="left")

add_subscript_symbol(
    header_title, "σ", "cj", f_title, f_subscript_title,
    HEADER_BG, HEADER_TEXT
).pack(side="left", padx=(5, 7))

tk.Label(
    header_title, text="and", font=f_title,
    bg=HEADER_BG, fg=HEADER_TEXT
).pack(side="left")

add_subscript_symbol(
    header_title, "E", "j", f_title, f_subscript_title,
    HEADER_BG, HEADER_TEXT
).pack(side="left", padx=(5, 0))

tk.Label(
    header,
    text="SVR-PSO and SVR-RSA Models",
    font=f_subtitle,
    bg=HEADER_BG,
    fg=HEADER_TEXT
).pack(
    pady=(0, 10)
)

tk.Frame(
    root,
    bg=GOLD,
    height=2
).pack(
    fill="x"
)


# ============================================================
# 17) MAIN AREA
# ============================================================

main = tk.Frame(
    root,
    bg=BG_COLOR
)

main.pack(
    fill="both",
    padx=14,
    pady=(10, 12)
)

main.grid_columnconfigure(
    0,
    minsize=440
)

main.grid_columnconfigure(
    1,
    minsize=440
)


# ============================================================
# 18) LEFT PANEL
# ============================================================

left = tk.Frame(
    main,
    bg=FRAME_COLOR,
    highlightbackground=ENTRY_BORDER,
    highlightthickness=2,
    padx=14,
    pady=12
)

left.grid(
    row=0,
    column=0,
    sticky="nsew"
)


# ============================================================
# 19) LEFT PANEL HEADER
# ============================================================

tk.Frame(
    left,
    bg=STEEL_BLUE,
    height=2
).pack(
    fill="x",
    pady=(0, 5)
)

tk.Label(
    left,
    text="INPUT PARAMETERS",
    font=f_section,
    bg=FRAME_COLOR,
    fg=STEEL_BLUE,
    anchor="w"
).pack(
    fill="x",
    pady=(0, 5)
)

tk.Label(
    left,
    text="Enter the four parameters used by the models.",
    font=f_hint,
    bg=FRAME_COLOR,
    fg=TEXT_SECONDARY,
    anchor="w"
).pack(
    fill="x",
    pady=(0, 8)
)


# ============================================================
# 20) INPUT FIELDS
# ============================================================

entries = []

input_entry_frames = []


for index, (
    symbol,
    fullname,
    range_text
) in enumerate(
    INPUT_LABELS
):

    row = tk.Frame(
        left,
        bg=CARD_BG,
        highlightbackground=ENTRY_BORDER,
        highlightthickness=1
    )

    row.pack(
        fill="x",
        pady=4
    )

    text_frame = tk.Frame(
        row,
        bg=CARD_BG
    )

    text_frame.pack(
        side="left",
        fill="x",
        expand=True,
        padx=(10, 5),
        pady=7
    )

    tk.Label(
        text_frame,
        text=fullname,
        font=f_label,
        bg=CARD_BG,
        fg=TEXT_PRIMARY,
        anchor="w"
    ).pack(
        fill="x"
    )

    tk.Label(
        text_frame,
        text=symbol,
        font=f_symbol,
        bg=CARD_BG,
        fg=TEXT_SECONDARY,
        anchor="w"
    ).pack(
        fill="x",
        pady=(1, 0)
    )

    range_label = tk.Label(
        text_frame,
        text=(
            "Allowed range: "
            f"{range_text}"
        ),
        font=f_range,
        bg=CARD_BG,
        fg=TEXT_SECONDARY,
        anchor="w"
    )

    range_label.pack(
        fill="x",
        pady=(2, 0)
    )

    entry_frame = tk.Frame(
        row,
        bg=ENTRY_BG,
        highlightbackground=ENTRY_BORDER,
        highlightcolor=STEEL_BLUE,
        highlightthickness=1
    )

    entry_frame.pack(
        side="right",
        padx=(5, 10),
        pady=8
    )

    entry = tk.Entry(
        entry_frame,
        font=f_entry,
        width=11,
        justify="center",
        bd=0,
        relief="flat",
        bg=ENTRY_BG,
        fg=TEXT_PRIMARY,
        insertbackground=TEXT_PRIMARY
    )

    entry.pack(
        padx=6,
        pady=5
    )

    entry.bind(
        "<FocusIn>",
        lambda e, f=entry_frame:
        on_focus_in(e, f)
    )

    entry.bind(
        "<FocusOut>",
        lambda e, f=entry_frame:
        on_focus_out(e, f)
    )

    entries.append(entry)

    input_entry_frames.append(
        entry_frame
    )


# ============================================================
# 21) INPUT FIELD SETUP
# ============================================================

# ============================================================
# 22) INPUT ORDER
# ============================================================

tk.Label(
    left,
    text="Input order: JRC  →  L  →  γ  →  θ",
    font=f_hint,
    bg=FRAME_COLOR,
    fg=STEEL_BLUE,
    anchor="center"
).pack(
    fill="x",
    pady=(8, 3)
)


# ============================================================
# 23) VALIDATION MESSAGE AREA
# ============================================================

validation_label = tk.Label(
    left,
    text="",
    font=f_validation,
    bg=FRAME_COLOR,
    fg=ERROR_RED,
    anchor="center",
    justify="center",
    wraplength=400
)

validation_label.pack(
    fill="x",
    pady=(5, 2)
)


# ============================================================
# 24) RIGHT PANEL
# ============================================================

right = tk.Frame(
    main,
    bg=FRAME_COLOR,
    highlightbackground=ENTRY_BORDER,
    highlightthickness=2,
    padx=14,
    pady=12
)

right.grid(
    row=0,
    column=1,
    sticky="nsew",
    padx=(10, 0)
)


# ============================================================
# 25) RIGHT PANEL HEADER
# ============================================================

tk.Frame(
    right,
    bg=GOLD,
    height=2
).pack(
    fill="x",
    pady=(0, 5)
)

tk.Label(
    right,
    text="PREDICTION",
    font=f_section,
    bg=FRAME_COLOR,
    fg=GOLD,
    anchor="w"
).pack(
    fill="x",
    pady=(0, 8)
)


# ============================================================
# 26) PREDICT BUTTON
# ============================================================

btn_predict = tk.Button(
    right,
    text="PREDICT OUTPUTS",
    font=f_button,
    bg=GOLD,
    fg=BG_COLOR,
    relief="flat",
    bd=0,
    activebackground=GOLD_HOVER,
    activeforeground=BG_COLOR,
    cursor="hand2",
    command=run_prediction
)

btn_predict.pack(
    fill="x",
    pady=(0, 4)
)

btn_predict.bind(
    "<Enter>",
    lambda _e:
    btn_predict.configure(
        bg=GOLD_HOVER
    )
)

btn_predict.bind(
    "<Leave>",
    lambda _e:
    btn_predict.configure(
        bg=GOLD
    )
)


tk.Label(
    right,
    text="Click PREDICT or press Enter ↵",
    font=f_hint,
    bg=FRAME_COLOR,
    fg=TEXT_SECONDARY
).pack(
    fill="x",
    pady=(0, 8)
)


# ============================================================
# 27) RESULTS HEADER
# ============================================================

tk.Frame(
    right,
    bg=ENTRY_BORDER,
    height=1
).pack(
    fill="x",
    pady=(0, 8)
)

tk.Label(
    right,
    text="MODEL RESULTS",
    font=f_section,
    bg=FRAME_COLOR,
    fg=STEEL_BLUE,
    anchor="w"
).pack(
    fill="x",
    pady=(0, 7)
)


# ============================================================
# 28) RESULT CARDS
# ============================================================

value_labels = {}
unit_labels = {}

for model_name in MODEL_PATHS.keys():

    card = tk.Frame(
        right,
        bg=CARD_BG,
        highlightbackground=ENTRY_BORDER,
        highlightthickness=1
    )
    card.pack(fill="x", pady=5)

    tk.Frame(card, bg=GOLD, width=4).pack(side="left", fill="y")

    output_description, output_unit, symbol_base, symbol_sub = OUTPUT_LABELS[model_name]

    # Model name and target are consistently left-aligned.
    info_frame = tk.Frame(card, bg=CARD_BG)
    info_frame.pack(side="left", fill="both", expand=True,
                    padx=(10, 5), pady=8)

    tk.Label(
        info_frame, text=model_name, font=f_cardname,
        bg=CARD_BG, fg=TEXT_PRIMARY, anchor="w", justify="left"
    ).pack(fill="x", anchor="w")

    description_frame = tk.Frame(info_frame, bg=CARD_BG)
    description_frame.pack(fill="x", anchor="w", pady=(2, 0))

    tk.Label(
        description_frame, text=output_description, font=f_hint,
        bg=CARD_BG, fg=TEXT_SECONDARY, anchor="w"
    ).pack(side="left")

    add_subscript_symbol(
        description_frame, symbol_base, symbol_sub,
        f_symbol, f_subscript_card, CARD_BG, TEXT_SECONDARY
    ).pack(side="left", padx=(3, 0), pady=(0, 1))

    result_frame = tk.Frame(card, bg=CARD_BG)
    result_frame.pack(side="right", padx=(4, 10), pady=8)

    unit_label = tk.Label(
        result_frame, text=output_unit, font=f_unit,
        bg=CARD_BG, fg=TEXT_SECONDARY
    )
    unit_label.pack(side="bottom", anchor="e")

    value_label = tk.Label(
        result_frame, text="—", font=f_value,
        bg=CARD_BG, fg=TEXT_DIM
    )
    value_label.pack(side="top", anchor="e")

    value_labels[model_name] = value_label
    unit_labels[model_name] = unit_label


# ============================================================
# 29) MODEL LOADING STATUS
# ============================================================

loaded_count = len(
    models
)

total_count = len(
    MODEL_PATHS
)


if loaded_count == total_count:

    status_text = (
        f"{loaded_count} of {total_count} models "
        f"loaded successfully"
    )

    status_color = SUCCESS_GREEN

else:

    status_text = (
        f"{loaded_count} of {total_count} models loaded"
    )

    status_color = ERROR_RED


tk.Label(
    right,
    text=status_text,
    font=f_hint,
    bg=FRAME_COLOR,
    fg=status_color
).pack(
    fill="x",
    pady=(8, 2)
)


output_note = tk.Frame(right, bg=FRAME_COLOR)
output_note.pack(fill="x", pady=(2, 0), anchor="e")

tk.Label(
    output_note, text="Outputs:", font=f_hint,
    bg=FRAME_COLOR, fg=TEXT_DIM
).pack(side="left")

add_subscript_symbol(
    output_note, "σ", "cj", f_hint, f_subscript_card,
    FRAME_COLOR, TEXT_DIM
).pack(side="left", padx=(4, 1))

tk.Label(
    output_note, text="(MPa) and", font=f_hint,
    bg=FRAME_COLOR, fg=TEXT_DIM
).pack(side="left", padx=(0, 4))

add_subscript_symbol(
    output_note, "E", "j", f_hint, f_subscript_card,
    FRAME_COLOR, TEXT_DIM
).pack(side="left", padx=(0, 1))

tk.Label(
    output_note, text="(GPa) · results rounded to 4 decimals",
    font=f_hint, bg=FRAME_COLOR, fg=TEXT_DIM
).pack(side="left")


# ============================================================
# 30) FIXED WINDOW SIZE AND POSITION
# ============================================================

root.update_idletasks()

window_width = root.winfo_reqwidth()

window_height = root.winfo_reqheight()

screen_width = root.winfo_screenwidth()

screen_height = root.winfo_screenheight()

x_position = max(
    (screen_width - window_width) // 2,
    0
)

y_position = max(
    (screen_height - window_height) // 3,
    0
)

root.geometry(
    f"{window_width}x{window_height}"
    f"+{x_position}+{y_position}"
)
root.minsize(window_width, window_height)
root.maxsize(window_width, window_height)


# ============================================================
# 31) ENTER KEY
# ============================================================

root.bind(
    "<Return>",
    lambda _event:
    run_prediction()
)


# ============================================================
# 32) STARTUP DIAGNOSTIC
# ============================================================

run_model_diagnostic()


# ============================================================
# 33) START GUI
# ============================================================

root.mainloop()
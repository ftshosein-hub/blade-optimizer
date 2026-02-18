# app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import io
import time

# Assuming you have this module
from ga_optimizer import genetic_algorithm

st.set_page_config(
    page_title="Blade Arrangement Optimizer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Title & Header ─────────────────────────────────────────
st.title("Blade Arrangement Optimizer – Genetic Algorithm")
st.markdown("**Input**: Blade label + moment value (one blade per line)")

# ─── Sidebar – GA Settings ──────────────────────────────────
with st.sidebar:
    st.header("Genetic Algorithm Settings")

    pop_size = st.slider(
        "Population size",
        min_value=50, max_value=2000, value=300, step=25,
        help="Number of individuals per generation"
    )

    max_gen = st.slider(
        "Maximum generations",
        min_value=200, max_value=10000, value=1200, step=100,
        help="Maximum number of generations to run"
    )

    mutation_rate = st.slider(
        "Mutation rate",
        min_value=0.01, max_value=0.50, value=0.08, step=0.01,
        format="%.2f"
    )

    st.markdown("---")
    st.caption("Created by: H.Malekmohammadi\nhoseinmm15@gmail.com")

# ─── Main area – Data Input ─────────────────────────────────
tab1, tab2 = st.tabs(["Direct Text Input", "Upload File"])

with tab1:
    default_text = """B1  12.45
B2  -8.70
B3  5.20
B4  -3.10
B5  9.80
# Add more blades...
"""
    raw_text = st.text_area(
        "Enter blade data (Label Moment – space or tab separated)",
        value=default_text,
        height=240,
        key="text_input"
    )

with tab2:
    uploaded_file = st.file_uploader("Text file (.txt)", type=["txt", "text"])
    if uploaded_file is not None:
        raw_text = uploaded_file.read().decode("utf-8")
        st.success("File uploaded successfully")
    else:
        raw_text = ""

# ─── Start Button & Processing ──────────────────────────────
if st.button("Start Optimization", type="primary", use_container_width=True, disabled=not raw_text.strip()):

    # Parse input
    lines = [line.strip() for line in raw_text.splitlines() if line.strip() and not line.strip().startswith('#')]

    labels = []
    moments_list = []
    parse_errors = []

    for i, line in enumerate(lines, 1):
        parts = line.split(maxsplit=1)
        if len(parts) < 2:
            parse_errors.append(f"Line {i}: invalid format → {line}")
            continue
        try:
            moment = float(parts[1].replace(',', '.'))
            labels.append(parts[0].strip())
            moments_list.append(moment)
        except ValueError:
            parse_errors.append(f"Line {i}: invalid moment value → {line}")

    if parse_errors:
        st.warning("Some lines had issues:\n" + "\n".join(parse_errors[:8]))
        if len(parse_errors) > 8:
            st.caption(f"... and {len(parse_errors)-8} more lines")

    if not moments_list:
        st.error("No valid data found.")
        st.stop()

    moments = np.array(moments_list)

    if len(moments) < 3:
        st.error("At least 3 blades are required.")
        st.stop()

    # Calculate initial residual
    n = len(moments)
    angles = 2 * np.pi * np.arange(n) / n
    vec_x = moments * np.cos(angles)
    vec_y = moments * np.sin(angles)
    initial_residual = np.sqrt(np.sum(vec_x)**2 + np.sum(vec_y)**2)

    st.info(f"Number of blades: **{n}**    –    Initial residual: **{initial_residual:.6f}**")

    # ─── Run Genetic Algorithm ──────────────────────────────
    progress_bar = st.progress(0)
    status_text = st.empty()

    def progress_callback(gen, total, best_residual):
        progress_bar.progress(gen / total)
        status_text.text(f"Generation {gen:4d} / {total}   •   Best residual: {best_residual:.6f}")
        time.sleep(0.015)  # smoother progress display

    with st.spinner("Running genetic algorithm... (may take several minutes)"):
        try:
            best_arrangement, best_residual, history = genetic_algorithm(
                moments=moments,
                labels=labels,
                pop_size=pop_size,
                generations=max_gen,
                mutation_rate=mutation_rate,
                progress_callback=progress_callback
            )
        except Exception as e:
            st.error(f"Error during optimization:\n{str(e)}")
            st.stop()

    progress_bar.progress(1.0)
    status_text.success(f"Optimization completed – Final residual: **{best_residual:.6f}**")

    # ─── Show Results ───────────────────────────────────────
    st.subheader("Optimal Arrangement")

    data = []
    for pos, idx in enumerate(best_arrangement, 1):
        data.append({
            "Position": pos,
            "Label": labels[idx],
            "Moment": f"{moments[idx]:.4f}"
        })

    df = pd.DataFrame(data)
    st.dataframe(
        df.style.format({"Moment": "{:.4f}"}),
        use_container_width=True,
        hide_index=True
    )

    # Excel download
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name="Optimized")
    buffer.seek(0)

    st.download_button(
        label="Download Table as Excel",
        data=buffer,
        file_name="Optimized_Blade_Arrangement.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

    # ─── Plots ──────────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Moment Vectors")

        fig_vec = plt.figure(figsize=(6, 5.5), dpi=110)
        ax = fig_vec.add_subplot(111)

        # Before optimization
        x_before = moments * np.cos(angles)
        y_before = moments * np.sin(angles)
        ax.plot(x_before, y_before, '*--', ms=9, alpha=0.7, label='Before optimization')

        # After optimization
        x_after = moments[best_arrangement] * np.cos(angles)
        y_after = moments[best_arrangement] * np.sin(angles)
        ax.plot(x_after, y_after, 'o-', lw=1.8, label='After optimization')

        ax.set_title("Vector Sum of Moments")
        ax.set_xlabel("X component")
        ax.set_ylabel("Y component")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis("equal")
        ax.set_axisbelow(True)

        st.pyplot(fig_vec)

    with col_right:
        st.subheader("Convergence Plot")

        fig_conv = plt.figure(figsize=(6, 5.5), dpi=110)
        ax = fig_conv.add_subplot(111)
        ax.semilogy(history, 'b-', lw=1.4)
        ax.set_title("Residual Convergence over Generations")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Residual (log scale)")
        ax.grid(True, which="both", alpha=0.25)
        ax.yaxis.set_major_formatter(ScalarFormatter())

        st.pyplot(fig_conv)

else:
    st.info("Enter blade data and click 'Start Optimization' to begin.")
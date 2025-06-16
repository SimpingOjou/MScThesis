import plotly.graph_objects as go
import numpy as np

from collections import Counter, namedtuple

PlotStyle = namedtuple("PlotStyle", [
    "scatter_size", "scatter_symbol", "scatter_line_width",
    "legend_font_size", "title_font_size", "axis_font_size", "axis_title_font_size",
    "plot_width", "plot_height", "healthy_key"
])

PLOT_STYLE = PlotStyle(
    scatter_size=16,
    scatter_symbol="circle",
    scatter_line_width=1,
    legend_font_size=18,
    title_font_size=24,
    axis_font_size=16,
    axis_title_font_size=20,
    plot_width=900,
    plot_height=700,
    healthy_key="Pre"
)

## Phase color constants
# STANCE_COLOR = (255, 100, 100)  # Red
# SWING_COLOR = (100, 100, 255)   # Blue
OTHER_COLOR = (150, 150, 150)   # Gray
STANCE_COLOR = (189, 147, 249) # Purple
SWING_COLOR = (139, 233, 253) # Cyan

def majority_vote(labels):
    counts = Counter(labels)
    return counts.most_common(1)[0][0]

def determine_phase(frame, stances, swings):
    if any(start <= frame <= end for start, end in stances):
        return 'stance'
    elif any(start <= frame <= end for start, end in swings):
        return 'swing'
    return 'other'

def get_phase_color(phase, alpha=1):
    """
    Determine phase color and label for a given frame index.
    
    Parameters:
    -----------
    frame_idx : int
        Current frame index
    total_frames : int
        Total number of frames (for alpha calculation)
    stance_start : int
        Start frame of stance phase
    stance_end : int
        End frame of stance phase
    swing_start : int
        Start frame of swing phase
    swing_end : int
        End frame of swing phase
    
    Returns:
    --------
    str : color_string
        RGBA color string and phase label ('stance', 'swing', or 'other')
    """
    # Check if frame is in stance phase
    if phase == 'stance':
        r, g, b = STANCE_COLOR
        return f'rgba({r},{g},{b},{alpha})'
    
    # Check if frame is in swing phase
    if phase == 'swing':
        r, g, b = SWING_COLOR
        return f'rgba({r},{g},{b},{alpha})'
    
    # Default color if not in either phase
    r, g, b = OTHER_COLOR
    return f'rgba({r},{g},{b},{alpha})'

# === Key Extraction ===
def extract_joint_keys(keys):
    x_keys = [key for key in keys if 'x' in key.lower()]
    y_keys = [key for key in keys if 'y' in key.lower()]
    joint_names = [x.split('_x')[0] for x in x_keys]
    return x_keys, y_keys, joint_names

# === Frame Trace Builder ===
def create_frame_trace(frame_data, x_keys, y_keys, joint_names, frame, phase, alpha=1):
    x_positions = [frame_data[x] for x in x_keys]
    y_positions = [frame_data[y] for y in y_keys]
    color = get_phase_color(phase, alpha)

    return go.Scatter(
        x=x_positions,
        y=y_positions,
        mode='lines',
        name=f"Frame {frame} ({phase})",
        line=dict(width=PLOT_STYLE.scatter_line_width, color=color),
        marker=dict(size=0, symbol=PLOT_STYLE.scatter_symbol, color=color),
        text=[f"{joint} - Frame {frame} ({phase})" for joint in joint_names],
        hoverinfo='text',
        showlegend=False
    )

# === Spatial Normalization Helper (One Axis) ===
def get_normalized_feature(df, key):
    values = df[key].values
    return normalize_spatial_curve(values)

def normalize_spatial_curve(values):
    v_min, v_max = np.min(values), np.max(values)
    return (values - v_min) / (v_max - v_min) if v_max > v_min else values * 0

# === Joint Trajectory Segmentation ===
def get_joint_trajectory_segmented_by_phase(step_df, x_key, y_key, stances, swings, total_frames):
    segments = []
    current_phase = None
    xs, ys, alphas = [], [], []

    for i, frame in enumerate(step_df.index):
        x, y = step_df.loc[frame, x_key], step_df.loc[frame, y_key]
        alpha = 1#0.3 + 0.7 * i / total_frames
        phase = determine_phase(frame, stances, swings)

        if phase != current_phase and xs:
            segments.append((xs, ys, current_phase, alpha))
            xs, ys = [], []

        xs.append(x)
        ys.append(y)
        current_phase = phase

    if xs:
        segments.append((xs, ys, current_phase, alpha))

    return segments

def compute_avg_stance_ratio(steps):
    ratios = []
    for step in steps:
        s1, s2 = step['stance']
        sw1, sw2 = step['swing']
        stance = s2 - s1 + 1
        swing = sw2 - sw1 + 1
        total = stance + swing
        if total > 0:
            ratios.append(stance / total)
    return np.mean(ratios) if ratios else 0.6

# === Shared Phase-Aligned Plot Helper ===
def compute_phase_aligned_average(steps, feature_key, n_points):
    curves, phases = [], []

    step_ratio = compute_avg_stance_ratio(steps)  # use group-level ratio

    for step in steps:
        df = step['step']
        if feature_key not in df.columns:
            continue

        values = df[feature_key].to_numpy()
        if len(values) < 2:
            continue

        interp_time = np.linspace(0, 1, n_points)
        interp_values = np.interp(interp_time, np.linspace(0, 1, len(values)), values)

        # Use same stance/swing split across all steps in this group
        interp_phases = ['stance' if t < step_ratio else 'swing' for t in interp_time]

        curves.append(interp_values)
        phases.append(interp_phases)

    if not curves:
        return None, None, None

    curves = np.array(curves)
    phases = np.array(phases)

    mean_vals = np.mean(curves, axis=0)
    sem_vals = np.std(curves, axis=0) / np.sqrt(curves.shape[0])

    interp_time = np.linspace(0, 1, n_points)
    group_phases = ['stance' if t < step_ratio else 'swing' for t in interp_time]

    return mean_vals, sem_vals, group_phases


def plot_phase_aligned_trace(fig, row, col, time_vals, mean_vals, sem_vals, phase_labels, light=False):
    for i in range(len(time_vals) - 1):
        phase = phase_labels[i]
        if phase == 'swing':
            color = '#8be9fd' if not light else 'rgba(139, 233, 253, 0.5)'
            sem_color = 'rgba(139, 233, 253, 0.7)' if not light else 'rgba(139, 233, 253, 0.35)'
        elif phase == 'stance':
            color = '#bd93f9' if not light else 'rgba(189, 147, 249, 0.5)'
            sem_color = 'rgba(189, 147, 249, 0.7)' if not light else 'rgba(189, 147, 249, 0.2)'
        else:
            color = 'lightgray' if light else 'gray'
            sem_color = 'rgba(192,192,192,0.2)'

        fig.add_trace(go.Scatter(
            x=time_vals[i:i+2],
            y=mean_vals[i:i+2],
            mode='lines',
            line=dict(color=color, width=3),
            showlegend=False,
            hoverinfo='skip'
        ), row=row, col=col)

        fig.add_trace(go.Scatter(
            x=np.concatenate([time_vals[i:i+2], time_vals[i:i+2][::-1]]),
            y=np.concatenate([mean_vals[i:i+2] + sem_vals[i:i+2], (mean_vals[i:i+2] - sem_vals[i:i+2])[::-1]]),
            fill='toself',
            fillcolor=sem_color,
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            hoverinfo='skip'
        ), row=row, col=col)
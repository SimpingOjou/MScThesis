import webbrowser
import os
import itertools
import umap
import math

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from scipy.interpolate import CubicSpline
from plotly.subplots import make_subplots
from collections import Counter, defaultdict

def majority_vote(labels):
    counts = Counter(labels)
    return counts.most_common(1)[0][0]

## Plot constants
HEALTHY_KEY = 'Pre'
SCATTER_SIZE = 6
SCATTER_LINE_WIDTH = 1
SCATTER_SYMBOL = 'circle'
LEGEND_FONT_SIZE = 18
TITLE_FONT_SIZE = 24
AXIS_FONT_SIZE = 16
AXIS_TITLE_FONT_SIZE = 20
PLOT_WIDTH = 900
PLOT_HEIGHT = 700

## Phase color constants
STANCE_COLOR = (255, 100, 100)  # Red
SWING_COLOR = (100, 100, 255)   # Blue
OTHER_COLOR = (150, 150, 150)   # Gray

def get_phase_color(phase, alpha=0.7):
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

def determine_phase(frame_idx, stance_start, stance_end, swing_start, swing_end):
    """
    Determine phase for a given frame index.
    
    Parameters:
    -----------
    frame_idx : int
        Current frame index
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
    str : Phase label ('stance', 'swing', or 'other')
    """
    # Check if frame is in stance phase
    if stance_start <= frame_idx <= stance_end:
        return 'stance'
    
    # Check if frame is in swing phase
    if swing_start <= frame_idx <= swing_end:
        return 'swing'
    
    # Default phase if not in either
    return 'other'

def extract_joint_keys(keys):
    """Extract x and y keys and joint names from keys."""
    x_keys = [key for key in keys if 'x' in key.lower()]
    y_keys = [key for key in keys if 'y' in key.lower()]
    joint_names = [x.split('_x')[0] for x in x_keys]
    return x_keys, y_keys, joint_names

def create_frame_trace(frame_data, x_keys, y_keys, joint_names, frame, phase, alpha=0.7):
    """Create a scatter trace for a single frame."""
    x_positions = [frame_data[x] for x in x_keys]
    y_positions = [frame_data[y] for y in y_keys]
    
    color = get_phase_color(phase, alpha)
    
    return go.Scatter(
        x=x_positions,
        y=y_positions,
        mode='lines+markers',
        name=f"Frame {frame} ({phase})",
        line=dict(width=SCATTER_LINE_WIDTH, color=color),
        marker=dict(size=SCATTER_SIZE, symbol=SCATTER_SYMBOL, color=color),
        text=[f"{joint} - Frame {frame} ({phase})" for joint in joint_names],
        hoverinfo='text',
        showlegend=False
    )

def plot_step_trajectory(segmented_steps, keys, limb_name='Hindlimb',
                         joint_trace_keys=('rhindlimb lHindfingers - X pose (m)', 
                                            'rhindlimb lHindfingers - Y pose (m)')):
    """
    Plot pose trajectories with stance and swing phase coloring,
    with a subplot for the selected joint's XY trajectory.

    Parameters:
    -----------
    segmented_steps : list
        List of step dictionaries with step data and metadata
    keys : list
        List of column names for joints (should include x and y coords)
    limb_name : str
        Name of the limb being plotted
    joint_trace_keys : tuple
        Keys for X and Y columns of the joint to show in subplot

    Returns:
    --------
    None
    """
    max_plots = 2
    processed_count = 0

    for step in segmented_steps:
        if processed_count >= max_plots:
            break

        # Step data
        step_df = step["step"]
        mouse = step["mouse"]
        group = step["group"]
        stance_start, stance_end = step["stance"]
        swing_start, swing_end = step["swing"]

        if HEALTHY_KEY not in group:
            continue

        # Extract joint info
        x_keys, y_keys, joint_names = extract_joint_keys(keys)

        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=False,
            row_heights=[0.75, 0.25],
            vertical_spacing=0.08,
            subplot_titles=[
                f"{limb_name} Pose Trajectory - {group} {mouse}", ""
                #f"{joint_trace_keys[1]} vs {joint_trace_keys[0]}"
            ]
        )

        first_frame = step_df.index[0]
        last_frame = step_df.index[-1]
        total_frames = last_frame - first_frame + 1

        # --- Pose plot (top row)
        for frame_idx, frame in enumerate(step_df.index):
            frame_data = step_df.loc[frame]
            phase = determine_phase(frame_idx, stance_start, stance_end, swing_start, swing_end)
            alpha = 0.8#frame_idx / total_frames
            frame_trace = create_frame_trace(frame_data, x_keys, y_keys, joint_names, frame, phase, alpha=alpha)
            fig.add_trace(frame_trace, row=1, col=1)

        # --- Subplot: Joint trajectory with phase coloring
        x_joint_key, y_joint_key = joint_trace_keys
        if x_joint_key in step_df.columns and y_joint_key in step_df.columns:
            xs, ys, alphas = [], [], []
            current_phase = None

            for frame_idx, frame in enumerate(step_df.index):
                x = step_df.loc[frame, x_joint_key]
                y = step_df.loc[frame, y_joint_key]

                phase = determine_phase(frame_idx, stance_start, stance_end, swing_start, swing_end)

                if phase != current_phase and xs:
                    color = get_phase_color(current_phase, alpha=alpha)
                    fig.add_trace(go.Scatter(
                        x=xs,
                        y=ys,
                        mode='lines',
                        # marker=dict(size=SCATTER_SIZE, color=color, symbol=SCATTER_SYMBOL),
                        line=dict(width=1, color=color),
                        showlegend=False,
                        hoverinfo='skip'
                    ), row=2, col=1)
                    xs, ys, alphas = [], [], []

                alpha = frame_idx / total_frames
                xs.append(x)
                ys.append(y)
                alphas.append(alpha)
                current_phase = phase

            # Final flush
            if xs:
                color = get_phase_color(current_phase, alpha=alpha)
                fig.add_trace(go.Scatter(
                    x=xs,
                    y=ys,
                    mode='lines+markers',
                    marker=dict(size=SCATTER_SIZE, color=color, symbol=SCATTER_SYMBOL),
                    line=dict(width=1, color=color),
                    showlegend=False,
                    hoverinfo='skip'
                ), row=2, col=1)
        else:
            print(f"Warning: Joint keys {joint_trace_keys} not found in step data.")

        # Layout
        fig.update_layout(
            width=PLOT_WIDTH,
            height=PLOT_HEIGHT,
            template='plotly_white',
            showlegend=False,
            title_font=dict(size=TITLE_FONT_SIZE),
            xaxis=dict(title="", showgrid=False, title_font=dict(size=AXIS_TITLE_FONT_SIZE)),
            yaxis=dict(title="Y Position (m)", showgrid=False, title_font=dict(size=AXIS_TITLE_FONT_SIZE)),
            xaxis2=dict(title=x_joint_key, showgrid=False, title_font=dict(size=AXIS_TITLE_FONT_SIZE)),
            yaxis2=dict(title=y_joint_key, showgrid=False, title_font=dict(size=AXIS_TITLE_FONT_SIZE)),
        )

        fig.show()
        processed_count += 1

def plot_trajectory(data, keys, limb_name='Hindlimb'):
    """
    Plot pose trajectories with stance and swing phase coloring over the whole run.
    
    Parameters:
    -----------
    data : list
        List of experiment dictionaries containing run data
    keys : list
        List of column names for joints (should contain x and y coordinates)
    limb_name : str, optional (default='Hindlimb')
        Name of the limb being plotted
    
    Returns:
    --------
    None (displays plot)
    """
    max_plots = 2
    processed_count = 0
    
    for exp in data:
        if processed_count >= max_plots:
            break
            
        # Extract step data
        step_df = exp["data"][keys]
        mouse = exp["mouse"]
        group = exp["group"]
        run = exp["run"]
        limb = "hindlimb" if limb_name.lower() == 'hindlimb' else 'forelimb'
        stances = exp["stance"][limb]  # list[tuple]: [(start, end), ...]
        swings = exp["swing"][limb]    # list[tuple]: [(start, end), ...]
        
        if HEALTHY_KEY not in group:
            continue
            
        # Extract joint information
        x_keys, y_keys, joint_names = extract_joint_keys(keys)
        
        # Create figure
        fig = go.Figure()
        
        # Get frame range for alpha calculation
        frames = list(step_df.index)
        total_frames = len(frames)
        
        # Add frame traces
        for frame_idx, frame in enumerate(frames):
            if frame < stances[0][0] or frame > swings[-1][1]:
                continue
            frame_data = step_df.loc[frame]
            
            # Determine phase for this frame
            phase = 'other'  # default

            if any(stance_start <= frame <= stance_end for stance_start, stance_end in stances):
                phase = 'stance'
            elif any(swing_start <= frame <= swing_end for swing_start, swing_end in swings):
                phase = 'swing'
            
            # Calculate alpha based on progression through the run
            alpha = 0.3 + 0.7 * frame_idx / total_frames
            
            frame_trace = create_frame_trace(frame_data, x_keys, y_keys, joint_names, frame, phase, alpha=alpha)
            fig.add_trace(frame_trace)
        
        # Update layout and show
        fig.update_layout(dict(
            title=f"{limb_name} Pose Trajectory - {group} {mouse} {run}",
            xaxis_title="X Position (m)",
            yaxis_title="Y Position (m)",
            width=PLOT_WIDTH*2,
            height=PLOT_HEIGHT,
            showlegend=False,
            title_font=dict(size=TITLE_FONT_SIZE),
            xaxis=dict(title_font=dict(size=AXIS_TITLE_FONT_SIZE), tickfont=dict(size=AXIS_FONT_SIZE), showgrid=False),
            yaxis=dict(title_font=dict(size=AXIS_TITLE_FONT_SIZE), tickfont=dict(size=AXIS_FONT_SIZE), showgrid=False),
            template='plotly_white'
        ))
        fig.show()
        
        processed_count += 1

def plot_animated_trajectory(data, keys, limb_name='Hindlimb'):
    max_plots = 2
    processed_count = 0

    for exp in data:
        if processed_count >= max_plots:
            break

        if HEALTHY_KEY not in exp["group"]:
            continue

        # Extract and prepare data
        step_df = exp["data"][keys]
        mouse = exp["mouse"]
        group = exp["group"]
        run = exp["run"]
        limb = "hindlimb" if limb_name.lower() == 'hindlimb' else 'forelimb'
        stances = exp["stance"][limb]
        swings = exp["swing"][limb]

        x_keys, y_keys, joint_names = extract_joint_keys(keys)
        frames = list(step_df.index)
        total_frames = len(frames)

        # Base figure
        fig = go.Figure()

        # Initial frame
        init_frame = frames[0]
        init_data = step_df.loc[init_frame]
        init_phase = get_phase(init_frame, stances, swings)
        alpha = 0.3 + 0.7 * 0 / total_frames

        frame_trace = create_frame_trace(init_data, x_keys, y_keys, joint_names, init_frame, init_phase, alpha=alpha)
        fig.add_trace(frame_trace)

        # Animation frames
        animation_frames = []
        for i, frame in enumerate(frames):
            if frame < stances[0][0] or frame > swings[-1][1]:
                continue

            frame_data = step_df.loc[frame]
            phase = get_phase(frame, stances, swings)
            alpha = 0.3 + 0.7 * i / total_frames
            trace = create_frame_trace(frame_data, x_keys, y_keys, joint_names, frame, phase, alpha=alpha)

            animation_frames.append(go.Frame(data=[trace], name=str(i)))

        fig.frames = animation_frames

        x_max = step_df[x_keys].max().max()
        # Animation controls
        fig.update_layout(
            title=f"{limb_name} Pose Trajectory - {group} {mouse} {run}",
            xaxis=dict(title="X Position (m)", showgrid=False, range=[0, x_max]),
            yaxis=dict(title="Y Position (m)", showgrid=False),
            width=PLOT_WIDTH * 2,
            height=PLOT_HEIGHT,
            showlegend=False,
            template='plotly_white',
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                buttons=[dict(label="Play", method="animate", args=[None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True}]),
                         dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0}, "mode": "immediate", "transition": {"duration": 0}}])]
            )],
            sliders=[{
                "steps": [{"args": [[str(k)], {"frame": {"duration": 0, "redraw": True},
                                                "mode": "immediate"}],
                           "label": str(k), "method": "animate"} for k in range(len(fig.frames))],
                "transition": {"duration": 0},
                "x": 0, "y": -0.1,
                "currentvalue": {"prefix": "Frame: "}
            }]
        )

        fig.show()
        # Save the figure as an HTML file and open it in the browser
        html_path = f"/tmp/animated_trajectory_{mouse}_{run}.html"
        fig.write_html(html_path)
        webbrowser.open('file://' + os.path.realpath(html_path))
        processed_count += 1

def get_phase(frame, stances, swings):
    if any(start <= frame <= end for start, end in stances):
        return 'stance'
    elif any(start <= frame <= end for start, end in swings):
        return 'swing'
    return 'stance'

def plot_trajectory_with_joint_trace(data, keys, limb_name='Hindlimb',
                                     trace_keys=('rhindlimb lHindfingers - X pose (m)', 
                                                 'rhindlimb lHindfingers - Y pose (m)')):
    """
    Plot full pose trajectory with stance/swing phase coloring and a subplot showing
    a single joint's trajectory (Y vs X) over time.

    Parameters:
    -----------
    data : list
        List of experiment dictionaries containing run data
    keys : list
        List of column names for joints (should contain x and y coordinates)
    limb_name : str, optional
        Name of the limb being plotted
    trace_keys : tuple
        Keys for X and Y columns of the joint to plot below (e.g., for lHindfingers)

    Returns:
    --------
    None (displays plot)
    """
    max_plots = 2
    processed_count = 0

    for exp in data:
        if processed_count >= max_plots:
            break

        if HEALTHY_KEY not in exp["group"]:
            continue

        step_df = exp["data"]
        mouse = exp["mouse"]
        group = exp["group"]
        run = exp["run"]
        limb = "hindlimb" if limb_name.lower() == 'hindlimb' else 'forelimb'
        stances = exp["stance"][limb]
        swings = exp["swing"][limb]

        x_keys, y_keys, joint_names = extract_joint_keys(keys)
        frames = list(step_df.index)
        total_frames = len(frames)

        # Create subplot: 2 rows, shared x-axis not needed
        fig = make_subplots(rows=2, cols=1,
                            shared_xaxes=False,
                            vertical_spacing=0.08,
                            row_heights=[0.75, 0.25],
                            subplot_titles=[
                                f"{limb_name} Pose Trajectory - {group} {mouse} {run}", ""
                                # f"Trajectory of {trace_keys[1]} vs {trace_keys[0]}"
                            ])

        # Main pose trajectory
        for frame_idx, frame in enumerate(frames):
            if frame < stances[0][0] or frame > swings[-1][1]:
                continue

            frame_data = step_df.loc[frame]
            phase = 'other'
            if any(start <= frame <= end for start, end in stances):
                phase = 'stance'
            elif any(start <= frame <= end for start, end in swings):
                phase = 'swing'

            alpha = 0.8
            trace = create_frame_trace(frame_data, x_keys, y_keys, joint_names, frame, phase, alpha=alpha)
            fig.add_trace(trace, row=1, col=1)

        # Joint trajectory subplot
        x_joint_key, y_joint_key = trace_keys
        # Joint trajectory subplot with phase coloring
        if x_joint_key in step_df.columns and y_joint_key in step_df.columns:
            current_phase = None
            xs, ys, alphas = [], [], []

            for frame_idx, frame in enumerate(frames):
                if frame < stances[0][0] or frame > swings[-1][1]:
                    continue

                x = step_df.loc[frame, x_joint_key]
                y = step_df.loc[frame, y_joint_key]

                # Determine phase
                phase = 'other'
                if any(start <= frame <= end for start, end in stances):
                    phase = 'stance'
                elif any(start <= frame <= end for start, end in swings):
                    phase = 'swing'

                # Segment change: flush buffer and start new trace
                if phase != current_phase and xs:
                    color = get_phase_color(current_phase, alpha=0.8)
                    fig.add_trace(go.Scatter(
                        x=xs,
                        y=ys,
                        mode='lines+markers',
                        marker=dict(size=SCATTER_SIZE, color=color, symbol=SCATTER_SYMBOL),
                        line=dict(width=1, color=color),
                        showlegend=False,
                        hoverinfo='skip'
                    ), row=2, col=1)
                    xs, ys, alphas = [], [], []

                alpha = 0.3 + 0.7 * frame_idx / total_frames
                xs.append(x)
                ys.append(y)
                alphas.append(alpha)
                current_phase = phase

            # Final flush
            if xs:
                color = get_phase_color(current_phase, alpha=0.8)
                fig.add_trace(go.Scatter(
                    x=xs,
                    y=ys,
                    mode='lines+markers',
                    marker=dict(size=SCATTER_SIZE, color=color, symbol=SCATTER_SYMBOL),
                    line=dict(width=1, color=color),
                    showlegend=False,
                    hoverinfo='skip'
                ), row=2, col=1)
        else:
            print(f"Warning: One or both joint keys not found: {trace_keys}")

        # Layout
        fig.update_layout(
            width=PLOT_WIDTH * 2,
            height=int(PLOT_HEIGHT * 1.2),
            showlegend=False,
            template='plotly_white',
            title_font=dict(size=TITLE_FONT_SIZE),
            xaxis=dict(title="", showgrid=False, title_font=dict(size=AXIS_TITLE_FONT_SIZE)),
            yaxis=dict(title="", showgrid=False, title_font=dict(size=AXIS_TITLE_FONT_SIZE)),
            xaxis2=dict(title=x_joint_key, showgrid=False, title_font=dict(size=AXIS_TITLE_FONT_SIZE)),
            yaxis2=dict(title=y_joint_key, showgrid=False, title_font=dict(size=AXIS_TITLE_FONT_SIZE)),
        )

        fig.show()
        processed_count += 1

def plot_trajectory_with_joint_traces(data, keys_1, keys_2, limb_name='Hindlimb'):
    """
    Plot full pose trajectory with stance/swing phase coloring and a subplot showing
    a single joint's trajectory (Y vs X) over time.

    Parameters:
    -----------
    data : list
        List of experiment dictionaries containing run data
    keys : list
        List of column names for joints (should contain x and y coordinates)
    limb_name : str, optional
        Name of the limb being plotted

    Returns:
    --------
    None (displays plot)
    """
    max_plots = 2
    processed_count = 0

    for exp in data:
        if processed_count >= max_plots:
            break

        if HEALTHY_KEY not in exp["group"]:
            continue

        step_df = exp["data"]
        mouse = exp["mouse"]
        group = exp["group"]
        run = exp["run"]
        limb = "hindlimb" if limb_name.lower() == 'hindlimb' else 'forelimb'
        stances = exp["stance"][limb]
        swings = exp["swing"][limb]

        x_keys, y_keys, joint_names = extract_joint_keys(keys_1)
        x_joint_keys, y_joint_keys, joint_names_2 = extract_joint_keys(keys_2)
        frames = list(step_df.index)
        total_frames = len(frames)

        # Create subplot: 2 rows, shared x-axis not needed
        fig = make_subplots(rows=2, cols=1,
                            shared_xaxes=False,
                            vertical_spacing=0.08,
                            row_heights=[0.5, 0.5],
                            subplot_titles=[
                                f"Pose Trajectory - {group} {mouse} {run}", ""
                                # f"Trajectory of {trace_keys[1]} vs {trace_keys[0]}"
                            ])

        # Main pose trajectory
        for frame_idx, frame in enumerate(frames):
            if frame < stances[0][0] or frame > swings[-1][1]:
                continue

            frame_data = step_df.loc[frame]
            phase = 'other'
            if any(start <= frame <= end for start, end in stances):
                phase = 'stance'
            elif any(start <= frame <= end for start, end in swings):
                phase = 'swing'

            alpha = 0.8
            trace = create_frame_trace(frame_data, x_keys, y_keys, joint_names, frame, phase, alpha=alpha)
            fig.add_trace(trace, row=1, col=1)

        # Second pose trajectory
        for frame_idx, frame in enumerate(frames):
            if frame < stances[0][0] or frame > swings[-1][1]:
                continue

            frame_data = step_df.loc[frame]
            phase = 'other'
            if any(start <= frame <= end for start, end in stances):
                phase = 'stance'
            elif any(start <= frame <= end for start, end in swings):
                phase = 'swing'

            alpha = 0.8
            trace = create_frame_trace(frame_data, x_joint_keys, y_joint_keys, joint_names_2, frame, phase, alpha=alpha)
            fig.add_trace(trace, row=2, col=1)

        # Layout
        fig.update_layout(
            width=PLOT_WIDTH * 2,
            height=int(PLOT_HEIGHT * 1.2),
            showlegend=False,
            template='plotly_white',
            title_font=dict(size=TITLE_FONT_SIZE),
            xaxis=dict(title="", showgrid=False, title_font=dict(size=AXIS_TITLE_FONT_SIZE)),
            yaxis=dict(title="", showgrid=False, title_font=dict(size=AXIS_TITLE_FONT_SIZE)),
            xaxis2=dict(title="", showgrid=False, title_font=dict(size=AXIS_TITLE_FONT_SIZE)),
            yaxis2=dict(title="", showgrid=False, title_font=dict(size=AXIS_TITLE_FONT_SIZE)),
        )

        fig.show()
        processed_count += 1

def compare_step_features_in_batches(step_dict, batch_size=25, output_dir="./figures/SCI"):
    """
    Compare all feature pairs (scatter plots) within a single step run.

    Parameters:
    -----------
    step_dict : dict
        Single step dictionary with 'step': DataFrame and metadata.
    batch_size : int
        Number of feature pair plots per HTML file.
    output_dir : str
        Directory where HTML files will be saved.
    """
    os.makedirs(os.path.abspath(output_dir), exist_ok=True)

    step_df = step_dict["step"]
    mouse = step_dict.get("mouse", "UnknownMouse")
    run = step_dict.get("run", "Run1")
    stance_start, stance_end = step_dict.get("stance", (0, -1))
    swing_start, swing_end = step_dict.get("swing", (0, -1))

    # Select features
    features = [col for col in step_df.columns if "pose" in col or "CoM" in col]
    feature_pairs = list(itertools.combinations(features, 2))

    print(f"Selected {len(features)} features → {len(feature_pairs)} pairs")

    n_batches = (len(feature_pairs) + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(feature_pairs))
        batch_pairs = feature_pairs[start:end]

        fig = make_subplots(
            rows=len(batch_pairs), cols=1,
            vertical_spacing=0.02,
            subplot_titles=[f"{x} vs {y}" for x, y in batch_pairs]
        )

        total_frames = len(step_df)

        for i, (x_feat, y_feat) in enumerate(batch_pairs):
            xs_stance, ys_stance = [], []
            xs_swing, ys_swing = [], []
            xs_other, ys_other = [], []

            for frame_idx, frame in enumerate(step_df.index):
                x = step_df.loc[frame, x_feat]
                y = step_df.loc[frame, y_feat]

                if stance_start <= frame_idx <= stance_end:
                    xs_stance.append(x)
                    ys_stance.append(y)
                elif swing_start <= frame_idx <= swing_end:
                    xs_swing.append(x)
                    ys_swing.append(y)
                else:
                    xs_other.append(x)
                    ys_other.append(y)

            # Add traces for each phase
            if xs_stance:
                fig.add_trace(go.Scatter(
                    x=xs_stance,
                    y=ys_stance,
                    mode='markers',
                    marker=dict(size=5, color=get_phase_color('stance', alpha=0.8)),
                    name="Stance",
                    showlegend=(i == 0),
                    hoverinfo='x+y'
                ), row=i + 1, col=1)

            if xs_swing:
                fig.add_trace(go.Scatter(
                    x=xs_swing,
                    y=ys_swing,
                    mode='markers',
                    marker=dict(size=5, color=get_phase_color('swing', alpha=0.8)),
                    name="Swing",
                    showlegend=(i == 0),
                    hoverinfo='x+y'
                ), row=i + 1, col=1)

            if xs_other:
                fig.add_trace(go.Scatter(
                    x=xs_other,
                    y=ys_other,
                    mode='markers',
                    marker=dict(size=5, color=get_phase_color('other', alpha=0.6)),
                    name="Other",
                    showlegend=(i == 0),
                    hoverinfo='x+y'
                ), row=i + 1, col=1)

        fig.update_layout(
            height=250 * len(batch_pairs),
            width=1200,
            title=f"Feature Pair Comparison with Phase Colors: {mouse} - {run} (Batch {batch_idx + 1})",
            template="plotly_white",
            legend=dict(font=dict(size=10)),
            margin=dict(t=40, l=40, r=20, b=20)
        )

        file_path = os.path.join(output_dir, f"{mouse}_{run}_feature_pairs_batch_{batch_idx + 1}.html")
        fig.write_html(file_path)
        print(f"Saved: {file_path}")

        if batch_idx == 0:
            webbrowser.open('file://' + os.path.realpath(file_path))

def plot_umap_from_step(step_dict):
    """
    Run UMAP on time-series feature vectors from one step.

    Parameters:
    -----------
    step_dict : dict
        Single step with 'step': DataFrame and 'stance'/'swing' keys.
    """
    df = step_dict['step']
    stance_start, stance_end = step_dict['stance']
    swing_start, swing_end = step_dict['swing']

    # Select features
    features = [col for col in df.columns if 'pose' in col or 'CoM' in col]
    X = df[features].values  # shape: (n_frames, n_features)

    # UMAP embedding
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42, n_components=3)
    X_embedded = reducer.fit_transform(X)  # shape: (n_frames, 2)

    # Build plot
    frame_indices = df.index.tolist()
    phase_labels = []
    for frame in frame_indices:
        if stance_start <= frame <= stance_end:
            phase_labels.append('stance')
        elif swing_start <= frame <= swing_end:
            phase_labels.append('swing')
        else:
            phase_labels.append('other')

    # Group coordinates by phase
    coords = {'stance': [], 'swing': [], 'other': []}
    for i, label in enumerate(phase_labels):
        coords[label].append(X_embedded[i])

    fig = go.Figure()

    color_map = {
        'stance': 'red',
        'swing': 'blue',
        'other': 'gray'
    }

    for phase, color in color_map.items():
        if coords[phase]:
            pts = np.array(coords[phase])
            fig.add_trace(go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode='markers+lines',
                marker=dict(size=4, color=color, opacity=0.7),
                name=phase
            ))

    fig.update_layout(
        title="3D UMAP Projection of Full-Body Time Series",
        scene=dict(
            xaxis_title="UMAP-1",
            yaxis_title="UMAP-2",
            zaxis_title="UMAP-3"
        ),
        width=900,
        height=800,
        template="plotly_white"
    )
    fig.show()

def plot_umap_all_steps(segmented_steps, n_neighbors=15, min_dist=0.1):
    """
    Perform UMAP on all step frames and plot each step trajectory in 3D,
    colored by gait phase and using opacity to indicate time progression.

    Parameters:
    -----------
    segmented_steps : list of dicts
        Each dict should contain: 'step' (DataFrame), 'stance', 'swing', etc.
    feature_filter : str
        Substring to select columns (e.g., 'pose', 'CoM').
    n_neighbors : int
        UMAP local neighborhood size.
    min_dist : float
        UMAP minimum distance parameter.
    """

    all_features = []
    metadata = []

    for step_id, step_data in enumerate(segmented_steps):
        df = step_data["step"]
        stance_start, stance_end = step_data["stance"]
        swing_start, swing_end = step_data["swing"]

        # Select relevant features
        features = [col for col in df.columns if 'pose' in col or 'CoM' in col]
        if not features:
            continue

        X = df[features].values  # shape: (n_frames, n_features)
        n_frames = X.shape[0]
        frame_indices = df.index.to_numpy()

        # Phase labels and normalized time
        for i in range(n_frames):
            idx = frame_indices[i]
            if stance_start <= idx <= stance_end:
                phase = 'stance'
            elif swing_start <= idx <= swing_end:
                phase = 'swing'
            else:
                phase = 'other'

            all_features.append(X[i])
            metadata.append({
                'step_id': step_id,
                'frame': idx,
                'phase': phase,
                'time_norm': i / (n_frames - 1),
                'mouse': step_data.get('mouse', 'unknown'),
                'run': step_data.get('run', 'unknown')
            })

    if not all_features:
        print("No usable step data.")
        return

    # UMAP
    all_features = np.array(all_features)
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, random_state=42)
    embedding = reducer.fit_transform(all_features)  # shape: (n_total_frames, 3)

    # Group points by step_id
    grouped = defaultdict(list)
    for i, meta in enumerate(metadata):
        key = meta['step_id']
        grouped[key].append((embedding[i], meta))

    # Phase to color
    phase_color = {
        'swing': 'blue',
        'stance': 'red',
        'other': 'gray'
    }

    # Plot
    fig = go.Figure()

    for step_id, points in grouped.items():
        points.sort(key=lambda p: p[1]['time_norm'])  # sort by time
        coords = np.array([p[0] for p in points])
        phases = [p[1]['phase'] for p in points]
        opacities = [0.3 + 0.7 * p[1]['time_norm'] for p in points]  # 0.3 to 1.0

        # Add line segments per point (colored by phase, dynamic alpha)
        for i in range(len(coords) - 1):
            p1, p2 = coords[i], coords[i + 1]
            color = phase_color.get(phases[i], 'gray')
            alpha = opacities[i]
            fig.add_trace(go.Scatter(
                x=[p1[0], p2[0]],
                y=[p1[1], p2[1]],
                mode='markers+lines',
                line=dict(color=color, width=4),
                opacity=alpha,
                showlegend=False,
                hoverinfo='skip'
            ))

    fig.update_layout(
        title="2D UMAP Trajectories Across Steps",
        xaxis_title="UMAP-1",
        yaxis_title="UMAP-2",
        width=900,
        height=800,
        template="plotly_white"
    )
    fig.show()

def compare_phase_aligned_average_single(healthy_steps, unhealthy_steps, feature_keys, n_points=100):
    """
    Plot normalized average trajectories with SEM for angle or CoM features,
    comparing healthy and unhealthy groups. Each subplot shows healthy vs unhealthy
    colored by majority phase (swing = blue, stance = red), with feature types in rows
    and body parts in columns.

    Parameters:
    -----------
    healthy_steps : list of dicts
    unhealthy_steps : list of dicts
        Each dict must have keys: 'step' (DataFrame), 'stance', 'swing'.
    feature_keys : list of str
        Column names to align and compare.
    n_points : int
        Number of points to resample each step to.
    """


    pose_keys = [k for k in feature_keys if ('Angle -' in k or 'CoM' in k) and 'velocity' not in k and 'acceleration' not in k]
    vel_keys = [k for k in feature_keys if 'velocity' in k and 'acceleration' not in k]
    acc_keys = [k for k in feature_keys if 'acceleration' in k]

    grouped = list(zip(pose_keys, vel_keys, acc_keys))
    n_cols = len(grouped)
    n_rows = 3

    fig = make_subplots(rows=n_rows, cols=n_cols, shared_xaxes=True,
                        horizontal_spacing=0.03, vertical_spacing=0.08)

    time_vals = np.linspace(0, 100, n_points)

    def process_group(steps, feature_key):
        all_curves, all_phases = [], []
        for step in steps:
            df = step['step']
            if feature_key not in df.columns:
                continue
            stance_start, stance_end = step['stance']
            swing_start, swing_end = step['swing']
            frames = df.index.to_numpy()
            values = df[feature_key].to_numpy()

            phase_labels = np.full(len(df), 'other', dtype=object)
            for i, idx in enumerate(frames):
                if stance_start <= idx <= stance_end:
                    phase_labels[i] = 'stance'
                elif swing_start <= idx <= swing_end:
                    phase_labels[i] = 'swing'

            norm_time = np.linspace(0, 1, len(values))
            interp_time = np.linspace(0, 1, n_points)
            interp_values = np.interp(interp_time, norm_time, values)

            interp_indices = np.round(np.linspace(0, len(values) - 1, n_points)).astype(int)
            interp_phases = phase_labels[interp_indices]

            all_curves.append(interp_values)
            all_phases.append(interp_phases)
        return np.array(all_curves), np.array(all_phases)

    for col_idx, (pose_key, vel_key, acc_key) in enumerate(grouped):
        for row_idx, feature_key in enumerate([pose_key, vel_key, acc_key], start=1):
            healthy_curves, healthy_phases = process_group(healthy_steps, feature_key)
            unhealthy_curves, unhealthy_phases = process_group(unhealthy_steps, feature_key)

            if healthy_curves.size == 0 or unhealthy_curves.size == 0:
                print(f"No valid steps found for feature: {feature_key}")
                continue

            def plot_group(curves, phases, light=False):
                mean_vals = np.mean(curves, axis=0)
                sem_vals = np.std(curves, axis=0) / np.sqrt(curves.shape[0])
                majority_phase = np.array([majority_vote(phases[:, i]) for i in range(n_points)])

                for i in range(n_points - 1):
                    phase = majority_phase[i]
                    if phase == 'swing':
                        color = 'lightblue' if light else 'blue'
                        sem_color = 'rgba(173,216,230,0.5)' if light else 'rgba(0,0,255,0.2)'
                    elif phase == 'stance':
                        color = 'lightcoral' if light else 'red'
                        sem_color = 'rgba(240,128,128,0.2)' if light else 'rgba(255,0,0,0.2)'
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
                    ), row=row_idx, col=col_idx + 1)

                    fig.add_trace(go.Scatter(
                        x=np.concatenate([time_vals[i:i+2], time_vals[i:i+2][::-1]]),
                        y=np.concatenate([mean_vals[i:i+2] + sem_vals[i:i+2], (mean_vals[i:i+2] - sem_vals[i:i+2])[::-1]]),
                        fill='toself',
                        fillcolor=sem_color,
                        line=dict(color='rgba(255,255,255,0)'),
                        showlegend=False,
                        hoverinfo='skip'
                    ), row=row_idx, col=col_idx + 1)

            plot_group(healthy_curves, healthy_phases, light=False)
            plot_group(unhealthy_curves, unhealthy_phases, light=True)

    fig.update_layout(
        height=350 * n_rows,
        width=300 * n_cols,
        title="Phase-Aligned Healthy vs Unhealthy Comparison",
        template='plotly_white'
    )
    fig.show()

def compare_phase_aligned_average_xy(healthy_steps, unhealthy_steps, feature_keys, n_points=100):
    """
    Plot normalized average trajectories with SEM for multiple features,
    comparing healthy and unhealthy groups. Colored by majority phase
    (swing = blue, stance = red), displayed in a 2-row subplot layout
    where 'X' features go on the top and 'Y' features go on the bottom.

    Parameters:
    -----------
    healthy_steps : list of dicts
    unhealthy_steps : list of dicts
        Each dict must have keys: 'step' (DataFrame), 'stance', 'swing'.
    feature_keys : list of str
        Column names to align and compare.
    n_points : int
        Number of points to resample each step to.
    """

    x_keys = [k for k in feature_keys if 'X' in k]
    y_keys = [k for k in feature_keys if 'Y' in k]
    max_len = max(len(x_keys), len(y_keys))

    fig = make_subplots(rows=2, cols=max_len, shared_yaxes=False,
                        subplot_titles=x_keys + y_keys, horizontal_spacing=0.05)

    time_vals = np.linspace(0, 100, n_points)

    def process_group(steps, feature_key):
        all_curves, all_phases = [], []
        for step in steps:
            df = step['step']
            if feature_key not in df.columns:
                continue
            stance_start, stance_end = step['stance']
            swing_start, swing_end = step['swing']
            frames = df.index.to_numpy()
            values = df[feature_key].to_numpy()

            phase_labels = np.full(len(df), 'other', dtype=object)
            for i, idx in enumerate(frames):
                if stance_start <= idx <= stance_end:
                    phase_labels[i] = 'stance'
                elif swing_start <= idx <= swing_end:
                    phase_labels[i] = 'swing'

            norm_time = np.linspace(0, 1, len(values))
            interp_time = np.linspace(0, 1, n_points)
            interp_values = np.interp(interp_time, norm_time, values)

            interp_indices = np.round(np.linspace(0, len(values) - 1, n_points)).astype(int)
            interp_phases = phase_labels[interp_indices]

            all_curves.append(interp_values)
            all_phases.append(interp_phases)
        return np.array(all_curves), np.array(all_phases)

    for col_idx in range(max_len):
        for row_idx, side_keys in zip([1, 2], [x_keys, y_keys]):
            if col_idx >= len(side_keys):
                continue
            feature_key = side_keys[col_idx]

            healthy_curves, healthy_phases = process_group(healthy_steps, feature_key)
            unhealthy_curves, unhealthy_phases = process_group(unhealthy_steps, feature_key)

            if healthy_curves.size == 0 or unhealthy_curves.size == 0:
                print(f"No valid steps found for feature: {feature_key}")
                continue

            def plot_group(curves, phases, light=False):
                mean_vals = np.mean(curves, axis=0)
                sem_vals = np.std(curves, axis=0) / np.sqrt(curves.shape[0])
                majority_phase = np.array([majority_vote(phases[:, i]) for i in range(n_points)])

                for i in range(n_points - 1):
                    phase = majority_phase[i]
                    if phase == 'swing':
                        color = 'lightblue' if light else 'blue'
                        sem_color = 'rgba(173,216,230,0.5)' if light else 'rgba(0,0,255,0.2)'
                    elif phase == 'stance':
                        color = 'lightcoral' if light else 'red'
                        sem_color = 'rgba(240,128,128,0.2)' if light else 'rgba(255,0,0,0.2)'
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
                    ), row=row_idx, col=col_idx + 1)

                    fig.add_trace(go.Scatter(
                        x=np.concatenate([time_vals[i:i+2], time_vals[i:i+2][::-1]]),
                        y=np.concatenate([mean_vals[i:i+2] + sem_vals[i:i+2], (mean_vals[i:i+2] - sem_vals[i:i+2])[::-1]]),
                        fill='toself',
                        fillcolor=sem_color,
                        line=dict(color='rgba(255,255,255,0)'),
                        showlegend=False,
                        hoverinfo='skip'
                    ), row=row_idx, col=col_idx + 1)

            plot_group(healthy_curves, healthy_phases, light=False)
            plot_group(unhealthy_curves, unhealthy_phases, light=True)

    fig.update_layout(
        height=700,
        width=350 * max_len,
        title="Phase-Aligned Healthy vs Unhealthy Comparison (X/Y)",
        template='plotly_white'
    )
    fig.show()


def plot_phase_aligned_average_single(segmented_steps, feature_keys, n_points=100):
    """
    Plot normalized average trajectories with SEM for angle or CoM features,
    colored by majority phase (swing = blue, stance = red), each feature in its own row.

    Parameters:
    -----------
    segmented_steps : list of dicts
        Each dict must have keys: 'step' (DataFrame), 'stance', 'swing'.
    feature_keys : list of str
        Column names to align and compare.
    n_points : int
        Number of points to resample each step to.
    """
    # Classify by type
    pose_keys = [k for k in feature_keys if ('Angle -' in k or 'CoM' in k) and 'velocity' not in k and 'acceleration' not in k]
    vel_keys = [k for k in feature_keys if 'velocity' in k and 'acceleration' not in k]
    acc_keys = [k for k in feature_keys if 'acceleration' in k]

    # Use keys to infer common body parts
    grouped = list(zip(pose_keys, vel_keys, acc_keys))
    n_cols = len(grouped)
    n_rows = 3  # pose, velocity, acceleration

    fig = make_subplots(rows=n_rows, cols=n_cols, shared_xaxes=True,
                        subplot_titles=pose_keys + vel_keys + acc_keys,
                        horizontal_spacing=0.03, vertical_spacing=0.08)

    time_vals = np.linspace(0, 100, n_points)

    for col_idx, (pose_key, vel_key, acc_key) in enumerate(grouped):
        for row_idx, feature_key in enumerate([pose_key, vel_key, acc_key], start=1):
            all_curves = []
            all_phases = []

            for step in segmented_steps:
                df = step['step']
                if feature_key not in df.columns:
                    continue

                stance_start, stance_end = step['stance']
                swing_start, swing_end = step['swing']
                frames = df.index.to_numpy()
                values = df[feature_key].to_numpy()

                phase_labels = np.full(len(df), 'other', dtype=object)
                for i, idx in enumerate(frames):
                    if stance_start <= idx <= stance_end:
                        phase_labels[i] = 'stance'
                    elif swing_start <= idx <= swing_end:
                        phase_labels[i] = 'swing'

                norm_time = np.linspace(0, 1, len(values))
                interp_time = np.linspace(0, 1, n_points)
                interp_values = np.interp(interp_time, norm_time, values)

                interp_indices = np.round(np.linspace(0, len(values) - 1, n_points)).astype(int)
                interp_phases = phase_labels[interp_indices]

                all_curves.append(interp_values)
                all_phases.append(interp_phases)

            if len(all_curves) == 0:
                print(f"No valid steps found for feature: {feature_key}")
                continue

            all_curves = np.array(all_curves)
            all_phases = np.array(all_phases)

            mean_vals = np.mean(all_curves, axis=0)
            sem_vals = np.std(all_curves, axis=0) / np.sqrt(all_curves.shape[0])

            majority_phase = np.array([majority_vote(all_phases[:, i]) for i in range(n_points)])

            for i in range(n_points - 1):
                phase = majority_phase[i]
                color = 'blue' if phase == 'swing' else 'red' if phase == 'stance' else 'gray'
                fig.add_trace(go.Scatter(
                    x=time_vals[i:i+2],
                    y=mean_vals[i:i+2],
                    mode='lines',
                    line=dict(color=color, width=3),
                    showlegend=False,
                    hoverinfo='skip'
                ), row=row_idx, col=col_idx + 1)

                fig.add_trace(go.Scatter(
                    x=np.concatenate([time_vals[i:i+2], time_vals[i:i+2][::-1]]),
                    y=np.concatenate([mean_vals[i:i+2] + sem_vals[i:i+2], (mean_vals[i:i+2] - sem_vals[i:i+2])[::-1]]),
                    fill='toself',
                    fillcolor='rgba(0,0,255,0.2)' if phase == 'swing' else 'rgba(255,0,0,0.2)' if phase == 'stance' else 'rgba(128,128,128,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    showlegend=False,
                    hoverinfo='skip'
                ), row=row_idx, col=col_idx + 1)

    fig.update_layout(
        height=350 * n_rows,
        width=300 * n_cols,
        title="Phase-Aligned Averages by Kinematic Type with SEM",
        template='plotly_white'
    )
    fig.show()


def plot_phase_aligned_average_xy(segmented_steps, feature_keys, n_points=100):
    """
    Plot normalized average trajectories with SEM for multiple features,
    colored by majority phase (swing = blue, stance = red) across steps,
    displayed in a 2-row subplot layout where 'X' features go on the top
    and 'Y' features go on the bottom.

    Parameters:
    -----------
    segmented_steps : list of dicts
        Each dict must have keys: 'step' (DataFrame), 'stance', 'swing'.
    feature_keys : list of str
        Column names to align and compare.
    n_points : int
        Number of points to resample each step to.
    """
    x_keys = [k for k in feature_keys if 'X' in k]
    y_keys = [k for k in feature_keys if 'Y' in k]
    max_len = max(len(x_keys), len(y_keys))

    fig = make_subplots(rows=2, cols=max_len, shared_yaxes=False,
                        subplot_titles=x_keys + y_keys, horizontal_spacing=0.05)

    time_vals = np.linspace(0, 100, n_points)

    for col_idx in range(max_len):
        for row_idx, side_keys in zip([1, 2], [x_keys, y_keys]):
            if col_idx >= len(side_keys):
                continue
            feature_key = side_keys[col_idx]

            all_curves = []
            all_phases = []

            for step in segmented_steps:
                df = step['step']
                if feature_key not in df.columns:
                    continue

                stance_start, stance_end = step['stance']
                swing_start, swing_end = step['swing']
                frames = df.index.to_numpy()
                values = df[feature_key].to_numpy()

                phase_labels = np.full(len(df), 'other', dtype=object)
                for i, idx in enumerate(frames):
                    if stance_start <= idx <= stance_end:
                        phase_labels[i] = 'stance'
                    elif swing_start <= idx <= swing_end:
                        phase_labels[i] = 'swing'

                norm_time = np.linspace(0, 1, len(values))
                interp_time = np.linspace(0, 1, n_points)
                interp_values = np.interp(interp_time, norm_time, values)

                interp_indices = np.round(np.linspace(0, len(values) - 1, n_points)).astype(int)
                interp_phases = phase_labels[interp_indices]

                all_curves.append(interp_values)
                all_phases.append(interp_phases)

            if len(all_curves) == 0:
                print(f"No valid steps found for feature: {feature_key}")
                continue

            all_curves = np.array(all_curves)
            all_phases = np.array(all_phases)

            mean_vals = np.mean(all_curves, axis=0)
            sem_vals = np.std(all_curves, axis=0) / np.sqrt(all_curves.shape[0])

            majority_phase = np.array([majority_vote(all_phases[:, i]) for i in range(n_points)])

            for i in range(n_points - 1):
                phase = majority_phase[i]
                color = 'blue' if phase == 'swing' else 'red' if phase == 'stance' else 'gray'
                fig.add_trace(go.Scatter(
                    x=time_vals[i:i+2],
                    y=mean_vals[i:i+2],
                    mode='lines',
                    line=dict(color=color, width=3),
                    showlegend=False,
                    hoverinfo='skip'
                ), row=row_idx, col=col_idx + 1)

                fig.add_trace(go.Scatter(
                    x=np.concatenate([time_vals[i:i+2], time_vals[i:i+2][::-1]]),
                    y=np.concatenate([mean_vals[i:i+2] + sem_vals[i:i+2], (mean_vals[i:i+2] - sem_vals[i:i+2])[::-1]]),
                    fill='toself',
                    fillcolor='rgba(0,0,255,0.2)' if phase == 'swing' else 'rgba(255,0,0,0.2)' if phase == 'stance' else 'rgba(128,128,128,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    showlegend=False,
                    hoverinfo='skip'
                ), row=row_idx, col=col_idx + 1)

    fig.update_layout(
        height=700,
        width=350 * max_len,
        title="Phase-Aligned Averages with SEM",
        template='plotly_white'
    )
    fig.show()

def plot_mean_spatial_trajectory(segmented_steps,
                                  x_key='rhindlimb lHindfingers - X pose (m)',
                                  y_key='rhindlimb lHindfingers - Y pose (m)',
                                  n_points=100):
    """
    Compute and plot the mean spatial trajectory (Y vs X) across steps,
    colored by phase (stance/swing/other) over normalized time.
    """
    aligned_x = []
    aligned_y = []
    aligned_phase = []

    for step in segmented_steps:
        df = step["step"]
        if x_key not in df.columns or y_key not in df.columns:
            continue

        stance_start, stance_end = step["stance"]
        swing_start, swing_end = step["swing"]
        frames = list(df.index)
        L = len(frames)
        norm_time = np.linspace(0, 1, L)

        # Phase labels per frame
        phase_seq = []
        for i in range(L):
            idx = frames[i]
            if stance_start <= idx <= stance_end:
                phase_seq.append("stance")
            elif swing_start <= idx <= swing_end:
                phase_seq.append("swing")
            else:
                phase_seq.append("other")

        x = np.interp(np.linspace(0, 1, n_points), norm_time, df[x_key].values)
        y = np.interp(np.linspace(0, 1, n_points), norm_time, df[y_key].values)
        phase = np.array(phase_seq)[np.round(np.linspace(0, L-1, n_points)).astype(int)]

        aligned_x.append(x)
        aligned_y.append(y)
        aligned_phase.append(phase)

    aligned_x = np.array(aligned_x)
    aligned_y = np.array(aligned_y)
    aligned_phase = np.array(aligned_phase)

    mean_x = np.mean(aligned_x, axis=0)
    mean_y = np.mean(aligned_y, axis=0)

    # Majority phase per point
    phase_labels = []
    for i in range(n_points):
        phases = aligned_phase[:, i]
        vals, counts = np.unique(phases, return_counts=True)
        majority = vals[np.argmax(counts)]
        phase_labels.append(majority)

    # Split trajectory by phase segments
    fig = go.Figure()
    color_map = {'stance': 'red', 'swing': 'blue', 'other': 'gray'}

    current_phase = phase_labels[0]
    xs, ys = [], []

    for i in range(n_points):
        if phase_labels[i] != current_phase and xs:
            fig.add_trace(go.Scatter(
                x=xs,
                y=ys,
                mode='lines+markers',
                line=dict(color=color_map[current_phase]),
                marker=dict(size=5, color=color_map[current_phase]),
                name=current_phase,
                showlegend=False
            ))
            xs, ys = [], []
        xs.append(mean_x[i])
        ys.append(mean_y[i])
        current_phase = phase_labels[i]

    if xs:
        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode='lines+markers',
            line=dict(color=color_map[current_phase]),
            marker=dict(size=5, color=color_map[current_phase]),
            name=current_phase,
            showlegend=False
        ))

    fig.update_layout(
        title="Mean Spatial Trajectory (Y vs X) with Phase Coloring",
        xaxis_title="X Position (m)",
        yaxis_title="Y Position (m)",
        width=800,
        height=600,
        template="plotly_white"
    )
    fig.show()
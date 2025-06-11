import webbrowser
import os
import itertools
import umap
from scipy.interpolate import CubicSpline

import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

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

def plot_xyt_trajectory(data, keys, limb_name='Hindlimb'):
    """
    Plot pose x-y-t trajectories with stance and swing phase coloring over the whole run.
    
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
            
            x_positions = [frame_data[x] for x in x_keys]
            y_positions = [frame_data[y] for y in y_keys]
            
            color = get_phase_color(phase, alpha)
            
            fig.add_trace( go.Scatter3d(
                x=x_positions,
                y=y_positions,
                z=frames,
                mode='lines+markers',
                name=f"Frame {frame} ({phase})",
                line=dict(width=SCATTER_LINE_WIDTH, color=color),
                marker=dict(size=SCATTER_SIZE, symbol=SCATTER_SYMBOL, color=color),
                text=[f"{joint} - Frame {frame} ({phase})" for joint in joint_names],
                hoverinfo='text',
                showlegend=False
            ))
        
        # Update layout and show
        fig.update_layout(dict(
            title=f"{limb_name} Pose Trajectory - {group} {mouse} {run}",
            xaxis_title="X Position (m)",
            yaxis_title="Y Position (m)",
            width=PLOT_WIDTH,
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
                mode='markers',
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

def plot_phase_aligned_average(segmented_steps, feature_key, n_points=100):
    """
    Plot normalized average trajectory with SEM for a single feature.

    Parameters:
    -----------
    segmented_steps : list of dicts
        Each with 'step': DataFrame and metadata.
    feature_key : str
        Column name to align and compare.
    n_points : int
        How many timepoints to resample to.
    """
    all_curves = []

    for step in segmented_steps:
        df = step['step']
        if feature_key not in df.columns:
            continue

        values = df[feature_key].values
        orig_len = len(values)

        # Normalize to n_points using interpolation
        norm_time = np.linspace(0, 1, orig_len)
        interp_time = np.linspace(0, 1, n_points)
        interp_values = np.interp(interp_time, norm_time, values)

        all_curves.append(interp_values)

    all_curves = np.array(all_curves)  # shape: (n_steps, n_points)
    mean_vals = np.mean(all_curves, axis=0)
    sem_vals = np.std(all_curves, axis=0) / np.sqrt(all_curves.shape[0])

    # Plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=np.linspace(0, 100, n_points),
        y=mean_vals,
        mode='lines',
        name='Mean',
        line=dict(color='black')
    ))

    fig.add_trace(go.Scatter(
        x=np.concatenate([np.linspace(0, 100, n_points), np.linspace(100, 0, n_points)]),
        y=np.concatenate([mean_vals + sem_vals, (mean_vals - sem_vals)[::-1]]),
        fill='toself',
        fillcolor='rgba(100,100,100,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='SEM'
    ))

    fig.update_layout(
        title=f"Phase-Aligned Average: {feature_key}",
        xaxis_title='Normalized Step (%)',
        yaxis_title=feature_key,
        template='plotly_white',
        width=800,
        height=400
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

def plot_phase_aligned_loop(segmented_steps,
                            x_key='rhindlimb lHindfingers - X pose (m)',
                            y_key='rhindlimb lHindfingers - Y pose (m)',
                            n_points=100):
    """
    Plot a phase-aligned average trajectory (Y vs X), reordered to show a loop
    with swing phase preceding stance phase for cyclic clarity.

    Parameters:
    -----------
    segmented_steps : list
        List of step dictionaries with 'step', 'stance', and 'swing' info.
    x_key, y_key : str
        Keys for the X and Y coordinates to plot.
    n_points : int
        Number of points to normalize each trajectory to.
    show_all_steps : bool
        If True, show individual steps as faint lines.
    """
    aligned_x, aligned_y, aligned_phase = [], [], []

    for step in segmented_steps:
        df = step["step"]
        if x_key not in df.columns or y_key not in df.columns:
            continue

        stance_start, stance_end = step["stance"]
        swing_start, swing_end = step["swing"]
        frames = list(df.index)
        L = len(frames)
        norm_time = np.linspace(0, 1, L)

        # Phase per frame
        phase_seq = []
        for i in range(L):
            idx = frames[i]
            if stance_start <= idx <= stance_end:
                phase_seq.append("stance")
            elif swing_start <= idx <= swing_end:
                phase_seq.append("swing")
            else:
                phase_seq.append("other")

        # Interpolate
        x = np.interp(np.linspace(0, 1, n_points), norm_time, df[x_key].values)
        y = np.interp(np.linspace(0, 1, n_points), norm_time, df[y_key].values)
        phases = np.array(phase_seq)[np.round(np.linspace(0, L - 1, n_points)).astype(int)]

        # Reorder: swing → stance
        indices = np.arange(len(phases))
        swing_idx = indices[phases == 'swing']
        stance_idx = indices[phases == 'stance']
        reordered_idx = np.concatenate([swing_idx, stance_idx])

        if len(reordered_idx) < 5:
            continue

        rx = x[reordered_idx]
        ry = y[reordered_idx]
        rphase = phases[reordered_idx]

        aligned_x.append(rx)
        aligned_y.append(ry)
        aligned_phase.append(rphase)

    if len(aligned_x) == 0:
        print("No valid steps with sufficient stance/swing frames.")
        return

    aligned_x = np.array(aligned_x)
    aligned_y = np.array(aligned_y)
    aligned_phase = np.array(aligned_phase)

    fig = go.Figure()

    # Plot mean with phase colors
    mean_x = np.mean(aligned_x, axis=0)
    mean_y = np.mean(aligned_y, axis=0)
    mean_phase = aligned_phase[0]  # assume same alignment

    for i in range(len(mean_x) - 1):
        phase = mean_phase[i]
        color = 'rgba(255,0,0,1)' if phase == 'stance' else (
                'rgba(0,0,255,1)' if phase == 'swing' else 'rgba(150,150,150,0.6)')
        fig.add_trace(go.Scatter(
            x=mean_x[i:i+2],
            y=mean_y[i:i+2],
            mode='lines',
            line=dict(color=color, width=3),
            hoverinfo='skip',
            showlegend=False
        ))

    # Start/end markers
    fig.add_trace(go.Scatter(
        x=[mean_x[0]], y=[mean_y[0]],
        mode='markers+text',
        marker=dict(size=8, color='green'),
        text=["start"], textposition="top center", showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[mean_x[-1]], y=[mean_y[-1]],
        mode='markers+text',
        marker=dict(size=8, color='black'),
        text=["end"], textposition="bottom center", showlegend=False
    ))

    fig.update_layout(
        title="Phase-Aligned Gait Loop (Y vs X) with Phase Coloring",
        xaxis_title="X Position (m)",
        yaxis_title="Y Position (m)",
        width=800,
        height=600,
        template="plotly_white"
    )

    fig.show()
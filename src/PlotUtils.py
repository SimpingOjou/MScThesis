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

def plot_step_trajectory(segmented_steps, keys, limb_name='Hindlimb'):
    """
    Plot pose trajectories with stance and swing phase coloring.
    
    Parameters:
    -----------
    segmented : list
        List of step dictionaries containing step data, mouse info, group, stance, and swing phases
    keys : list
        List of column names for joints (should contain x and y coordinates)
    max_plots : int, optional (default=10)
        Maximum number of plots to generate
    healthy_key : str, optional (default=HEALTHY_KEY)
        Key to identify healthy group data
    
    Returns:
    --------
    None (displays plots)
    """
    max_plots = 2
    
    processed_count = 0
    
    for step in segmented_steps:
        if processed_count >= max_plots:
            break
        
        # Extract step data
        step_df = step["step"][keys]
        mouse = step["mouse"]
        group = step["group"]
        stance_start, stance_end = step["stance"]
        swing_start, swing_end = step["swing"]
        
        if HEALTHY_KEY not in group:
            continue

        # Extract joint information
        x_keys, y_keys, joint_names = extract_joint_keys(keys)
        
        # Create figure
        fig = go.Figure()
        
        # Add frame traces
        first_frame = step_df.index[0]
        last_frame = step_df.index[-1]
        for frame_idx, frame in enumerate(step_df.index):
            frame_data = step_df.loc[frame]
            phase = determine_phase(frame_idx, stance_start, stance_end, swing_start, swing_end)
            
            alpha = frame / (last_frame - first_frame + 1)  # Normalize alpha based on frame index
            frame_trace = create_frame_trace(frame_data, x_keys, y_keys, joint_names, frame, phase, alpha=alpha)
            fig.add_trace(frame_trace)
        
        # Update layout and show
        fig.update_layout(dict(
            title=f"{limb_name} Pose Trajectory - {group} {mouse}",
            xaxis_title="X Position (m)",
            yaxis_title="Y Position (m)",
            width=PLOT_WIDTH,
            height=PLOT_HEIGHT,
            showlegend=False,
            title_font=dict(size=TITLE_FONT_SIZE),
            xaxis=dict(title_font=dict(size=AXIS_TITLE_FONT_SIZE), tickfont=dict(size=AXIS_FONT_SIZE), showgrid=False),
            yaxis=dict(title_font=dict(size=AXIS_TITLE_FONT_SIZE), tickfont=dict(size=AXIS_FONT_SIZE), showgrid=False),
            template='plotly_white',
        ))
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


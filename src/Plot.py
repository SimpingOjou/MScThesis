import webbrowser
import os
import itertools
import umap
import math

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from collections import defaultdict, Counter

from src.PlotUtils import (
    PLOT_STYLE,
    majority_vote, determine_phase, get_phase_color, 
    get_joint_trajectory_segmented_by_phase,
    extract_joint_keys, create_frame_trace, 
    compute_phase_aligned_average, plot_phase_aligned_trace,
    normalize_spatial_curve
)

# === Step Plotting Function ===
def plot_step_trajectory(segmented_steps, keys, limb_name='Hindlimb',
                         joint_trace_keys=('rhindlimb lHindfingers - X pose (m)', 
                                            'rhindlimb lHindfingers - Y pose (m)')):
    from plotly.subplots import make_subplots
    max_plots = 2
    processed_count = 0

    for step in segmented_steps:
        if processed_count >= max_plots:
            break

        step_df = step["step"]
        mouse = step["mouse"]
        group = step["group"]
        stance_start, stance_end = step["stance"]
        swing_start, swing_end = step["swing"]

        if PLOT_STYLE.healthy_key not in group:
            continue

        x_keys, y_keys, joint_names = extract_joint_keys(keys)

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=False,
            row_heights=[0.75, 0.25],
            vertical_spacing=0.08,
            subplot_titles=[
                f"{limb_name} Pose Trajectory - {group} {mouse}", ""
            ]
        )

        frames = list(step_df.index)
        total_frames = len(frames)
        stances = [(stance_start, stance_end)]
        swings = [(swing_start, swing_end)]

        for i, frame in enumerate(frames):
            frame_data = step_df.loc[frame]
            phase = determine_phase(frame, stances, swings)
            alpha = 0.3 + 0.7 * i / total_frames
            trace = create_frame_trace(frame_data, x_keys, y_keys, joint_names, frame, phase, alpha=alpha)
            fig.add_trace(trace, row=1, col=1)

        x_joint_key, y_joint_key = joint_trace_keys
        if x_joint_key in step_df.columns and y_joint_key in step_df.columns:
            segments = get_joint_trajectory_segmented_by_phase(step_df, x_joint_key, y_joint_key, stances, swings, total_frames)
            for xs, ys, phase, alpha in segments:
                color = get_phase_color(phase, alpha)
                fig.add_trace(go.Scatter(
                    x=xs,
                    y=ys,
                    mode='lines+markers',
                    line=dict(width=1, color=color),
                    marker=dict(size=PLOT_STYLE.scatter_size, color=color, symbol=PLOT_STYLE.scatter_symbol),
                    showlegend=False,
                    hoverinfo='skip'
                ), row=2, col=1)
        else:
            print(f"Warning: Joint keys {joint_trace_keys} not found in step data.")

        fig.update_layout(
            width=PLOT_STYLE.plot_width,
            height=PLOT_STYLE.plot_height,
            template='plotly_white',
            showlegend=False,
            title_font=dict(size=PLOT_STYLE.title_font_size),
            xaxis=dict(title="", showgrid=False, title_font=dict(size=PLOT_STYLE.axis_title_font_size)),
            yaxis=dict(title="Y Position (m)", showgrid=False, title_font=dict(size=PLOT_STYLE.axis_title_font_size)),
            xaxis2=dict(title=x_joint_key, showgrid=False, title_font=dict(size=PLOT_STYLE.axis_title_font_size)),
            yaxis2=dict(title=y_joint_key, showgrid=False, title_font=dict(size=PLOT_STYLE.axis_title_font_size)),
        )

        fig.show()
        processed_count += 1

# === Trajectory Plotting Function ===
def plot_trajectory(data, keys, limb_name='Hindlimb'):
    max_plots = 2
    processed_count = 0

    for exp in data:
        if processed_count >= max_plots:
            break

        if PLOT_STYLE.healthy_key not in exp["group"]:
            continue

        step_df = exp["data"][keys]
        mouse = exp["mouse"]
        group = exp["group"]
        run = exp["run"]
        limb = "hindlimb" if limb_name.lower() == 'hindlimb' else 'forelimb'
        stances = exp["stance"][limb]
        swings = exp["swing"][limb]

        x_keys, y_keys, joint_names = extract_joint_keys(keys)

        fig = go.Figure()
        frames = list(step_df.index)
        total_frames = len(frames)

        for i, frame in enumerate(frames):
            if frame < stances[0][0] or frame > swings[-1][1]:
                continue

            frame_data = step_df.loc[frame]
            phase = determine_phase(frame, stances, swings)
            alpha = 0.3 + 0.7 * i / total_frames
            trace = create_frame_trace(frame_data, x_keys, y_keys, joint_names, frame, phase, alpha)
            fig.add_trace(trace)

        fig.update_layout(
            title=f"{limb_name} Pose Trajectory - {group} {mouse} {run}",
            xaxis_title="X Position (m)",
            yaxis_title="Y Position (m)",
            width=PLOT_STYLE.plot_width * 2,
            height=PLOT_STYLE.plot_height,
            showlegend=False,
            title_font=dict(size=PLOT_STYLE.title_font_size),
            xaxis=dict(title_font=dict(size=PLOT_STYLE.axis_title_font_size), tickfont=dict(size=PLOT_STYLE.axis_font_size), showgrid=False),
            yaxis=dict(title_font=dict(size=PLOT_STYLE.axis_title_font_size), tickfont=dict(size=PLOT_STYLE.axis_font_size), showgrid=False),
            template='plotly_white'
        )

        fig.show()
        processed_count += 1

def plot_animated_trajectory(data, keys, limb_name='Hindlimb'):
    max_plots = 2
    processed_count = 0

    for exp in data:
        if processed_count >= max_plots:
            break

        if PLOT_STYLE.healthy_key not in exp["group"]:
            continue

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

        fig = go.Figure()

        init_frame = frames[0]
        init_data = step_df.loc[init_frame]
        init_phase = determine_phase(init_frame, stances, swings)
        alpha = 0.3
        init_trace = create_frame_trace(init_data, x_keys, y_keys, joint_names, init_frame, init_phase, alpha)
        fig.add_trace(init_trace)

        animation_frames = []
        for i, frame in enumerate(frames):
            if frame < stances[0][0] or frame > swings[-1][1]:
                continue

            frame_data = step_df.loc[frame]
            phase = determine_phase(frame, stances, swings)
            alpha = 0.3 + 0.7 * i / total_frames
            trace = create_frame_trace(frame_data, x_keys, y_keys, joint_names, frame, phase, alpha)

            animation_frames.append(go.Frame(data=[trace], name=str(i)))

        fig.frames = animation_frames

        x_max = step_df[x_keys].max().max()

        fig.update_layout(
            title=f"{limb_name} Pose Trajectory - {group} {mouse} {run}",
            xaxis=dict(title="X Position (m)", showgrid=False, range=[0, x_max]),
            yaxis=dict(title="Y Position (m)", showgrid=False),
            width=PLOT_STYLE.plot_width * 2,
            height=PLOT_STYLE.plot_height,
            showlegend=False,
            template='plotly_white',
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(label="Play", method="animate", args=[None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True}]),
                    dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0}, "mode": "immediate", "transition": {"duration": 0}}])
                ]
            )],
            sliders=[{
                "steps": [
                    {"args": [[str(k)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}], "label": str(k), "method": "animate"}
                    for k in range(len(fig.frames))
                ],
                "transition": {"duration": 0},
                "x": 0, "y": -0.1,
                "currentvalue": {"prefix": "Frame: "}
            }]
        )

        fig.show()
        html_path = f"/tmp/animated_trajectory_{mouse}_{run}.html"
        fig.write_html(html_path)
        webbrowser.open('file://' + os.path.realpath(html_path))
        processed_count += 1

# === Trajectory with Dual Joint Traces ===
def plot_trajectory_with_joint_traces(data, keys_1, keys_2, limb_name='Hindlimb'):
    max_plots = 2
    processed_count = 0

    for exp in data:
        if processed_count >= max_plots:
            break

        if PLOT_STYLE.healthy_key not in exp["group"]:
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

        fig = make_subplots(rows=2, cols=1,
                            shared_xaxes=False,
                            vertical_spacing=0.08,
                            row_heights=[0.5, 0.5],
                            subplot_titles=[
                                f"Pose Trajectory - {group} {mouse} {run}", ""
                            ])

        for i, frame in enumerate(frames):
            if frame < stances[0][0] or frame > swings[-1][1]:
                continue
            frame_data = step_df.loc[frame]
            phase = determine_phase(frame, stances, swings)
            alpha = 0.3 + 0.7 * i / total_frames
            trace1 = create_frame_trace(frame_data, x_keys, y_keys, joint_names, frame, phase, alpha=alpha)
            trace2 = create_frame_trace(frame_data, x_joint_keys, y_joint_keys, joint_names_2, frame, phase, alpha=alpha)
            fig.add_trace(trace1, row=1, col=1)
            fig.add_trace(trace2, row=2, col=1)

        fig.update_layout(
            width=PLOT_STYLE.plot_width * 2,
            height=int(PLOT_STYLE.plot_height * 1.2),
            showlegend=False,
            template='plotly_white',
            title_font=dict(size=PLOT_STYLE.title_font_size),
            xaxis=dict(title="", showgrid=False, title_font=dict(size=PLOT_STYLE.axis_title_font_size)),
            yaxis=dict(title="", showgrid=False, title_font=dict(size=PLOT_STYLE.axis_title_font_size)),
            xaxis2=dict(title="", showgrid=False, title_font=dict(size=PLOT_STYLE.axis_title_font_size)),
            yaxis2=dict(title="", showgrid=False, title_font=dict(size=PLOT_STYLE.axis_title_font_size)),
        )

        fig.show()
        processed_count += 1

# === Compare Step Features in Batches ===
def compare_step_features_in_batches(step_dict, batch_size=25, output_dir="./figures/SCI"):
    import os
    import itertools
    from plotly.subplots import make_subplots

    os.makedirs(os.path.abspath(output_dir), exist_ok=True)

    step_df = step_dict["step"]
    mouse = step_dict.get("mouse", "UnknownMouse")
    run = step_dict.get("run", "Run1")
    stance_start, stance_end = step_dict.get("stance", (0, -1))
    swing_start, swing_end = step_dict.get("swing", (0, -1))
    stances = [(stance_start, stance_end)]
    swings = [(swing_start, swing_end)]

    features = [col for col in step_df.columns if "pose" in col or "CoM" in col]
    feature_pairs = list(itertools.combinations(features, 2))

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

        for i, (x_feat, y_feat) in enumerate(batch_pairs):
            xs, ys, phases = [], [], []
            for frame_idx, frame in enumerate(step_df.index):
                x = step_df.loc[frame, x_feat]
                y = step_df.loc[frame, y_feat]
                phase = determine_phase(frame_idx, stances, swings)
                xs.append(x)
                ys.append(y)
                phases.append(phase)

            for phase_name in ["stance", "swing", "other"]:
                filtered = [(x, y) for x, y, p in zip(xs, ys, phases) if p == phase_name]
                if not filtered:
                    continue
                color = get_phase_color(phase_name, alpha=0.8)
                fx, fy = zip(*filtered)
                fig.add_trace(go.Scatter(
                    x=fx,
                    y=fy,
                    mode='markers',
                    marker=dict(size=5, color=color),
                    name=phase_name.capitalize(),
                    showlegend=(i == 0),
                    hoverinfo='x+y'
                ), row=i+1, col=1)

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
            import webbrowser
            webbrowser.open('file://' + os.path.realpath(file_path))


# === UMAP from Single Step ===
def plot_umap_from_step(step_dict):
    import numpy as np
    import umap

    df = step_dict['step']
    stance_start, stance_end = step_dict['stance']
    swing_start, swing_end = step_dict['swing']
    stances = [(stance_start, stance_end)]
    swings = [(swing_start, swing_end)]

    features = [col for col in df.columns if 'pose' in col or 'CoM' in col]
    X = df[features].values

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42, n_components=3)
    X_embedded = reducer.fit_transform(X)

    phase_labels = [determine_phase(idx, stances, swings) for idx in df.index]
    coords = {p: [] for p in ['stance', 'swing', 'other']}
    for i, phase in enumerate(phase_labels):
        coords[phase].append(X_embedded[i])

    fig = go.Figure()
    color_map = {'stance': 'red', 'swing': 'blue', 'other': 'gray'}
    for phase, pts in coords.items():
        if pts:
            pts = np.array(pts)
            fig.add_trace(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode='markers+lines',
                marker=dict(size=4, color=color_map[phase], opacity=0.7),
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

# === UMAP from All Steps ===
def plot_umap_all_steps(segmented_steps, n_neighbors=15, min_dist=0.1):
    all_features = []
    metadata = []

    for step_id, step_data in enumerate(segmented_steps):
        df = step_data["step"]
        stance_start, stance_end = step_data["stance"]
        swing_start, swing_end = step_data["swing"]
        stances = [(stance_start, stance_end)]
        swings = [(swing_start, swing_end)]

        features = [col for col in df.columns if 'pose' in col or 'CoM' in col]
        if not features:
            continue

        X = df[features].values
        frame_indices = df.index.to_numpy()
        n_frames = X.shape[0]

        for i in range(n_frames):
            idx = frame_indices[i]
            phase = determine_phase(idx, stances, swings)
            all_features.append(X[i])
            metadata.append({
                'step_id': step_id,
                'frame': idx,
                'phase': phase,
                'time_norm': i / (n_frames - 1 if n_frames > 1 else 1),
                'mouse': step_data.get('mouse', 'unknown'),
                'run': step_data.get('run', 'unknown')
            })

    if not all_features:
        print("No usable step data.")
        return

    all_features = np.array(all_features)
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, random_state=42)
    embedding = reducer.fit_transform(all_features)

    grouped = defaultdict(list)
    for i, meta in enumerate(metadata):
        key = meta['step_id']
        grouped[key].append((embedding[i], meta))

    phase_color = {'swing': 'blue', 'stance': 'red', 'other': 'gray'}
    fig = go.Figure()

    for step_id, points in grouped.items():
        points.sort(key=lambda p: p[1]['time_norm'])
        coords = np.array([p[0] for p in points])
        phases = [p[1]['phase'] for p in points]
        opacities = [0.3 + 0.7 * p[1]['time_norm'] for p in points]

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
        template='plotly_white'
    )
    fig.show()


# === Compare Phase-Aligned Averages ===
def plot_mean_spatial_trajectory(segmented_steps,
                                  x_key='rhindlimb lHindfingers - X pose (m)',
                                  y_key='rhindlimb lHindfingers - Y pose (m)',
                                  n_points=100):
    aligned_x, aligned_y, aligned_phase = [], [], []

    for step in segmented_steps:
        df = step["step"]
        if x_key not in df.columns or y_key not in df.columns:
            continue

        stance_start, stance_end = step["stance"]
        swing_start, swing_end = step["swing"]
        stances = [(stance_start, stance_end)]
        swings = [(swing_start, swing_end)]

        frames = list(df.index)
        norm_time = np.linspace(0, 1, len(frames))
        phase_seq = [determine_phase(idx, stances, swings) for idx in frames]

        x_interp = np.interp(np.linspace(0, 1, n_points), norm_time, df[x_key].values)
        y_interp = np.interp(np.linspace(0, 1, n_points), norm_time, df[y_key].values)
        phase_interp = np.array(phase_seq)[np.round(np.linspace(0, len(frames)-1, n_points)).astype(int)]

        aligned_x.append(x_interp)
        aligned_y.append(y_interp)
        aligned_phase.append(phase_interp)

    aligned_x = np.array(aligned_x)
    aligned_y = np.array(aligned_y)
    aligned_phase = np.array(aligned_phase)

    mean_x = np.mean(aligned_x, axis=0)
    mean_y = np.mean(aligned_y, axis=0)

    from collections import Counter
    phase_labels = [Counter(aligned_phase[:, i]).most_common(1)[0][0] for i in range(n_points)]

    fig = go.Figure()
    current_phase = phase_labels[0]
    xs, ys = [], []

    for i in range(n_points):
        if phase_labels[i] != current_phase and xs:
            color = get_phase_color(current_phase)
            fig.add_trace(go.Scatter(
                x=xs, y=ys,
                mode='lines+markers',
                line=dict(color=color),
                marker=dict(size=5, color=color),
                name=current_phase,
                showlegend=False
            ))
            xs, ys = [], []

        xs.append(mean_x[i])
        ys.append(mean_y[i])
        current_phase = phase_labels[i]

    if xs:
        color = get_phase_color(current_phase)
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode='lines+markers',
            line=dict(color=color),
            marker=dict(size=5, color=color),
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


# === compare_phase_aligned_average_single ===
def compare_phase_aligned_average_single(healthy_steps, unhealthy_steps, feature_keys, n_points=100):
    pose_keys = [k for k in feature_keys if ('Angle -' in k or 'CoM' in k) and 'velocity' not in k and 'acceleration' not in k]
    vel_keys = [k for k in feature_keys if 'velocity' in k and 'acceleration' not in k]
    acc_keys = [k for k in feature_keys if 'acceleration' in k]
    grouped = list(zip(pose_keys, vel_keys, acc_keys))

    fig = make_subplots(rows=3, cols=len(grouped), shared_xaxes=True,
                        horizontal_spacing=0.03, vertical_spacing=0.08)
    time_vals = np.linspace(0, 100, n_points)

    for col_idx, (pose_key, vel_key, acc_key) in enumerate(grouped):
        for row_idx, key in enumerate([pose_key, vel_key, acc_key], start=1):
            h_mean, h_sem, h_phase = compute_phase_aligned_average(healthy_steps, key, n_points)
            u_mean, u_sem, u_phase = compute_phase_aligned_average(unhealthy_steps, key, n_points)
            if h_mean is None or u_mean is None:
                continue
            plot_phase_aligned_trace(fig, row_idx, col_idx + 1, time_vals, h_mean, h_sem, h_phase, light=False)
            plot_phase_aligned_trace(fig, row_idx, col_idx + 1, time_vals, u_mean, u_sem, u_phase, light=True)

    name = 'Phase-Aligned'
    fig.update_layout(
        height=350 * 3,
        width=300 * len(grouped),
        title=f"{name} Healthy vs Unhealthy Comparison",
        template='plotly_white'
    )
    fig.show()


# === compare_phase_aligned_average_xy ===
def compare_phase_aligned_average_xy(healthy_steps, unhealthy_steps, feature_keys, n_points=100):
    x_keys = [k for k in feature_keys if 'X' in k]
    y_keys = [k for k in feature_keys if 'Y' in k]
    max_len = max(len(x_keys), len(y_keys))

    fig = make_subplots(rows=2, cols=max_len, shared_yaxes=False,
                        subplot_titles=x_keys + y_keys, horizontal_spacing=0.05)
    time_vals = np.linspace(0, 100, n_points)

    for col_idx in range(max_len):
        for row_idx, key_list in zip([1, 2], [x_keys, y_keys]):
            if col_idx >= len(key_list):
                continue
            key = key_list[col_idx]
            h_mean, h_sem, h_phase = compute_phase_aligned_average(healthy_steps, key, n_points)
            u_mean, u_sem, u_phase = compute_phase_aligned_average(unhealthy_steps, key, n_points)
            if h_mean is None or u_mean is None:
                continue
            plot_phase_aligned_trace(fig, row_idx, col_idx + 1, time_vals, h_mean, h_sem, h_phase, light=False)
            plot_phase_aligned_trace(fig, row_idx, col_idx + 1, time_vals, u_mean, u_sem, u_phase, light=True)

    name = 'Phase-Aligned'
    fig.update_layout(
        height=700,
        width=350 * max_len,
        title=f"{name} Healthy vs Unhealthy Comparison (X/Y)",
        template='plotly_white'
    )
    fig.show()


# === plot_phase_aligned_average_single ===
def plot_phase_aligned_average_single(segmented_steps, feature_keys, n_points=100):
    pose_keys = [k for k in feature_keys if ('Angle -' in k or 'CoM' in k) and 'velocity' not in k and 'acceleration' not in k]
    vel_keys = [k for k in feature_keys if 'velocity' in k and 'acceleration' not in k]
    acc_keys = [k for k in feature_keys if 'acceleration' in k]
    grouped = list(zip(pose_keys, vel_keys, acc_keys))

    fig = make_subplots(rows=3, cols=len(grouped), shared_xaxes=True,
                        subplot_titles=pose_keys + vel_keys + acc_keys,
                        horizontal_spacing=0.03, vertical_spacing=0.08)
    time_vals = np.linspace(0, 100, n_points)

    for col_idx, (pose_key, vel_key, acc_key) in enumerate(grouped):
        for row_idx, key in enumerate([pose_key, vel_key, acc_key], start=1):
            mean, sem, phases = compute_phase_aligned_average(segmented_steps, key, n_points)
            if mean is None:
                continue
            plot_phase_aligned_trace(fig, row_idx, col_idx + 1, time_vals, mean, sem, phases, light=False)

    fig.update_layout(
        height=350 * 3,
        width=300 * len(grouped),
        title="Phase-Aligned Averages by Kinematic Type with SEM",
        template='plotly_white'
    )
    fig.show()


# === plot_phase_aligned_average_xy ===
def plot_phase_aligned_average_xy(segmented_steps, feature_keys, n_points=100):
    x_keys = [k for k in feature_keys if 'X' in k]
    y_keys = [k for k in feature_keys if 'Y' in k]
    max_len = max(len(x_keys), len(y_keys))

    fig = make_subplots(rows=2, cols=max_len, shared_yaxes=False,
                        subplot_titles=x_keys + y_keys, horizontal_spacing=0.05)
    time_vals = np.linspace(0, 100, n_points)

    for col_idx in range(max_len):
        for row_idx, key_list in zip([1, 2], [x_keys, y_keys]):
            if col_idx >= len(key_list):
                continue
            key = key_list[col_idx]
            mean, sem, phases = compute_phase_aligned_average(segmented_steps, key, n_points)
            if mean is None:
                continue
            plot_phase_aligned_trace(fig, row_idx, col_idx + 1, time_vals, mean, sem, phases, light=False)

    fig.update_layout(
        height=700,
        width=350 * max_len,
        title="Phase-Aligned Averages with SEM",
        template='plotly_white'
    )
    fig.show()

def plot_trajectory_with_joint_trace(data, keys, limb_name='Hindlimb',
                                     trace_keys=('rhindlimb lHindfingers - X pose (m)',
                                                 'rhindlimb lHindfingers - Y pose (m)')):
    from plotly.subplots import make_subplots

    max_plots = 2
    processed_count = 0

    for exp in data:
        if processed_count >= max_plots:
            break

        if PLOT_STYLE.healthy_key not in exp["group"]:
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

        fig = make_subplots(rows=2, cols=1,
                            shared_xaxes=False,
                            vertical_spacing=0.08,
                            row_heights=[0.75, 0.25],
                            subplot_titles=[
                                f"{limb_name} Pose Trajectory - {group} {mouse} {run}", ""
                            ])

        for i, frame in enumerate(frames):
            if frame < stances[0][0] or frame > swings[-1][1]:
                continue
            frame_data = step_df.loc[frame]
            phase = determine_phase(frame, stances, swings)
            alpha = 0.3 + 0.7 * i / total_frames
            trace = create_frame_trace(frame_data, x_keys, y_keys, joint_names, frame, phase, alpha=alpha)
            fig.add_trace(trace, row=1, col=1)

        x_joint_key, y_joint_key = trace_keys
        if x_joint_key in step_df.columns and y_joint_key in step_df.columns:
            segments = get_joint_trajectory_segmented_by_phase(step_df, x_joint_key, y_joint_key, stances, swings, total_frames)
            for xs, ys, phase, alpha in segments:
                color = get_phase_color(phase, alpha)
                fig.add_trace(go.Scatter(
                    x=xs,
                    y=ys,
                    mode='lines+markers',
                    line=dict(width=1, color=color),
                    marker=dict(size=PLOT_STYLE.scatter_size, color=color, symbol=PLOT_STYLE.scatter_symbol),
                    showlegend=False,
                    hoverinfo='skip'
                ), row=2, col=1)
        else:
            print(f"Warning: Joint keys {trace_keys} not found in step data.")

        fig.update_layout(
            width=PLOT_STYLE.plot_width * 2,
            height=int(PLOT_STYLE.plot_height * 1.2),
            showlegend=False,
            template='plotly_white',
            title_font=dict(size=PLOT_STYLE.title_font_size),
            xaxis=dict(title="", showgrid=False, title_font=dict(size=PLOT_STYLE.axis_title_font_size)),
            yaxis=dict(title="", showgrid=False, title_font=dict(size=PLOT_STYLE.axis_title_font_size)),
            xaxis2=dict(title=x_joint_key, showgrid=False, title_font=dict(size=PLOT_STYLE.axis_title_font_size)),
            yaxis2=dict(title=y_joint_key, showgrid=False, title_font=dict(size=PLOT_STYLE.axis_title_font_size)),
        )

        fig.show()
        processed_count += 1

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
                    line=dict(width=PLOT_STYLE.scatter_line_width, color=color),
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
def plot_trajectory_with_joint_traces(data, keys_1, keys_2, limb_name='Hindlimb', figure_path=None):
    for exp in data:
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
            alpha = 1
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

        if figure_path:
            filename = f"{group}_{mouse}_{run}_trajectory_with_foresteps.svg"
            fig.write_image(os.path.join(figure_path, filename), format='svg')
        else:
            fig.show()

# === Compare Step Features in Batches ===
def compare_step_features_in_batches(step_dict, batch_size=25, output_dir="./figures/SCI"):
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
        # decomporre in corpo in pezzi e far vedere che ogni parte riflette il SCI
        features = [col for col in df.columns if ('pose' in col or 'CoM' in col) and 'hindlimb' in col.lower()]
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
def compare_phase_aligned_average_single(healthy_steps, unhealthy_steps, feature_keys, n_points=100, figure_path=None):
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
        template='plotly_white',
        title_font=dict(size= PLOT_STYLE.title_font_size),
        annotations=[dict(font=dict(size=14)) for _ in fig.layout.annotations],
        xaxis=dict(showgrid=False)
    )
    if figure_path:    
        # save the figure in a vectorial format
        fig.write_image(figure_path, format='svg')
    else:
        fig.show()

def compare_spatial_angle_progression_over_time(
    healthy_steps, unhealthy_steps,
    feature_keys, frame_rate=200, n_points=100,
    figure_path=None
):

    def compute_derivatives(arr, dt):
        vel = np.gradient(arr, dt)
        acc = np.gradient(vel, dt)
        return vel, acc

    def interpolate_group(steps, key):
        all_times, all_angles, all_vels, all_accs, all_phases = [], [], [], [], []

        for step in steps:
            df = step['step']
            if key not in df.columns:
                continue

            angle = df[key].to_numpy()
            frames = df.index.to_numpy()
            times = frames / frame_rate if frame_rate else frames - frames[0]
            dt = 1.0 / frame_rate if frame_rate else 1.0

            angle_range = np.max(angle) - np.min(angle)
            if angle_range == 0:
                continue

            norm_angle = (angle - angle[0]) / angle_range
            vel, acc = compute_derivatives(norm_angle, dt)

            interp_t = np.linspace(times[0], times[-1], n_points)
            interp_a = np.interp(interp_t, times, norm_angle)
            interp_v = np.interp(interp_t, times, vel)
            interp_ac = np.interp(interp_t, times, acc)

            stance_start, stance_end = step["stance"]
            swing_start, swing_end = step["swing"]
            stances = [(stance_start, stance_end)]
            swings = [(swing_start, swing_end)]

            interp_phases = [
                determine_phase(int(np.interp(t, times, frames)), stances, swings)
                for t in interp_t
            ]

            all_times.append(interp_t)
            all_angles.append(interp_a)
            all_vels.append(interp_v)
            all_accs.append(interp_ac)
            all_phases.append(interp_phases)

        if not all_times:
            return None, None, None, None, None

        all_angles = np.array(all_angles)
        all_vels = np.array(all_vels)
        all_accs = np.array(all_accs)
        all_phases = np.array(all_phases)
        mean_t = np.mean(np.array(all_times), axis=0)

        def summary(arr):
            return np.mean(arr, axis=0), np.std(arr, axis=0) / np.sqrt(arr.shape[0])

        a_mean, a_sem = summary(all_angles)
        v_mean, v_sem = summary(all_vels)
        ac_mean, ac_sem = summary(all_accs)
        dominant_phases = [Counter(all_phases[:, i]).most_common(1)[0][0] for i in range(n_points)]

        return mean_t, (a_mean, a_sem), (v_mean, v_sem), (ac_mean, ac_sem), dominant_phases

    angle_keys = [k for k in feature_keys if 'velocity' not in k and 'acceleration' not in k]
    n_cols = len(angle_keys)
    fig = make_subplots(
        rows=3, cols=n_cols,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[
            f"{key} (Angle)" for key in feature_keys
        ] + [
            f"{key} (Velocity)" for key in feature_keys
        ] + [
            f"{key} (Acceleration)" for key in feature_keys
        ]
    )

    for col_idx, key in enumerate(angle_keys):
        h_t, h_ang, h_vel, h_acc, h_phase = interpolate_group(healthy_steps, key)
        u_t, u_ang, u_vel, u_acc, u_phase = interpolate_group(unhealthy_steps, key)
        if h_ang is None or u_ang is None:
            continue

        for row, (h_data, u_data) in enumerate(zip([h_ang, h_vel, h_acc], [u_ang, u_vel, u_acc]), start=1):
            h_mean, h_sem = h_data
            u_mean, u_sem = u_data
            plot_phase_aligned_trace(fig, row, col_idx + 1, h_t, h_mean, h_sem, h_phase, light=False)
            plot_phase_aligned_trace(fig, row, col_idx + 1, u_t, u_mean, u_sem, u_phase, light=True)

    fig.update_layout(
        height=350 * 3,
        width=300 * n_cols,
        title="Angle, Velocity, and Acceleration over Time (Normalized to Angle Range)",
        xaxis_title="Time (s)" if frame_rate else "Frame Index",
        xaxis=dict(showgrid=False),
        yaxis_title="Normalized Units",
        template='plotly_white'
    )
    if figure_path:
        fig.write_image(figure_path, format='svg')
    else:
        fig.show()

# === compare_phase_aligned_average_xy ===
def compare_phase_aligned_average_xy(healthy_steps, unhealthy_steps, feature_keys, n_points=100, figure_path=None):
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
        template='plotly_white',
        xaxis=dict(showgrid=False)
    )
    if figure_path:
        fig.write_image(figure_path, format='svg')
    else:
        fig.show()

def compare_spatial_progression_over_time(healthy_steps, unhealthy_steps, finger_pose_key, frame_rate=None, n_points=100):

    def interpolate_group(steps):
        all_times, all_yvals, all_phases = [], [], []

        for step in steps:
            df = step['step']
            if finger_pose_key not in df.columns:
                continue

            x = df[finger_pose_key].to_numpy()
            frames = df.index.to_numpy()
            times = frames / frame_rate if frame_rate else frames - frames[0]

            step_length = x[-1] - x[0]
            if step_length == 0:
                continue

            norm_x = (x - x[0]) / step_length

            interp_t = np.linspace(times[0], times[-1], n_points)
            interp_y = np.interp(interp_t, times, norm_x)

            stance_start, stance_end = step["stance"]
            swing_start, swing_end = step["swing"]
            stances = [(stance_start, stance_end)]
            swings = [(swing_start, swing_end)]

            interp_phases = [
                determine_phase(int(np.interp(t, times, frames)), stances, swings)
                for t in interp_t
            ]

            all_times.append(interp_t)
            all_yvals.append(interp_y)
            all_phases.append(interp_phases)

        if not all_times:
            return None, None, None, None

        all_times = np.array(all_times)
        all_yvals = np.array(all_yvals)
        all_phases = np.array(all_phases)

        mean_t = np.mean(all_times, axis=0)
        mean_y = np.mean(all_yvals, axis=0)
        sem_y = np.std(all_yvals, axis=0) / np.sqrt(all_yvals.shape[0])
        dominant_phases = [Counter(all_phases[:, i]).most_common(1)[0][0] for i in range(n_points)]

        return mean_t, mean_y, sem_y, dominant_phases

    # === Interpolate each group ===
    h_time, h_mean, h_sem, h_phase = interpolate_group(healthy_steps)
    u_time, u_mean, u_sem, u_phase = interpolate_group(unhealthy_steps)

    if h_time is None or u_time is None:
        print("Missing data for one of the groups.")
        return

    # === Plot ===
    fig = go.Figure()

    def add_trace(mean_t, mean_y, sem_y, phase_labels, light=False):
        current_phase = phase_labels[0]
        xs, ys, ys_upper, ys_lower = [], [], [], []

        for i in range(n_points):
            if phase_labels[i] != current_phase and xs:
                color = get_phase_color(current_phase, alpha=0.7 if not light else 0.4)
                fig.add_trace(go.Scatter(
                    x=xs, y=ys,
                    mode='lines',
                    line=dict(color=color, width=3, dash='dot' if light else 'solid'),
                    showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x=xs + xs[::-1],
                    y=ys_upper + ys_lower[::-1],
                    fill='toself',
                    fillcolor=color.replace('0.7', '0.2') if not light else color.replace('0.4', '0.1'),
                    line=dict(color='rgba(255,255,255,0)'),
                    showlegend=False
                ))
                xs, ys, ys_upper, ys_lower = [], [], [], []

            xs.append(mean_t[i])
            ys.append(mean_y[i])
            ys_upper.append(mean_y[i] + sem_y[i])
            ys_lower.append(mean_y[i] - sem_y[i])
            current_phase = phase_labels[i]

        if xs:
            color = get_phase_color(current_phase, alpha=0.7 if not light else 0.4)
            fig.add_trace(go.Scatter(
                x=xs, y=ys,
                mode='lines',
                line=dict(color=color, width=3, dash='dot' if light else 'solid'),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=xs + xs[::-1],
                y=ys_upper + ys_lower[::-1],
                fill='toself',
                fillcolor=color.replace('0.7', '0.2') if not light else color.replace('0.4', '0.1'),
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False
            ))

    # Healthy = solid, Unhealthy = dotted
    add_trace(h_time, h_mean, h_sem, h_phase, light=False)
    add_trace(u_time, u_mean, u_sem, u_phase, light=True)

    fig.update_layout(
        title="Normalized Spatial Progression vs. Time (Healthy vs. Unhealthy)",
        xaxis_title="Time (s)" if frame_rate else "Frame Index",
        yaxis_title="Normalized Step Position (0–1)",
        width=800,
        height=600,
        template='plotly_white',
    )
    fig.show()

def compare_spatial_progression_xy_over_time(
    healthy_steps, unhealthy_steps,
    feature_keys, length_key, height_key,
    frame_rate=200, n_points=100, figure_path=None
):
    def interpolate_group(steps, key, norm_key, is_x_axis=True):
        all_times, all_yvals, all_phases = [], [], []

        for step in steps:
            df = step['step']
            if key not in df.columns or norm_key not in df.columns:
                continue

            values = df[key].to_numpy()
            norm_ref = df[norm_key].to_numpy()
            frames = df.index.to_numpy()
            times = frames / frame_rate if frame_rate else frames - frames[0]

            norm_range = max(norm_ref) - min(norm_ref)
            if norm_range == 0:
                continue

            norm_vals = (values - values[0]) / norm_range

            interp_t = np.linspace(times[0], times[-1], n_points)
            interp_y = np.interp(interp_t, times, norm_vals)

            stance_start, stance_end = step["stance"]
            swing_start, swing_end = step["swing"]
            stances = [(stance_start, stance_end)]
            swings = [(swing_start, swing_end)]

            interp_phases = [
                determine_phase(int(np.interp(t, times, frames)), stances, swings)
                for t in interp_t
            ]

            all_times.append(interp_t)
            all_yvals.append(interp_y)
            all_phases.append(interp_phases)

        if not all_times:
            return None, None, None, None

        all_times = np.array(all_times)
        all_yvals = np.array(all_yvals)
        all_phases = np.array(all_phases)

        mean_t = np.mean(all_times, axis=0)
        mean_y = np.mean(all_yvals, axis=0)
        sem_y = np.std(all_yvals, axis=0) / np.sqrt(all_yvals.shape[0])
        dominant_phases = [Counter(all_phases[:, i]).most_common(1)[0][0] for i in range(n_points)]

        return mean_t, mean_y, sem_y, dominant_phases

    x_keys = [k for k in feature_keys if 'X' in k]
    y_keys = [k for k in feature_keys if 'Y' in k]
    max_len = max(len(x_keys), len(y_keys))
    subplot_titles = x_keys + y_keys

    fig = make_subplots(
        rows=2, cols=max_len,
        shared_yaxes=False,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.05
    )

    for col_idx in range(max_len):
        for row_idx, key_list, norm_key in zip(
            [1, 2],
            [x_keys, y_keys],
            [length_key, height_key]
        ):
            if col_idx >= len(key_list):
                continue
            key = key_list[col_idx]

            h_time, h_mean, h_sem, h_phase = interpolate_group(healthy_steps, key, norm_key, is_x_axis=(row_idx == 1))
            u_time, u_mean, u_sem, u_phase = interpolate_group(unhealthy_steps, key, norm_key, is_x_axis=(row_idx == 1))

            if h_mean is None or u_mean is None:
                continue

            plot_phase_aligned_trace(
                fig, row_idx, col_idx + 1,
                h_time, h_mean, h_sem, h_phase, light=False
            )
            plot_phase_aligned_trace(
                fig, row_idx, col_idx + 1,
                u_time, u_mean, u_sem, u_phase, light=True
            )

    fig.update_layout(
        height=700,
        width=350 * max_len,
        title="Normalized XY Progression over Time (Healthy vs. Unhealthy)",
        xaxis_title="Time (s)" if frame_rate else "Frame Index",
        xaxis=dict(showgrid=False),
        yaxis_title="Normalized to Step Length / Height",
        template='plotly_white'
    )
    if figure_path:
        fig.write_image(figure_path, format='svg')
    else:
        fig.show()

# === plot_phase_aligned_average_single ===
def plot_phase_aligned_average_single(segmented_steps, feature_keys, n_points=100, figure_path=None):
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
        template='plotly_white',
        xaxis=dict(showgrid=False)
    )
    if figure_path:
        # save the figure in a vectorial format
        fig.write_image(figure_path, format='svg')
    else:
        fig.show()


# === plot_phase_aligned_average_xy ===
def plot_phase_aligned_average_xy(segmented_steps, feature_keys, n_points=100, figure_path=None):
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
        width=500 * max_len,
        title="Phase-Aligned Averages with SEM",
        template='plotly_white',
        xaxis=dict(showgrid=False)
    )
    if figure_path:
        # save the figure in a vectorial format
        fig.write_image(figure_path, format='svg')
    else:
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

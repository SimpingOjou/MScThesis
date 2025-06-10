import plotly.graph_objects as go

def plot_averaged_xy_trajectories_plotly(averaged_steps, 
                                        keys_to_plot=None,
                                        title="Averaged Step Trajectories (X vs Y)"):
    """
    Plots X vs Y for each mouse's averaged step trace using Plotly.

    Parameters:
        averaged_steps (list of dict): each dict with 'step_df', 'mouse', 'group'
    """
    fig = go.Figure()

    for entry in averaged_steps:
        step_df = entry["step"] if keys_to_plot is None else entry["step"][keys_to_plot]
        mouse = entry["mouse"]
        group = entry["group"]

        fig.add_trace(go.Scatter(
            x=step_df["x"],
            y=step_df["y"],
            mode='lines+markers',
            name=f"{mouse} ({group})",
            line=dict(width=3),
            marker=dict(size=4, symbol="circle")
        ))

    fig.update_layout(
        title=title,
        xaxis_title="X",
        yaxis_title="Y",
        legend=dict(font=dict(size=10)),
        xaxis=dict(scaleanchor="y"),  # Equal aspect ratio
        yaxis=dict(autorange="reversed"),  # Flip to match video orientation
        width=800,
        height=700
    )

    fig.show()

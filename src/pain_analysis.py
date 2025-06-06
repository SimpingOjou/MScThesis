import os
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import RobustScaler, StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import pairwise_distances
from skbio.stats.distance import permanova, DistanceMatrix
from xgboost import XGBRegressor
import shap
import umap

FIGURES_DIR = '../figures/'
DATA_DIR = '../csv/'
# DATA_FILE = 'Pain_hindlimb_mouse_features_2025-05-23_15-06-55.csv'
DATA_FILE = 'Pain_hindlimb_mouse_features_no_during_td_2025-06-05_16-48-56.csv'

def preprocess_data(df: pd.DataFrame):
    data = df.copy()

    label_encoder_dataset = LabelEncoder()
    label_encoder_mouse = LabelEncoder()

    data["Dataset_Encoded"] = label_encoder_dataset.fit_transform(data["Dataset"])
    data["Mouse_Encoded"] = label_encoder_mouse.fit_transform(data["Mouse"])

    dataset_label_map = dict(zip(data["Dataset_Encoded"], data["Dataset"]))
    mouse_label_map = dict(zip(data["Mouse_Encoded"], data["Mouse"]))
    label_cols = ["Dataset_Encoded", "Mouse_Encoded"]

    data = data.drop(columns=["Dataset", "Mouse"])
    features = [col for col in data.columns if col not in label_cols]

    ## Transform rads to degrees for interpretability
    for feature in features:
        if 'angle' in feature or 'phase' in feature:
            if 'rad' in feature:
                data[feature] = np.degrees(data[feature])

    ## Default to RobustScaler, but use StandardScaler for angle/phase features since they tend to be more normally distributed
    # angle_features = [f for f in features if 'angle' in f or 'phase' in f]
    # other_features = [f for f in features if f not in angle_features]
    
    # scaler_angle = StandardScaler()
    # scaler_other = StandardScaler()

    # X_angle_scaled = scaler_angle.fit_transform(data[angle_features])
    # X_other_scaled = scaler_other.fit_transform(data[other_features])
    # X_scaled = pd.DataFrame(
    #     data = np.hstack([X_angle_scaled, X_other_scaled]),
    #     columns = angle_features + other_features,
    #     index = data.index
    # )

    # X_scaled = data[features].values
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(data[features])
    y = data[label_cols].values

    df_preprocessed = pd.DataFrame(X_scaled, columns=features)
    df_preprocessed[label_cols] = y

    print("[DEBUG] Dataset Label Map:", dataset_label_map)
    print("[DEBUG] Mouse Label Map:", mouse_label_map)

    return df_preprocessed, dataset_label_map, mouse_label_map, label_cols

def aggregate_features(df: pd.DataFrame, label_cols: list, method: str = "mean"):
    if method not in ["mean", "median", "sum"]:
        raise ValueError("Aggregation method not supported")

    group = df.groupby(["Mouse_Encoded", "Dataset_Encoded"])
    agg_func = getattr(group, method)
    aggregated = agg_func().reset_index()

    X = aggregated.drop(columns=label_cols)
    y_mouse = aggregated["Mouse_Encoded"]
    y_dataset = aggregated["Dataset_Encoded"]

    return X, y_mouse, y_dataset

def select_features_with_lasso(X, y, feature_names=None, alpha=0.01, top_k=None, use_cv=False):
    """
    Performs LASSO-based feature selection.

    Parameters:
        X (pd.DataFrame): Feature matrix.
        y (array-like): Target vector (e.g. dataset encoding).
        feature_names (list): Names of the features (optional if X is DataFrame).
        alpha (float): Regularization strength for LASSO (ignored if use_cv=True).
        top_k (int): If provided, selects only the top-k features. If None, keeps all non-zero weights.
        use_cv (bool): If True, uses LassoCV to find optimal alpha.

    Returns:
        X_selected (pd.DataFrame): Reduced feature matrix.
        selected_features (list): Names of selected features.
        coef_series (pd.Series): All feature weights (can be used for plotting).
    """
    if feature_names is None:
        feature_names = X.columns if isinstance(X, pd.DataFrame) else [f"f{i}" for i in range(X.shape[1])]

    model = LassoCV(cv=5) if use_cv else Lasso(alpha=alpha, max_iter=10000)
    model.fit(X, y)

    coef_series = pd.Series(model.coef_, index=feature_names)
    nonzero = coef_series[coef_series != 0]

    if top_k is not None:
        top_features = nonzero.abs().sort_values(ascending=False).head(top_k).index.tolist()
    else:
        top_features = nonzero.index.tolist()

    print(f"[DEBUG] Selected {len(top_features)} features out of {len(feature_names)}")

    X_selected = X[top_features]

    return X_selected, top_features, coef_series

def plot_lasso_feature_importance(coef_series: pd.Series, top_k: int = 10):
    """
    Plot top-k features ranked by LASSO absolute coefficient value.

    Parameters:
        coef_series (pd.Series): Feature coefficients from LASSO (index=feature names).
        top_k (int): Number of top features to display.
    """
    nonzero = coef_series[coef_series != 0]
    if nonzero.empty:
        print("[LASSO] No non-zero coefficients to plot.")
        return

    top_features = nonzero.abs().sort_values(ascending=False).head(top_k)
    feature_names = top_features.index[::-1]
    importance = top_features.values[::-1]

    plt.figure(figsize=(15, 10))
    plt.barh(feature_names, importance, color='skyblue')
    plt.xlabel("Feature Importance (Lasso Coefficient)", fontsize=14)
    # plt.ylabel("Feature")
    plt.yticks(fontsize=12)
    plt.title(f"Top {top_k} Features by Lasso Regression", fontsize=16)
    plt.grid(True, axis='x')
    plt.tight_layout()

    datetime = time.strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(FIGURES_DIR, f"lasso_feature_importance_{datetime}.png")
    # plt.savefig(filename)

    plt.show()

def perform_dimensionality_reduction(
        X: pd.DataFrame,
        y_mouse,
        y_dataset,
        dataset_label_map,
        mouse_label_map,
        method: str = "PCA",
        n_components: int = 3,
        plot: bool = True,
        save: bool = True,
        **kwargs
    ):
    """
    Performs dimensionality reduction and plots 2D projections of 3 components.
    """

    # Perform dimensionality reduction
    if method == "PCA":
        full_pca = PCA(n_components=min(X.shape))
        full_result = full_pca.fit_transform(X)
        X_reduced = full_result[:, :n_components]
        reducer = full_pca

    elif method == "UMAP":
        reducer = umap.UMAP(
            n_components=n_components,
            random_state=42,
            target_metric=kwargs.get("target_metric", "categorical"),
            target_weight=kwargs.get("target_weight", 0),
            n_neighbors=kwargs.get("n_neighbors", 10),
            min_dist=kwargs.get("min_dist", 0.1),
        )
        X_reduced = reducer.fit_transform(X, y=y_dataset)

    else:
        raise ValueError("Reduction method not supported")

    # Prepare DataFrame for plotting
    comp_cols = [f"{method}{i+1}" for i in range(n_components)]
    plot_df = pd.DataFrame(X_reduced, columns=comp_cols)
    plot_df["Dataset"] = y_dataset
    plot_df["Mouse"] = y_mouse

    df_processed = plot_df.copy()
    embedding = X_reduced
    datasets = np.unique(y_dataset)

    if not plot:
        return reducer, plot_df

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    cmap = plt.get_cmap("tab20")

    handles_labels = []
    for i, (x_idx, y_idx) in enumerate([(0, 1), (1, 2), (0, 2)]):
        ax = axes[i]
        for j, dataset in enumerate(datasets):
            mask = df_processed["Dataset"] == dataset
            color = cmap(j % cmap.N)
            scatter = ax.scatter(
                embedding[mask, x_idx], embedding[mask, y_idx],
                label=dataset_label_map.get(dataset, str(dataset)),
                c=[color],
                edgecolors='black',  # Add black borders to markers
                linewidths=0.5
            )
            # Store one handle per dataset only once (during first subplot)
            if i == 0:
                handles_labels.append((scatter, dataset_label_map.get(dataset, str(dataset))))
        ax.grid(True)
        ax.set_xlabel(f"Component {x_idx+1}", fontsize=14)
        ax.set_ylabel(f"Component {y_idx+1}", fontsize=14)

    # Unpack handles and labels
    handles, labels = zip(*handles_labels)

    # Place the legend outside the right of the plots
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.9, 0.5), title="Dataset", fontsize=14)

    plt.suptitle(f"{method} Projection (2D views of 3 components)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])  # leave space for external legend

    # Save and show
    datetime = time.strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(FIGURES_DIR, f"{method.lower()}_projection_{datetime}.png")
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(filename, bbox_inches='tight') if save else None
    plt.show()

    if method == "PCA":
        print_top_pca_loadings(reducer, X.columns.tolist(), n_components=n_components)
    return reducer, plot_df

def print_top_pca_loadings(pca_model: PCA, feature_names: list, n_components: int = 3, top_k: int = 10):
    """
        Prints top features contributing to the first `n_components` principal components.
    """
    comp_cols = [f"PCA{i+1}" for i in range(n_components)]
    loadings = pd.DataFrame(
        pca_model.components_[:n_components].T,
        columns=comp_cols,
        index=feature_names
    )

    print("-" * 50)
    for i, pc in enumerate(comp_cols):
        top_feats = loadings[pc].abs().nlargest(top_k).index.tolist()
        explained_var = pca_model.explained_variance_ratio_[i] * 100
        print(f"Top features for {pc} (Explained Variance: {explained_var:.2f}%):")
        print("  •", ", ".join(top_feats))
    print("-" * 50)

def plot_umap_feature_importance_with_xgboost(X: pd.DataFrame, umap_embedding: np.ndarray, top_k: int = 10, save: bool = True):
    """
    Trains an XGBoost regressor to predict each UMAP component from features,
    and plots the top-k most important features for each component.

    Parameters:
        X (pd.DataFrame): Input features used in UMAP.
        umap_embedding (np.ndarray): UMAP-reduced coordinates (n_samples, 3).
        top_k (int): Number of top features to show per component.
    """
    component_names = ["UMAP1", "UMAP2", "UMAP3"]

    for i in range(3):
        model = XGBRegressor(n_estimators=100, random_state=42)
        model.fit(X, umap_embedding[:, i])

        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1][:top_k]
        top_features = X.columns[sorted_idx]
        top_importances = importances[sorted_idx]

        plt.figure(figsize=(15, 10))
        plt.barh(top_features[::-1], top_importances[::-1], color="skyblue")
        plt.title(f"Top {top_k} Features for {component_names[i]}", fontsize=16)
        plt.xlabel("XGBoost Importance", fontsize=14)
        plt.ylabel("Feature", fontsize=14)
        plt.yticks(fontsize=12)
        plt.grid(True, axis='x')
        plt.tight_layout()

        datetime = time.strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(FIGURES_DIR, f"xgboost_umap_feature_importance_{component_names[i].lower()}_{datetime}.png")
        plt.savefig(filename) if save else None

        # SHAP summary
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)

        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X, max_display=top_k,
                          plot_size=(12, 8),
                          show=False)  # Delay display until saved
        plt.title(f"SHAP Summary for {component_names[i]}", fontsize=16)
        shap_fig_name = f"shap_summary_{component_names[i].lower()}_{datetime}.png"
        plt.savefig(os.path.join(FIGURES_DIR, shap_fig_name), bbox_inches='tight') if save else None
        # plt.show()

def run_permanova_on_umap(umap_coords: np.ndarray, group_labels: list, sample_ids: list = None, n_permutations: int = 999):
    """
    Run PERMANOVA on UMAP embedding with group labels. PERMANOVA (Permutational Multivariate Analysis of Variance) is a non-parametric statistical test used to compare groups of multivariate samples (e.g., UMAP embeddings) based on distances between them. Are the centroids (means) of different groups in the multivariate space more different than you'd expect by chance?

    Parameters:
        umap_coords (np.ndarray): shape (n_samples, n_components)
        group_labels (list or array): group assignment for each sample (e.g., 'pre', 'post')
        sample_ids (list): optional list of unique IDs for each sample
        n_permutations (int): number of permutations for PERMANOVA test

    Returns:
        result (DataFrame): PERMANOVA summary
    """
    if sample_ids is None:
        sample_ids = [f"sample_{i}" for i in range(len(group_labels))]

    # Compute pairwise Euclidean distance matrix
    dist_matrix = pairwise_distances(umap_coords, metric='euclidean')
    dm = DistanceMatrix(dist_matrix, ids=sample_ids)

    # Run PERMANOVA
    result = permanova(dm, grouping=group_labels, permutations=n_permutations)
    print(result)
    return result

if __name__ == '__main__':
    ## Load the data
    full_data = pd.read_csv(os.path.join(DATA_DIR, DATA_FILE))

    ## Drop unnecessary columns and merge
    full_data['Dataset'] = full_data['Dataset'].str.replace('DLC', '_') + full_data['Mouse'].str.split('_').str[1]
    to_drop = ["Number of runs (#)", "Number of steps (#)"]
    full_data.drop(columns=to_drop, inplace=True)

    print(f'[DEBUG] Datasets: {full_data["Dataset"].unique()}')
    print(f'[DEBUG] Feature count: {len(full_data.columns) - 2}')
    print(f'[DEBUG] Mice count: {len(full_data)/2}')

    ## Preprocess data
    preprocessed_data, dataset_label_map, mouse_label_map, label_cols = preprocess_data(full_data)

    ## Aggregate features
    X_agg, y_mouse, y_dataset = aggregate_features(preprocessed_data, label_cols, method="mean")

    ## Feature selection with LASSO
    X_agg_lasso, selected_features, lasso_weights = select_features_with_lasso(
        X_agg,
        y_dataset,
        feature_names=X_agg.columns,
        # use_cv=True,  # Use LassoCV to find optimal alpha
        alpha=0.01,  # or use_cv=True
        top_k=None   # or top_k=50 for a fixed limit
    )

    # plot_lasso_feature_importance(lasso_weights, top_k=len(selected_features))
    # plot_lasso_feature_importance(lasso_weights, top_k=30)

    ### Perform dimensionality reduction
    # # PCA
    # perform_dimensionality_reduction(X_agg, y_mouse, y_dataset,
    #                                 dataset_label_map, mouse_label_map,
    #                                 method="PCA", n_components=3)
    
    # UMAP
    reducer, umap_plot_df = perform_dimensionality_reduction(
        X_agg, y_mouse, y_dataset,
        dataset_label_map, mouse_label_map,
        method="UMAP", n_components=3,
        target_metric='categorical',
        target_weight=0,
        n_neighbors=15,
        min_dist=0.05,
        plot=False
    )

    # LASSO -> UMAP
    # reducer, umap_plot_df = perform_dimensionality_reduction(
    #     X_agg_lasso, y_mouse, y_dataset,
    #     dataset_label_map, mouse_label_map,
    #     method="UMAP", n_components=3,
    #     target_metric='categorical',
    #     target_weight=0.5,
    #     n_neighbors=15,
    #     min_dist=0.05
    # )

    # ## LASSO -> PCA
    # perform_dimensionality_reduction(
    #     X_agg_lasso, y_mouse, y_dataset,
    #     dataset_label_map, mouse_label_map,
    #     method="PCA", n_components=3
    # )

    # UMAP -> XGBoost feature importance
    umap_embedding = umap_plot_df[["UMAP1", "UMAP2", "UMAP3"]].values
    # plot_umap_feature_importance_with_xgboost(X_agg, umap_embedding, top_k=30)

    ### Run PERMANOVA on UMAP coordinates
    
    ## Filter for specific datasets
    umap_plot_df["Dataset_label"] = umap_plot_df["Dataset"].map(dataset_label_map)
    # Only pre_left datasets
    umap_plot_df = umap_plot_df[umap_plot_df["Dataset_label"].str.contains("left")]
    # # Only pre_right datasets
    # umap_plot_df = umap_plot_df[umap_plot_df["Dataset_label"].str.contains("pre_right")]
    # Only C and E datasets
    # umap_plot_df = umap_plot_df[
    #     umap_plot_df["Dataset_label"].str.contains("C_post_left|E_post_left") |
    #     (umap_plot_df["Dataset_label"].str.contains("pre_left"))
    # ]

    ### Plot UMAP projection with all "pre" datasets colored the same
    plt.figure(figsize=(18, 6))
    sns.set(style="whitegrid")

    # Assign a new column for coloring: "pre" vs others
    umap_plot_df["ColorGroup"] = np.where(
        umap_plot_df["Dataset_label"].str.contains("pre"), "pre", umap_plot_df["Dataset_label"]
    )

    # Build color palette: one color for "pre", unique colors for others
    unique_groups = umap_plot_df["ColorGroup"].unique()
    palette = {}
    base_palette = sns.color_palette("husl", len(unique_groups))
    # Assign first color to "pre", rest to other groups
    pre_color = base_palette[0]
    palette["pre"] = pre_color
    other_colors = iter(base_palette[1:])
    for group in unique_groups:
        if group != "pre":
            palette[group] = next(other_colors)

    umap_components = [("UMAP1", "UMAP2"), ("UMAP2", "UMAP3"), ("UMAP1", "UMAP3")]
    titles = [
        "UMAP1 vs UMAP2",
        "UMAP2 vs UMAP3",
        "UMAP1 vs UMAP3"
    ]

    for i, (x_comp, y_comp) in enumerate(umap_components):
        ax = plt.subplot(1, 3, i + 1)
        sns.scatterplot(
            data=umap_plot_df,
            x=x_comp,
            y=y_comp,
            hue="ColorGroup",
            palette=palette,
            s=100,
            edgecolor='black',
            linewidth=0.5,
            ax=ax,
            legend=False if i < 2 else "brief"
        )
        ax.set_title(titles[i])
        ax.set_xlabel(x_comp)
        ax.set_ylabel(y_comp)
        ax.grid(True)

    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles, labels, title="Dataset", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.suptitle("UMAP Projections of Filtered Datasets", fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()

    # 3D interactive UMAP plot
    # import plotly.express as px
    # import webbrowser
    # import tempfile

    # fig = px.scatter_3d(
    #     umap_plot_df,
    #     x="UMAP1",
    #     y="UMAP2",
    #     z="UMAP3",
    #     color="Dataset_label",
    #     symbol="Dataset_label",
    #     hover_data=umap_plot_df.columns,
    #     title="3D UMAP Projection of Filtered Datasets"
    # )
    # fig.update_traces(marker=dict(size=7, line=dict(width=0.5, color='DarkSlateGrey')))
    # fig.update_layout(legend=dict(title="Dataset", x=1.05, y=1))
    # tmpfile = tempfile.NamedTemporaryFile(suffix='.html', delete=False)
    # fig.write_html(tmpfile.name)
    # webbrowser.open('file://' + tmpfile.name)

    ## Apply PERMANOVA
    # group_labels = umap_plot_df["Dataset_label"].values
    # umap_coords = umap_plot_df[["UMAP1", "UMAP2", "UMAP3"]].values
    # permanova_result = run_permanova_on_umap(
    #     umap_coords,
    #     group_labels,
    #     sample_ids=umap_plot_df.index.tolist(),
    #     n_permutations=999
    # )
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import umap
from scipy.stats import pearsonr, ttest_ind

def get_preprocessed_data(full_data):
    # Drop columns that are not needed
    columns_to_drop = ["Mouse", "Number of runs (#)", "Number of steps (#)"]
    columns_to_drop = [col for col in columns_to_drop if col in full_data.columns]
    full_data = full_data.drop(columns=columns_to_drop)

    # Identify the categorical and numerical columns
    category_col = "Dataset"
    numerical_cols = full_data.select_dtypes(include=[np.number]).columns.tolist()

    # Encode the categorical column to number
    label_encoder = LabelEncoder()
    full_data["Dataset_Encoded"] = label_encoder.fit_transform(full_data[category_col])

    # Standardize the features
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(full_data[numerical_cols])

    # Convert to df
    df_processed = pd.DataFrame(data_scaled, columns=numerical_cols)
    df_processed["Dataset"] = full_data["Dataset_Encoded"]

    print(df_processed.head())

    return df_processed

def get_PCA(full_data, df, plot:bool=True):
    pca = PCA(n_components=min(df.shape[0], df.shape[1]))
    pca_result = pca.fit_transform(df.drop(columns=["Dataset"]))
    pca_scores = pca.transform(df.drop(columns=["Dataset"]))
    print(f"PCA shape: {pca_result.shape}")

    loadings = pd.DataFrame(pca.components_.T, columns=[f"PC{i}" for i in range(1, pca_result.shape[1]+1)], index=df.drop(columns=["Dataset"]).columns)

    top_features_pc1 = loadings["PC1"].abs().sort_values(ascending=False).head(10)
    top_features_pc1_names = top_features_pc1.index.tolist()
    top_features_pc2 = loadings["PC2"].abs().sort_values(ascending=False).head(10)
    top_features_pc2_names = top_features_pc2.index.tolist()
    top_features_pc3 = loadings["PC3"].abs().sort_values(ascending=False).head(10)
    top_features_pc3_names = top_features_pc3.index.tolist()

    print(f"Top features for PC1:\n{top_features_pc1}\n")
    print(f"Top features for PC2:\n{top_features_pc2}\n")
    print(f"Top features for PC3:\n{top_features_pc3}\n")

    for feature in top_features_pc1_names[:4]:
        # print correlation between feature and PC1
        correlation, p_value = pearsonr(df[feature], pca.transform(df.drop(columns=["Dataset"]))[:, 0])
        print(f"Correlation between {feature} and PC1: {correlation:.2f} (p-value: {p_value:.2f})")

    if plot:
        # Plot the first three principal components
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        unique_labels = full_data["Dataset"].unique()
        for i, label in enumerate(unique_labels):
            indices = df["Dataset"] == i
            cmap = plt.get_cmap('viridis')
            colors = cmap(df["Dataset"][indices] / df["Dataset"].max())
            ax.scatter(pca_result[indices, 0], pca_result[indices, 1], pca_result[indices, 2], c=colors, s=60, label=label)
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')
        plt.grid(True)
        plt.legend()
        plt.title('PCA Projection')
        plt.show(block=False)

        # Plot the explained variance ratio
        plt.figure(figsize=(8, 6))
        plt.plot(np.cumsum(pca.explained_variance_ratio_ * 100), marker='o')
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Cumulative Explained Variance (%)')
        plt.title('Explained Variance by Principal Components')
        plt.grid(True)
        plt.show(block=False)

        # Plot features across groups
        plt.figure(figsize=(8,6))
        for feature in top_features_pc1_names[:4]:
            feature_split = feature.split(" - ")
            name = str(feature_split[0]) + " " + str(feature_split[-1])
            plt.subplot(2, 2, top_features_pc1_names.index(feature)+1)
            sns.boxplot(x=full_data["Dataset"], y=df[feature])
            sns.stripplot(x=full_data["Dataset"], y=df[feature], color='black', alpha=0.5, jitter=True)
            plt.title(f"{name}")
            plt.xlabel("Dataset (State)")
            plt.ylabel("Value")
            plt.tight_layout(pad=3.0)

        plt.show(block=False)

    return pca_result, pca_scores, (top_features_pc1_names, top_features_pc2_names, top_features_pc3_names)

def get_UMAP(full_data, df, reduced_data, 
            n_components:int=3, n_neighbors:int=10, min_dist:float=0.1, random_state:int=42, 
            plot:bool=True):
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    embedding = reducer.fit_transform(reduced_data)
    print(f"UMAP shape: {embedding.shape}")

    if plot:
        # Plot the UMAP
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        unique_labels = full_data["Dataset"].unique()
        for i, label in enumerate(unique_labels):
            indices = df["Dataset"] == i
            ax.scatter(embedding[indices, 0], embedding[indices, 1], embedding[indices, 2], s=60, label=label)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_zlabel('UMAP 3')
        plt.title('UMAP Projection')
        plt.legend()
        plt.show()

    return embedding

def main():
    ## Load data
    data_folder = './csv/'
    # data_file = 'A_left_forelimb_mouse_features_2025-02-25_14-10-48.csv'
    # data_file = 'E_left_right_hindlimb_post.csv' # E L vs R vs post > spine 4 ang acceleration -> hip compensation going on healthy part
    # data_file = 'AB_LR_combined.csv' # AB combined pre and post > too many clusters, can't see
    # data_file = 'AB_LR_combined_post.csv' # AB combined post > too many clusters, can't see
    # data_file = 'B_hindlimb.csv' # B pre vs post > small changes
    # data_file = 'CE_LR_post.csv'
    data_file = 'CE_L_prepost.csv'

    full_data = pd.read_csv(data_folder + data_file)
    print(full_data.head())

    # Preprocess data
    df = get_preprocessed_data(full_data)

    # Perform PCA
    reduced_data, pca_scores, top_features = get_PCA(full_data, df, plot=True)

    # Perform UMAP
    umap_embedding = get_UMAP(full_data, df, reduced_data, 
                              n_components=3, n_neighbors=10, min_dist=0.1, random_state=42,
                              plot=True)

    ## Correlate UMAP with Principal Components (How umap dimensions are influenced by pca components)
    pca_component_list = [f"PCA{i}" for i in range(1, pca_scores.shape[1]+1)]
    umap_dim_list = [f"UMAP{i}" for i in range(1, umap_embedding.shape[1]+1)]
    umap_df = pd.DataFrame(umap_embedding, columns=[umap_dim_list])
    pca_df = pd.DataFrame(pca_scores, columns=[pca_component_list])

    correlation_matrix = pd.DataFrame(index=[pca_component_list[:3]], columns=[umap_dim_list])
    for pca_name in pca_component_list[:3]:
        for umap_dim in umap_dim_list:
            correlation, p_value = pearsonr(umap_df[umap_dim], pca_df[pca_name])
            correlation_matrix.loc[pca_name, umap_dim] = correlation

    correlation_matrix = correlation_matrix.astype(float)

    # # Plot heatmap
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0, fmt=".2f")
    # plt.title("Correlation Between UMAP Dimensions and PCA Components")
    # plt.xlabel("UMAP Dimensions")
    # plt.ylabel("PCA Components")
    # plt.show()

    # Visualize UMAP with feature overlays (top features from PCA)
    top_features_pc1, top_features_pc2, top_features_pc3 = top_features

    feature_to_plot = top_features_pc1[0]
    plt.figure(figsize=(8, 6))
    scatter = sns.scatterplot(x=umap_df["UMAP1"].values.flatten(), y=umap_df["UMAP2"].values.flatten(), hue=df[feature_to_plot], palette="viridis", alpha=0.8, legend=False)
    plt.colorbar(scatter.collections[0], label=feature_to_plot)
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.title(f"UMAP Projection Colored by {feature_to_plot}")
    plt.show()


if __name__ == '__main__':
    main()

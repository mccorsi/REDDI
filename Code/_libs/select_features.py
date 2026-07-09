# Feature selection functions

from sklearn.feature_selection import mutual_info_classif
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from scipy import stats as spstats

def univariate_feature_selection(df, num_features=20, corr_treshold=0.5):
    """
    Perform feature selection by computing the Pearson's correlation to lower the inter-correlation between features 
    and using the mutual information metric with respect to the target to maximize relevance. Return num_features 
    (default=20) most relevant features.
    
    Inputs:
    :df: input dataframe
    :num_features: number of features to select
    :corr_treshold: correlation treshold to remove highly correlated features
    
    Returns:
    :selected_features: list of selected feature names
    :mi_df: DataFrame containing feature names and their mutual information scores
    :sub_corr_matrix: correlation matrix of selected features
    """
    # Calculate the correlation matrix 
    dataframe = df.copy()
    corr_matrix = dataframe.drop(columns=["class", "id"]).corr()
    
     # Compute mutual information between each feature and the target
    X = dataframe.drop(columns=["class", "id"])
    y = dataframe["class"]
    mutual_info = mutual_info_classif(X, y, random_state=42)

    # Create a DataFrame to hold feature names and their mutual information scores
    mi_df = pd.DataFrame({"feature": X.columns, "mutual_info": mutual_info})

    # Sort features by their mutual information scores
    sorted_features = mi_df.sort_values(by="mutual_info", ascending=False)["feature"].tolist()
    
    selected_features = []
    while len(selected_features) < num_features and sorted_features:
        # Select the feature with the highest mutual information score
        feature = sorted_features.pop(0)
        selected_features.append(feature)
        
        # Remove features with high inter-correlation with the selected feature
        sorted_features = [f for f in sorted_features if abs(corr_matrix[feature][f]) < corr_treshold]
    
    mi_df = mi_df[mi_df["feature"].isin(selected_features)]   # take only the selected features
    sub_corr_matrix = corr_matrix.loc[selected_features, selected_features] # correlation matrix of selected features
    
    return selected_features, mi_df, sub_corr_matrix


###################################################################################################################
#            The folloing class is based on the work done in FUCONE                                               #
###################################################################################################################
class FC_DimRed(TransformerMixin, BaseEstimator):
    """
    Feature reduction class based on functional connectivity (FC) matrices.
    Selects discriminative nodes using the Kruskal-Wallis test and ranks nodes by their degree.

    Parameters:
    - threshold: p-value threshold for Kruskal-Wallis test
    - nb_nodes: number of nodes to select
    """
    def __init__(self, p_threshold = 0.05, eta_threshold = 0.10, nb_nodes = 78):
        self.p_threshold = p_threshold
        self.nb_nodes = nb_nodes
        self.eta_threshold = eta_threshold

    def fit(self, X, y, metric='p-value'):
        """
        Fit the model by identifying discriminative nodes based on the Kruskal-Wallis test.

        Parameters:
        - X: 3D array (samples, channels, channels) representing FC matrices
        - y: 1D array of class labels
        - metric: 'p-value' or 'eta-squared'
        """
        # Split data by class
        unique_classes = np.unique(y)
        FC_classes = [X[np.where(y == cls)] for cls in unique_classes]

        # Kruskal-Wallis test
        H, pvals = spstats.kruskal(*FC_classes, axis=0)

        if metric == 'p-value':
            # Binarize edges based on p-value threshold
            thresh_mask = (pvals < self.p_threshold).astype(int)
        if metric == 'eta-squared':
            # Compute eta-squared
            eta_squared = (H - 4 + 1) / (X.shape[0] - 4) 
            # Binarize edges based on eta-squared threshold
            thresh_mask = (eta_squared > self.eta_threshold).astype(int)

        # Compute node degree and select top nodes
        node_strength = np.sum(thresh_mask, axis=0)
        self.node_strength_ = node_strength 
        idx = np.argsort(-node_strength)
        self.node_select_ = idx[:self.nb_nodes]

        return self

    def transform(self, X):
        """
        Transform the input data by selecting the top-ranked nodes.

        Parameters:
        - X: 3D array (samples, channels, channels) representing FC matrices

        Returns:
        - Transformed 3D array with reduced dimensions
        """
        return X[:, self.node_select_, :][:, :, self.node_select_]
    
class AverageFilter(FC_DimRed):

    def fit(self, X, y, metric='p-value'):
        """
        Compute the average p-value or eta-squared among the significant values (pvals < p_threshold and eta_squared > eta_threshold) 
        and then rank based on the most significant averages.
        """
        # Split data by class
        unique_classes = np.unique(y)
        FC_classes = [X[np.where(y == cls)] for cls in unique_classes]

        # Kruskal-Wallis test
        H, pvals = spstats.kruskal(*FC_classes, axis=0)

        if metric == 'p-value':
            # Mask for significant edges
            signif_mask = (pvals < self.p_threshold)
            # Compute average p-value for each node (ignore non-significant with nan)
            avg_pvals = np.where(signif_mask, pvals, np.nan)
            node_scores = np.nanmean(avg_pvals, axis=0)
        elif metric == 'eta-squared':
            # Compute eta-squared
            eta_squared = (H - 4 + 1) / (X.shape[0] - 4)
            signif_mask = (eta_squared > self.eta_threshold)
            avg_eta = np.where(signif_mask, eta_squared, np.nan)
            node_scores = np.nanmean(avg_eta, axis=0)
        else:
            raise ValueError("Unknown metric: choose 'p-value' or 'eta-squared'")

        self.node_scores_ = node_scores

        return self
    

import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Applying PCA method. Nothing special here. it returns the Componenets of the features.


def applyPCA(feature, no_comp):
    X = StandardScaler().fit_transform(feature)
    pca = PCA(n_components = no_comp)
    pcomp = pca.fit_transform(X)

    return pcomp

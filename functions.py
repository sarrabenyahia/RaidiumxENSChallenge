import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.preprocessing import LabelEncoder


def display_missing_values(df:pd.DataFrame)->pd.DataFrame:
    """
    Fonction pour voir le nombre et pourcentage de missing values par variable dans un dataframe
    """
    count_missing_val = {}
    for c in df.columns:
        nb_miss = ((df[c].isna()) | (df[c].astype(str).str.strip()==""))*1
        nb_miss = nb_miss.sum()
        count_missing_val.update({c:[nb_miss,nb_miss/df.shape[0]]})
    
    res = pd.DataFrame(count_missing_val).T
    res = res.reset_index()
    res.columns = ['variable','nb_missing','pct_missing']
    
    display(res.sort_values('nb_missing',ascending=False))


def nb_modalities(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function is used to count the number of modalities from a variable
    """
    counting_modalities = {}

    for x in df.columns: 
        nb_mod = len(df[x].value_counts())
        nb_miss = ((df[x].isna()) | (df[x].astype(str).str.strip()==""))*1
        nb_miss = nb_miss.sum()
        counting_modalities.update({x:[nb_mod, nb_miss/df.shape[0]]})
    
    res = pd.DataFrame(counting_modalities).T
    res = res.reset_index()
    res.columns = ['variables','nb_modalities', 'pct_missing']
    res = res.sort_values('nb_modalities', ascending=True)

    return res

def unique_vals(df:pd.DataFrame) -> pd.DataFrame:
    """
    Counting unique values
    """
    count_missing_val = {}
    for c in df.columns:
        nb_vals = df[c].nunique()
        count_missing_val.update({c:[nb_vals, nb_vals/len(df)*100, nb_vals/df[c].notna().sum()*100]})
    
    res = pd.DataFrame(count_missing_val).T
    res = res.reset_index()
    res.columns = ['variable','uniques', '% over total', '% over not na']
    
    display(res.sort_values('uniques',ascending=False))

def show_missing(df):
    """Return a Pandas dataframe describing the contents of a source dataframe including missing values."""
    
    variables = []
    dtypes = []
    count = []
    unique = []
    missing = []
    pc_missing = []
    
    for item in df.columns:
        variables.append(item)
        dtypes.append(df[item].dtype)
        count.append(len(df[item]))
        unique.append(len(df[item].unique()))
        missing.append(df[item].isna().sum())
        pc_missing.append(round((df[item].isna().sum() / len(df[item])) * 100, 2))

    output = pd.DataFrame({
        'variable': variables, 
        'dtype': dtypes,
        'count': count,
        'unique': unique,
        'missing': missing, 
        'pc_missing': pc_missing
    })    
        
    return output


def find_outliers_IQR(df:pd.DataFrame):
    """function to find outliers"""
    outliers = {}
    numerical_columns = [cols for cols in df.columns if df[cols].dtype=="float64" or df[cols].dtype=="int64"]
    
    for column in numerical_columns:
        dataf = df[column]
        q1=dataf.quantile(0.25)
        q3=dataf.quantile(0.75)
        IQR=q3-q1
        outlier = dataf[((dataf<(q1-1.5*IQR)) | (dataf>(q3+1.5*IQR)))]
        outliers.update({column:[len(outlier), outlier.max(), outlier.min(), len(outlier)/len(dataf)]})

    res = pd.DataFrame(outliers).T
    res = res.reset_index()
    res.columns = ['feature', 'nb_outliers', 'max_value_outlier', 'min_value_outlier', 'outlier_ratio']

    display(res.sort_values('nb_outliers', ascending=False).reset_index())



def corr_to_feature(df:pd.DataFrame, thresh_corr:int) -> pd.DataFrame:
    """function to look the number of features correlated more than thresh_corr"""
    numerical_columns = [cols for cols in df.columns if df[cols].dtype=="float64" or df[cols].dtype=="int64"]
    matrice_corr = abs(df[numerical_columns].corr())

    corr = {}
    for i in range(matrice_corr.shape[1]):
        pct = (matrice_corr.iloc[i]>thresh_corr).sum()
        corr.update({matrice_corr.columns.tolist()[i] : [pct]}) 
        
    resultat = pd.DataFrame(corr).T
    resultat = resultat.reset_index()
    resultat.columns = ["Feature","Nb_corr"]
    return resultat


def impute_par_regression(df,target_col,features_col):
    """
    Input:
        df : la dataframe contenant toutes les données
        target_col : la variable cible contenant des valeurs manquantes qu'on souhaite imputer
        features_col : les variables explicatives utilisées pour imputer la variable cible
    return :
        df : dataframe contenant une nouvelle colonne target_col+"_reg" sans valeurs manquante
    """
    # creation de la variable qui sera imputé 
    df[target_col+"_reg"] = df[target_col]
    
    # Recuperation de toutes les observations ne contenant aucune valeur manquante
    mask = df[target_col].notna()
    for c in features_col:
        mask = (mask) & (df[c].notna()) 
    
    # Données d'apprentissage
    X_train = df.loc[mask,features_col].values
    y_train = df.loc[mask,target_col].values
    
    # Entrainement du modele
    clf = LinearRegression()
    clf.fit(X_train,y_train)
    
    # Prediction sur les valeurs manquantes
    y_pred = clf.predict(df.loc[df[target_col+"_reg"].isna(),features_col].values)
    df.loc[df[target_col+"_reg"].isna(),target_col+"_reg"] = y_pred
    return df

def frequency_encoding(df, cols, threshold=0.01):
    """Encode object columns in a DataFrame using frequency encoding.
    Args:
        df: The DataFrame to encode.
        cols: A list of object columns to encode.
        threshold: The minimum frequency for a value to be encoded. Values with
            lower frequencies will be replaced with 0.
    Returns:
        A new DataFrame with frequency encoding applied to the object columns.
    """
    # Create a copy of the DataFrame
    df_encoded = df.copy()
    
    # Loop over the object columns
    for col in cols:
        # Get the counts of each unique value
        counts = df[col].value_counts()
        
        # Filter the values to include only those above the threshold
        counts = counts[counts / len(df) > threshold]
        
        # Encode the values using the counts
        df_encoded[col] = df[col].map(counts)
        
        # Replace any missing values with 0
        df_encoded[col] = df_encoded[col].fillna(0)
        
    return df_encoded

def truncate_outliers(df, cols, quantile=0.99):
    """Truncate outliers in the specified columns of a DataFrame.
    
    Args:
        df: The DataFrame to truncate.
        cols: A list of columns to truncate.
        quantile: The quantile at which to truncate the outliers.
        
    Returns:
        A new DataFrame with outliers truncated.
    """
    # Create a copy of the DataFrame
    df_truncated = df.copy()
    
    # Loop over the columns to truncate
    for col in cols:
        # Calculate the truncation threshold
        threshold = df[col].quantile(quantile)
        
        # Truncate the outliers
        df_truncated[col] = np.where(df[col] > threshold, threshold, df[col])
        
    return df_truncated


def label_encode_rare(df, col, rare_freq=0.01):
    # Extract the categories and their frequencies
    categories, frequencies = np.unique(df[col], return_counts=True)
    # Create a mapping to the "Other" category for categories with frequency < rare_freq
    mapping = {cat: "Other" for cat, freq in zip(categories, frequencies) if freq/len(df) < rare_freq}
    # Use the mapping to create a new column with the encoded values
    df[col] = df[col].map(mapping).fillna(df[col])
    # Create a label encoder object
    encoder = LabelEncoder()
    # Fit the encoder to the column, then transform it
    df[col] = encoder.fit_transform(df[col])

def select_correlated_features(df, targets, threshold=0.5):
    """Select features that are correlated with the specified targets.
    
    Args:
        df: The DataFrame to select features from.
        targets: A list of target columns to compute correlations with.
        threshold: The minimum absolute correlation for a feature to be selected.
        
    Returns:
        A list of the selected feature columns.
    """
    # Calculate the correlations between the features and targets
    correlations = df.corrwith(df[targets[0]])
    
    # Select the features with an absolute correlation above the threshold
    selected_features = correlations[abs(correlations) > threshold].index
    
    return selected_features


def select_features(df:pd.DataFrame, feature:str, threshold_corr:int) -> list:
  """Calculate the correlations between the specified feature and the other features"""
  
  feature_correlations = abs(df.drop(feature, axis=1).corrwith(df[feature]))

  # Select the features with a correlation less than or equal to 0.7
  selected_features = feature_correlations[feature_correlations <= threshold_corr].index.tolist()

  return selected_features
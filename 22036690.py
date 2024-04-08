# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 02:57:54 2024

@author: DELL
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import cluster_tools as ct
import scipy.optimize as opt
import errors as err
from sklearn.impute import SimpleImputer


def read_data(file_paths, selected_countries, year):
    
    
    """
    Read and preprocess data from CSV files.

    Parameters:
    - file_paths (list of str): List of file paths for multiple datasets.
    - selected_countries (list of str): List of countries to include in the analysis.
    - start_year (int): Start year for data extraction.
    - end_year (int): End year for data extraction.

    Returns:
    - merged_df: DataFrame.
    - dataframes_dict_transpose (dict): Dictionary of transposed DataFrames.
    """

    dfs = []
    # Dictionary to store transposed DataFrames
    dataframes_dict_transpose = {}
    
    # Columns to exclude
    exclude_columns = ['Country Code', 'Indicator Name', 'Indicator Code']
    
    
    # Iterate over each file path
    for path in file_paths:
        
        # Extract file name without extension
        file_name = path.split('.')[0].replace(' ', '_')
        
        # Load the dataset, skipping the first 4 rows
        df = pd.read_csv(path, skiprows = 4, usecols = lambda x: x.strip() != "Unnamed: 67" if x not in ['Indicator Name', 'Indicator Code', 'Country Code'] else True)
        
        # Exclude specified columns
        df = df.drop(columns = exclude_columns, errors = 'ignore')
        
        df = df.set_index('Country Name')
        # Calculate the mean of each row
        # Define the indices you want to drop
        indices_to_drop = ["Africa Eastern and Southern", "Bosnia and Herzegovina",
                           "Central Europe and the Baltics", "East Asia & Pacific (excluding high income)",
                           "Early-demographic dividend", "East Asia & Pacific", "Europe & Central Asia",
                           "Euro area","European Union","IDA & IBRD total", "Latin America & Caribbean (excluding high income)",
                           "Latin America & Caribbean","Low & middle income","Late-demographic dividend",
                           "Macao SAR, China", "Middle income", "North Macedonia", "South Asia",
                           "East Asia & Pacific (IDA & IBRD countries)", "Europe & Central Asia (IDA & IBRD countries)",
                           "Latin America & the Caribbean (IDA & IBRD countries)","South Asia (IDA & IBRD)",
                           "Upper middle income"
                           ]
        
        # Remove the rows have nan [] from the dictionary
        df = df.dropna(subset = ['2012', '2022'])
        # Drop rows with the specified indices
        df = df.drop(indices_to_drop)

        df = df[[str(year)]]
        # Calculate the mean of each row
        #df['mean'] = df.mean(axis=0)
        
        # Fill null values in each column with the mean of that row
        df = df.apply(lambda col: col.fillna(col.mean()), axis=0)
        
        # Transpose the DataFrame
        df_trans = df.transpose()
        
        #clean the data
        df_trans.dropna(axis=0, how="all", inplace=True)
        
        # Reset index to make years a column
        df_trans.reset_index(inplace=True)
        
        df.columns = [file_name]  # Rename the column with file name
        dfs.append(df)
     #Initialize merged_df with the first dataframe   
    merged_df = dfs[0]
    # Iterate over remaining dataframes and merge with suffixes
    for i, df in enumerate(dfs[1:], start = 1):
        merged_df = pd.merge(merged_df, df, left_index = True, right_index = True, suffixes = ('', f'_file{i}'))
    
    dataframes_dict_transpose[file_name] = df_trans
    return merged_df, dataframes_dict_transpose


def read_data_for_fit(data_sets, selected_country, start_year, end_year):
    
    """
    Read and preprocess data from CSV files.

    Parameters:
    - file_paths (list of str): List of file paths for multiple datasets.
    - selected_countries (list of str): List of countries to include in the analysis.
    - start_year (int): Start year for data extraction.
    - end_year (int): End year for data extraction.

    Returns:
    - dataframes_dict (dict): Dictionary of DataFrames with keys as file names.
    """

    # Dictionary to store original DataFrames
    dataframes_dict = {}
    # Dictionary to store transposed DataFrames
    dataframes_dict_transpose = {}

    # Columns to exclude
    exclude_columns = ['Country Code', 'Indicator Name', 'Indicator Code']

    # Iterate over each file path
    for dataname in data_sets:

        # Extract file name without extension
        file_name = dataname.split('.')[0].replace(' ', '_')

        # Load the dataset, skipping the first 4 rows
        df_new = pd.read_csv(dataname, skiprows = 4, usecols = lambda x: x.strip() != "Unnamed: 67" if x not in [
                         'Indicator Name', 'Indicator Code', 'Country Code'] else True)

        # Exclude specified columns
        df_new = df_new.drop(columns=exclude_columns, errors='ignore')

        # Set 'Country Name' as the index
        df_new.set_index("Country Name", inplace = True)


        # Filter data for selected countries and the specified year range
        df_new = df_new.loc[selected_country, "2000":"2022"]
        column_name = file_name.replace('_', ' ')
        data_new = {'Year': df_new.index, column_name: df_new.values.flatten()}
        df_fit = pd.DataFrame(data_new)
        
        df_fit = df_fit.reset_index()
        print(df_fit)
        #print(df_result["Year"])
        # Store the new DataFrame in dictionaries
        dataframes_dict[file_name] = df_fit
        #print(dataframes_dict)
        # Store DataFrames in dictionaries
        #dataframes_dict[file_name] = df
        
    return dataframes_dict


def merg_data(dataframes_dict, indicator_inflation):
    
    
    """ merge multiple files in one dataset 
        giving indicators name as column names
    """
    
    
    result_df = dataframes_dict[indicator_inflation]
    suffix_count = 1
    for key, df1 in dataframes_dict.items():
       #print(df1)
       if key != indicator_inflation:
           # Merge if the dataframe is not empty and 'Year' is present
           if not df1.empty and 'Year' in df1.columns:
               # Specify columns to merge on
               merge_columns = ['Year']

               # Merge with explicit suffixes
               result_df = pd.merge(result_df, df1, on = merge_columns, how = 'outer', suffixes = ('', f'_{key}'))
    
     # Drop unnecessary index columns
    result_df = result_df.loc[:, ~result_df.columns.str.startswith('index')]
    result_df = result_df.loc[:, ~result_df.columns.str.startswith('Year_')]
    # Set 'Year' as the index
    result_df.set_index('Year', inplace = True)
    result_df.reset_index(inplace = True)
    # Convert 'Year' column to integers
    result_df['Year'] = result_df['Year'].astype(int)
    # Display the resulting dataframe
    
    # Create a new DataFrame without the index column
    df_fitted = result_df.iloc[:,0:2]
    #indicators.to_excel('result_df_output.xlsx', index=False)
    return df_fitted


def create_bar_chart(df12, df22, selected_country_bar_chart):
    # We will create bar charts for each of the indicators across the countries provided.
    
    #df.set_index("Country Name", inplace = True)
    df_bar_chart12 = df12.loc[selected_country_bar_chart]
    df_bar_chart22 = df22.loc[selected_country_bar_chart]
    # First Bar Chart for 2012
    df_bar_chart12 = df12.loc[selected_country_bar_chart]
    plt.figure(figsize=(10, 5))
    plt.bar(df_bar_chart12.index, df_bar_chart12['GDP_Growth'], color='skyblue')
    plt.title('GDP Growth (%) 2012')
    plt.ylabel('Percentage')
    plt.xlabel('Country')
    plt.savefig(f"images/bar_chart_2012.png", bbox_inches = 'tight', dpi=300)
    plt.show()
    
    # Second Bar Chart for 2022
    df_bar_chart22 = df22.loc[selected_country_bar_chart]
    plt.figure(figsize=(10, 5))
    plt.bar(df_bar_chart22.index, df_bar_chart22['GDP_Growth'], color='lightgreen')
    plt.title('GDP Growth (%) 2022')
    plt.ylabel('Percentage')
    plt.xlabel('Country')
    plt.savefig(f"images/bar_chart_2022.png", bbox_inches = 'tight', dpi=300)
    plt.show()




def create_elbow_plot(df_2022, selected_country_bar_chart):
    
    
    # Assuming df_2022 is your DataFrame and you're interested in "GDP_Growth" for clustering
   df22 = df_2022[["GDP_Growth"]]  # Ensuring it remains a DataFrame
   
   # Scaling the data
   scaler = StandardScaler()
   df_scaled = scaler.fit_transform(df22)
   
   # Reshaping the data is not necessary here as we are keeping it as a DataFrame
   
   # Calculating WCSS for a range of number of clusters
   wcss = []
   for i in range(1, min(len(df22), 11)):  # Adjusting to avoid exceeding the number of samples
       kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
       kmeans.fit(df_scaled)
       wcss.append(kmeans.inertia_)
   
   # Plotting the elbow plot
   plt.figure(figsize=(10, 6))
   plt.plot(range(1, min(len(df22), 11)), wcss, marker='o')
   plt.title('Elbow plot of GDP Growth For Optimal Number of Clusters')
   plt.xlabel('Number of Clusters')
   plt.ylabel('WCSS')
   plt.xticks(range(1, min(len(df22), 11)))
   plt.grid(True)
   plt.savefig(f"images/elbow_plot.png", bbox_inches = 'tight', dpi=300)
   plt.show()
    
    
# Create and display a correlation heatmap
def create_correlation_heatmap(data, year):
    
    """ Creating Correlation map 
     data: required data 
     year: concern year
     """
     
    plt.figure(figsize = (10, 8))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot = True, cmap = 'coolwarm', fmt = ".2f")
    plt.title(f"Correlation Heatmap of Indicators year {year}")
    if(year == 2012):
        plt.savefig(f"images/heatmap_2012.png", bbox_inches='tight', dpi=300)
    else:
        plt.savefig(f"images/heatmap_2022.png", bbox_inches='tight', dpi=300)
    plt.show()
    
    
# Create and display a correlation heatmap
def create_scatter_matrix(data, year):
    
    """
    create scatter matrix to analyse weak and strong relation between indicators
    """
    
    # scatter plot
    pd.plotting.scatter_matrix(data, figsize = (9.0, 9.0))
    plt.tight_layout() # helps to avoid overlap of labels
    if(year == 2012):
        plt.savefig(f"images/scatter_matrix_2012.png", bbox_inches = 'tight', dpi=300)
    else:
        plt.savefig(f"images/scatter_matrix_2022.png", bbox_inches = 'tight', dpi=300)
    plt.show()
    
    
# Perform K-Means clustering
def perform_clustering(data, cluster_countries, year):
    
    """
    create country clusters for two indicators depend on years
    """
    
    #print(cluster_countries[1])
    df_fit = data[cluster_countries].copy()
    df_fit, df_min, df_max = ct.scaler(df_fit)

    silhouette_scores = {}
    for ic in range(2, 7):
        kmeans = cluster.KMeans(n_clusters=ic)
        kmeans.fit(df_fit)
        labels = kmeans.labels_
        silhouette_scores[ic] = skmet.silhouette_score(df_fit, labels)

    # Print silhouette scores for each number of clusters
    for clusters, score in silhouette_scores.items():
        print(f"Silhouette Score for {clusters} clusters: {score}")

    # Perform clustering with desired number of clusters (nc)
    if(year == 2012):
        nc = 5
    else:
        nc = 2
    kmeans = cluster.KMeans(n_clusters=nc)
    kmeans.fit(df_fit)
    labels = kmeans.labels_
    cen = kmeans.cluster_centers_
    
    # Define cluster labels
    #cluster_labels = [f'Cluster {i+1}' for i in range(nc)]

    # Plotting
    plt.figure(figsize = (6.0, 6.0))
    #plt.scatter(df_fit[cluster_countries[0]], df_fit[cluster_countries[1]], c = labels, cmap = "tab10")
    #plt.scatter(cen[:, 0], cen[:, 1], c = "k", marker = "d", s = 80)
    for i in range(nc):
        plt.scatter(df_fit[labels == i][cluster_countries[0]], df_fit[labels == i][cluster_countries[1]], label=f'Cluster {i+1}')
    plt.scatter(cen[:, 0], cen[:, 1], c="black", marker="X", s=100, label='Centers')

    plt.xlabel(cluster_countries[0])
    plt.ylabel(cluster_countries[1])
    plt.legend()
    plt.title(f"country clusters based on Inflation & GDP Growth {year}")
    if(year == 2012):
        plt.savefig(f"images/cluster_2012.png", bbox_inches = 'tight', dpi = 300)
    else:
        plt.savefig(f"images/cluster_2022.png", bbox_inches = 'tight', dpi = 300)
    
    plt.show()

    # Display data for a specific cluster
    data["Cluster_Labels"] = labels
    cluster_data = data[data["Cluster_Labels"] == 1]
    data.to_excel(f"cluster_{year}.xlsx")
    #print("Data points in Cluster 4:")
    #print(cluster_data)
  
    
def exp_growth(t, scale, growth):
    
    """ Computes exponential function with scale and growth as free parameters
    """
    #f = scale * np.exp(growth * t)
    f = scale * np.exp(growth * (t-1950))
    return f


def create_data_fit_graph(df_fitted, indicator, country):
    
    """Create line graph with fitting data towards prediction for 10 years
    """

    initial_guess = [1.0, 0.02]
    popt, pcovar = opt.curve_fit(exp_growth, df_fitted["Year"], df_fitted[indicator], p0 = initial_guess, maxfev = 10000)
    print("Fit parameters:", popt)
    # Create a new column with the fitted values
    df_fitted["pop_exp"] = exp_growth(df_fitted["Year"], *popt)
    #plot
    plt.figure()
    plt.plot(df_fitted["Year"], df_fitted[indicator], label = "data")
    plt.plot(df_fitted["Year"], df_fitted["pop_exp"], label = "fit")
    plt.legend()
    plt.xlabel("Year")
    plt.ylabel(indicator)
    plt.title(f"{indicator} in {country} ")

    plt.savefig(f"images/{indicator}_{country}_fit_graph.png", bbox_inches = 'tight', dpi = 300)
    plt.show()
    print()
    # call function to calculate upper and lower limits with extrapolation
    # create extended year range
    years = np.linspace(2000, 2030)
    pop_exp_growth = exp_growth(years, *popt)
    sigma = err.error_prop(years, exp_growth, popt, pcovar)
    low = pop_exp_growth - sigma
    up = pop_exp_growth + sigma
    plt.figure()
    plt.title(f"{indicator} in {country} Forecast untill 2030 ")
    plt.plot(df_fitted["Year"], df_fitted[indicator], label = "data")
    plt.plot(years, pop_exp_growth, label = "Forecast")
    # plot error ranges with transparency
    plt.fill_between(years, low, up, alpha = 0.5, color = "y")
    plt.legend(loc = "upper right")
    plt.xlabel("Year")
    plt.ylabel(indicator)
    plt.savefig(f"images/{indicator}_{country}_predict_graph.png", bbox_inches = 'tight', dpi = 300)
    plt.show()
    # Predict future values
    pop_2030 = exp_growth(np.array([2030]), *popt)
    # Assuming you want predictions for the next 10 years
    sigma_2030 = err.error_prop(np.array([2030]), exp_growth, popt, pcovar)
    print(f"{indicator} in")
    print("2030:", exp_growth(2030, *popt) / 1.0e6, "Mill.")

    # for next 10 years
    print(f"{indicator} in")
    for year in range(2024, 2034):
        print(f"{indicator} in",year)
        print("2030:", exp_growth(year, *popt) / 1.0e6, "Mill.")


    
if __name__ == "__main__":
    # File paths
    file_paths = ['urban population.csv','rural population.csv','inflation.csv','GDP Growth.csv','Fuel imports.csv','Trade.csv']
    #selected_countries = ['Canada', 'China', 'Germany', 'United Kingdom', 'India', 'United States']
    selected_countries = ['Canada', 'China', 'Germany', 'United Kingdom', 'India', 'United States',
                          'Australia', 'France', 'Israel','Japan','Malaysia','Russian Federation',
                           'Turkiye','Spain','Pakistan'
                          ]
    selected_country_bar_chart = ['Canada', 'China', 'Germany', 'United Kingdom', 'India']
    year_2012 = 2012
    year_2022 = 2022
    start_year = 2000
    end_year = 2022

    #dataset of 2012
    df_12, dataframes_dict_transpose_12 = read_data(file_paths, selected_countries, year_2012)
    #print(dataframes_dict_transpose_12)
    #dataset of 2022
    df_22,  dataframes_dict_transpose_12 = read_data(file_paths, selected_countries, year_2022)
    
    #dataset fro 2012 to 2022
    country_india = "India"
    data_sets = ['inflation.csv', 'GDP Growth.csv',"Population.csv"]
    dataframes_dict = read_data_for_fit(data_sets, country_india, start_year, end_year)
    #data for turkey 
    country_turkiye = "Turkiye"
    dataframes_dict_turkiye = read_data_for_fit(data_sets, country_turkiye, start_year, end_year)
    #print(dataframes_dict)
    indicator_inflation = "inflation"
    indicator_gdp = "GDP_Growth"
    indicator_population = "Population"
    data_inflation = merg_data(dataframes_dict, indicator_inflation)
    data_inflation_turkiye = merg_data(dataframes_dict_turkiye, indicator_inflation)
    data_gdp = merg_data(dataframes_dict, indicator_gdp)
    data_gdp_turkiye = merg_data(dataframes_dict_turkiye, indicator_gdp)
    data_population = merg_data(dataframes_dict, indicator_population)
    data_population_turkiye = merg_data(dataframes_dict_turkiye, indicator_population)#
    
    #df_12.to_excel("df_12.xlsx") 
    
    # Datasets without country names
    df_indicators_2012 = df_12.iloc[:,1:]
    df_indicators_2022 = df_22.iloc[:,1:]
    
    create_bar_chart(df_indicators_2012, df_indicators_2022, selected_country_bar_chart)
    
    
    create_elbow_plot(df_indicators_2022, selected_country_bar_chart)
    

    #print(df_indicators)
    # Create correlation heatmap for the years 2012 and 2022
    create_correlation_heatmap(df_indicators_2012, year_2012)
    create_correlation_heatmap(df_indicators_2022, year_2022)
    
    
    #create scatter matrix for the years 2012 and 2022
    create_scatter_matrix(df_indicators_2012, year_2012)
    create_scatter_matrix(df_indicators_2022, year_2022)
    
    #creater country cluster for the years 2012 and 2022
    cluster_indicators = ["inflation", "GDP_Growth"]
    perform_clustering(df_12, cluster_indicators, year_2012)
    perform_clustering(df_22, cluster_indicators, year_2022)
#....................................................................................................
    #code for part to ..data fitting and prediction 
    create_data_fit_graph(data_inflation, indicator_inflation, country_india)
    create_data_fit_graph(data_inflation_turkiye, indicator_inflation, country_turkiye)
    
    indicator_gdp_growth = "GDP Growth"
    create_data_fit_graph(data_gdp, indicator_gdp_growth, country_india)
    create_data_fit_graph(data_gdp_turkiye, indicator_gdp_growth, country_turkiye)
    #create_data_fit_graph(data_population, indicator_population)
    
    
    
#...................................................................................
    #Basic descriptive statistics
    desc_stats = df_indicators_2012.describe()
    print("Descriptive Statistics:\n", desc_stats, "\n")
    
    # Calculating mean, median, standard deviation, skewness, kurtosis
    numeric_df = df_indicators_2012.select_dtypes(include=[float, int])  # Filter only numeric columns
    mean = numeric_df.mean()
    median = numeric_df.median()
    std_dev = numeric_df.std()
    skewness = numeric_df.skew()
    kurtosis = numeric_df.kurtosis()
    print("Mean:\n", mean, "\n")
    print("Median:\n", median, "\n")
    print("Standard Deviation:\n", std_dev, "\n")
    print("Skewness:\n", skewness, "\n")
    print("Kurtosis:\n", kurtosis, "\n")
    
    # Generating a correlation matrix for the numerical columns
    corr_matrix = numeric_df.corr()
    print("Correlation Matrix:\n", corr_matrix, "\n\n")


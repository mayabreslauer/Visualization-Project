from sqlalchemy import create_engine
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
import joypy
import seaborn as sns
from matplotlib.patches import Patch
import statsmodels.api as sm
import numpy as np

#Clusters File
def merge_eshkol_files():
    file_path = "מדד חברתי כלכלי 2015-יישובים בתוך מועצות אזוריות.xlsx"
    dataset11 = pd.read_excel(file_path, sheet_name=0, skiprows=0, header=[0, 1])
    dataset11.reset_index(drop=True, inplace=True)
    dataset11.columns = dataset11.columns.map(lambda x: x[0].strip() if isinstance(x[0], str) else x[0])
    selected_columns = ["שם יישוב", "ערך מדד\nVALUE INDEX", "אשכול\n(10 מ-1 עד)\nCLUSTER (1 to 10)"]
    dataset11 = dataset11[selected_columns]

    dataset11 = dataset11.loc[:, ~dataset11.columns.duplicated()]
    print(dataset11)

    new_column_names = {
        'שם יישוב': 'City_Name',
        'ערך מדד\nVALUE INDEX': 'Value_Index',
        'אשכול\n(10 מ-1 עד)\nCLUSTER (1 to 10)': 'Cluster'
    }
    dataset11.rename(columns=new_column_names, inplace=True)
    dataset11.to_excel("Clusters.xlsx", index=False)
# merge_eshkol_files()

#Religions File
def religions_file():
    dataset3 = pd.read_excel("Religions.xlsx", engine='openpyxl')
    dataset3['שם יישוב'] = dataset3['שם יישוב'].str.replace("אזור ", "")
    dataset3['שם יישוב'] = dataset3['שם יישוב'].str.split(' מ"א').str[0]
    dataset3['שם יישוב'] = dataset3['שם יישוב'].str.split(' של"ש').str[0]
    dataset3['שם יישוב'] = dataset3['שם יישוב'].str.replace("*", "")
    dataset3['שם יישוב'] = dataset3['שם יישוב'].drop_duplicates()
    dataset3 = dataset3[dataset3['שם יישוב'].notna() & (dataset3['שם יישוב'] != '')]
    dataset3.to_excel("religions_new.xlsx", index=False)
# religions_file()

def create_eshkol_table():
    data_frame = pd.read_excel("Clusters.xlsx")
    data_frame["City_Name"] = [item.rstrip() if isinstance(item, str) else item for item in data_frame["City_Name"]]
    print(data_frame)
    engine = create_engine('mysql+pymysql://root:mayaB1406!@localhost/visualization')
    data_frame.to_sql('eshkol', engine, if_exists='replace', index=False)
    engine.dispose()
# create_eshkol_table()

def creation_of_ages_table():
    data_frame = pd.read_excel("Ages.xlsx")
    data_frame["שם_ישוב"] = [item.rstrip() if isinstance(item, str) else item for item in data_frame["שם_ישוב"]]
    print(data_frame)
    engine = create_engine('mysql+pymysql://root:mayaB1406!@localhost/visualization')
    data_frame.to_sql('ages', engine, if_exists='replace', index=False)
    engine.dispose()
# creation_of_ages_table()

def creation_of_religions_table():
    data_frame = pd.read_excel("religions_new.xlsx")
    data_frame["שם_יישוב"] = [item.rstrip() if isinstance(item, str) else item for item in data_frame["שם_יישוב"]]
    print(data_frame)
    engine = create_engine('mysql+pymysql://root:mayaB1406!@localhost/visualization')
    data_frame.to_sql('religion', engine, if_exists='replace', index=False)
    engine.dispose()
#creation_of_religions_table()

def interactive_heatmap():
    data = pd.read_excel("dataset_with_coordinates.xlsx")

    age_groups = ['0_5', '6_18', '19_45', '46_55', '56_64', '65_']
    mapping = {1: "Jerusalem District", 2: "Northern District", 3: "Haifa District", 4: "Central District",
               5: "Judea and Samaria District", 6: "Southern District"}

    data['region'] = data['region'].map(mapping)
    grouped_data = data.groupby(['rank', 'region'], observed=False)[age_groups].mean().reset_index()

    fig = go.Figure()

    annotations = {}

    for age_group in age_groups:
        heatmap_data = grouped_data.pivot(index='region', columns='rank', values=age_group).fillna(0)
        zmin, zmax = heatmap_data.min().min(), heatmap_data.max().max()

        fig.add_trace(go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            text=[['%.2f' % val if not pd.isnull(val) else '' for val in row] for row in heatmap_data.values],
            hoverinfo='text',
            colorscale='Magma_r',  # Reversed Magma color scale
            colorbar=dict(title='Mean Value of Selected Age Group'),
            zmin=zmin,
            zmax=zmax,
            name=age_group,
            hovertemplate='Region: %{y}<br>Cluster: %{x}<br>Age Group: ' + age_group + '<br>Mean Value: %{z}<extra></extra>',
            visible= age_group == age_groups[0]

        ))

        annotations[age_group] = [
            dict(
                x=rank,
                y=region,
                text=str('%.2f' % heatmap_data.loc[region, rank]),
                font=dict(color='black' if heatmap_data.loc[region, rank] <= (zmin + zmax) / 3 else 'white'),
                # Set font color to black for values below the threshold, otherwise white
                showarrow=False,
                xref='x',
                yref='y',
                xanchor='center',
                yanchor='middle',

            )
            for rank in heatmap_data.columns
            for region in heatmap_data.index
        ]

    fig.add_annotation(
        text="Select an age group",
        x=0,
        y=-0.2,  # Adjusted y coordinate to bring the annotation inside the plot
        xref="x",
        yref="y",
        showarrow=False,
        font=dict(
            family="Arial",
            size=12,
            color="black"
        )
    )

    button_list = [
        dict(label=age_group,
             method="update",
             args=[{"visible": [age == age_group for age in age_groups]},
                   {"annotations": annotations[age_group]}]) for age_group in age_groups
    ]

    fig.update_layout(
        updatemenus=[
            dict(
                buttons=button_list,
                direction="down",
                showactive=True,
                x=1.3,
                xanchor="left",
                y=0.9,
                yanchor="top"
            ),
        ]
    )

    fig.update_layout(
        title="Interactive Heatmap of Region VS Cluster According to Selected Age Group",
        xaxis_title="Cluster",
        yaxis_title="Region",
        height=800,
        width=1000,
        margin=dict(l=200, r=100, t=100, b=150),  # Adjusted margin for bottom annotations
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.5)'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.5)'),
    )

    fig.update_layout(annotations=annotations[age_groups[0]])

    data_min = heatmap_data.min().min()
    data_max = heatmap_data.max().max()
    lower_third_threshold = data_min + (data_max - data_min) / 3

    fig.update_layout(annotations=[
        dict(
            x=annotation['x'],
            y=annotation['y'],
            text=annotation['text'],
            font=dict(color='black' if float(annotation['text']) <= lower_third_threshold else 'white',
                      size=12),
            showarrow=False,
            xref='x',
            yref='y',
            xanchor='center',
            yanchor='middle',
        )
        for annotation in annotations[age_groups[0]]
    ])

    fig.show()
interactive_heatmap()

def sanky():

    data = pd.read_excel("dataset_with_coordinates.xlsx")

    data = data[data['rank'].notna()]
    data = data[data['religion'].notna()]
    data = data[data['rank'] != 0]

    religion_mapping = {1: 'Judaism', 2: 'Islam', 4: 'Other'}

    data['religion'] = data['religion'].replace(religion_mapping)
    mapping = {1: "Jerusalem District", 2: "Northern District", 3: "Haifa District", 4: "Central District",
               5: "Judea and Samaria District", 6: "Southern District"}

    data['region'] = data['region'].map(mapping)
    # Convert all values in both columns to strings
    data['rank'] = data['rank'].astype(str)
    data['religion'] = data['religion'].astype(str)

    flows = data.groupby(['rank', 'religion', 'region']).size().reset_index(name='value')

    unique_clusters = sorted(data['rank'].unique())
    unique_religions = sorted(data['religion'].unique())
    unique_regions = sorted(data['region'].unique())

    color_clusters = 'rgba(191, 219, 255, 0.8)'  # Light blue
    color_religions = 'rgba(255, 204, 153, 0.8)'  # Light orange
    color_regions = 'rgba(204, 255, 204, 0.8)'  # Light green

    grouped_flows = flows.groupby(['rank', 'religion']).sum().reset_index()

    rank_religion_links = [
        {"source": unique_clusters.index(row['rank']),
         "target": len(unique_clusters) + unique_religions.index(row['religion']),
         "value": row['value'],
         "color": color_clusters} for _, row in grouped_flows.iterrows()
    ]

    grouped_flows = flows.groupby(['religion', 'region']).sum().reset_index()

    religion_region_links = [
        {"source": len(unique_clusters) + unique_religions.index(row['religion']),
         "target": len(unique_clusters) + len(unique_religions) + unique_regions.index(row['region']),
         "value": row['value'],
         "color": color_religions} for _, row in grouped_flows.iterrows()
    ]

    links = rank_religion_links + religion_region_links

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            label=[f'<span style="font-size:12px;">{cluster}</span>' for cluster in unique_clusters] +
                  [f'<span style="font-size:12px;">{religion}</span>' for religion in unique_religions] +
                  [f'<span style="font-size:12px;">{region}</span>' for region in unique_regions],
            color=[color_clusters] * len(unique_clusters) + [color_religions] * len(unique_religions) +
                  [color_regions] * len(unique_regions),  # Use different colors for clusters, religions, and regions
            pad=15,  # Increase padding
            thickness=40  # Increase thickness
        ),
        link=dict(
            source=[link['source'] for link in links],
            target=[link['target'] for link in links],
            value=[link['value'] for link in links],
            color=[link['color'] for link in links]  # Assign the colors of sources to the color attribute
        )  # Assign the links list to the link property with appropriate keys
    )])

    fig.add_annotation(
        x=0,  # Adjusted for three categories (cluster, religion, and region)
        y=1.05,
        text="Clusters",
        showarrow=False,
        font=dict(
            size=16,
            color="black"
        )
    )

    fig.add_annotation(
        x=0.5,  # Adjusted for three categories (cluster, religion, and region)
        y=1.05,
        text="Religions",
        showarrow=False,
        font=dict(
            size=16,
            color="black"
        )
    )

    fig.add_annotation(
        x=1,  # Adjusted for three categories (cluster, religion, and region)
        y=1.05,
        text="Regions",
        showarrow=False,
        font=dict(
            size=16,
            color="black"
        )
    )

    fig.update_layout(
        title="Flow between Clusters, Religions and Regions",
        font=dict(size=18)
    )

    fig.show()
# sanky()

def scatter_plot():

    data = pd.read_excel("מועצות אזוריות 2018.xlsx")
    plt.figure(figsize=(10, 6))
    plt.scatter(data['Percentage entitled to a bagrut certificate'], data['Gini index'], s=data['Total settelments']*5, alpha=0.5)
    plt.xlabel('Percentage Entitled to a Bagrut Certificate')
    plt.ylabel('Gini Index')
    plt.title('Percentage Entitled to a Bagrut Certificate and Total settelments Increases with The Gini Index')
    plt.grid(True)
    plt.show()
#scatter_plot()

### Data Preparation of Cordinates:
def data_preparation_of_coordinates():
    def get_coordinates(location):
        geolocator = Nominatim(user_agent="my_geocoder")
        location = geolocator.geocode(location)
        if location:
            return location.latitude, location.longitude
        else:
            return None, None
    dataset = pd.read_excel("dataset.xlsx")

    latitudes = []
    longitudes = []
    for city_name in dataset["City_Name"]:
        latitude, longitude = get_coordinates(city_name)
        latitudes.append(latitude)
        longitudes.append(longitude)
    dataset["Latitude"] = latitudes
    dataset["Longitude"] = longitudes
    dataset.to_excel("dataset_with_coordinates.xlsx", index=False)
# data_preparation_of_coordinates()
## the rest of interactive map creation is by using Tableau

### Social Economic Score by Region:
def social_economic_score_by_region():
    dataset = pd.read_excel("dataset_with_coordinates.xlsx")
    region_names = {
        1: "Jerusalem District",
        2: "Northern District",
        3: "Haifa District",
        4: "Central District",
        5: "Judea and Samaria Area",
        6: "Southern District"
    }
    region_data = [dataset[dataset['region'] == i]['socio_ec'] for i in range(1, 7)]
    fig, axes = joypy.joyplot(region_data, labels=[region_names[i] for i in range(1, 7)])
    for ax in axes:
        for line in ax.get_lines():
            peak_value = line.get_ydata().max()
            if peak_value > 0:
                peak_index = line.get_ydata().argmax()
                peak_x = line.get_xdata()[peak_index]
                ax.annotate(f'{peak_value:.2f}', xy=(peak_x, peak_value), xytext=(peak_x + 1, peak_value),
                            arrowprops=dict(arrowstyle='->'))
    plt.xlabel('Socio-Economic Score')
    plt.ylabel('Region')
    plt.title('Balancing Socio-Economic Disparities in the South Region, Haifa Region, and Central Region')

    plt.show()
# social_economic_score_by_region()

### Population Distribution by Age Groups and Religion in the Town
def box_plot():
    dataset_with_coordinates = pd.read_excel('dataset_with_coordinates.xlsx')
    dataset_with_coordinates = dataset_with_coordinates[(dataset_with_coordinates['religion'] == 1) | (dataset_with_coordinates['religion'] == 2)]
    religion_labels = {1: 'Jewish', 2: 'Muslim'}
    plot_colors = sns.color_palette("colorblind")
    melted_df = pd.melt(dataset_with_coordinates, id_vars=['religion'], value_vars=['0_5', '6_18', '19_45', '46_55', '56_64', '65_'], var_name='age_group', value_name='value')
    melted_df['age_group'] = melted_df['age_group'].str.replace('_', '-')
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='age_group', y='value', hue='religion', data=melted_df, palette=plot_colors)
    plt.xticks(ha='left')
    plt.xlabel('Age Group')
    plt.ylabel('Number of People')
    legend_labels = [religion_labels[label] for label in melted_df['religion'].unique()]
    legend_handles = [Patch(facecolor=plot_colors[int(label)-1], edgecolor='black') for label in melted_df['religion'].unique()]
    plt.legend(title='Religion', labels=legend_labels, handles=legend_handles, facecolor='lightgray')
    plt.title('Age 0-45 Distribution Disparity Between Arab and Jewish Settlements')
    plt.show()
# box_plot()

def glm_regression():
    #GLM regression
    dataset = pd.read_excel('dataset_with_coordinates.xlsx')
    religion_1_data = dataset[dataset['religion'] == 1]
    religion_2_data = dataset[dataset['religion'] == 2]
    poisson_model_1 = sm.GLM(religion_1_data['6_18'], sm.add_constant(religion_1_data['socio_ec']), family=sm.families.Poisson())
    poisson_results_1 = poisson_model_1.fit()
    poisson_model_2 = sm.GLM(religion_2_data['6_18'], sm.add_constant(religion_2_data['socio_ec']), family=sm.families.Poisson())
    poisson_results_2 = poisson_model_2.fit()

    # Display parameter estimates for religion 1
    print("Parameter Estimates for Religion 1:")
    print(poisson_results_1.summary())

    # Display parameter estimates for religion 2
    print("\nParameter Estimates for Religion 2:")
    print(poisson_results_2.summary())

    def predicted_values(constant, slope, x):
        return np.exp(constant + slope * x)
    x_values = np.linspace(religion_1_data['socio_ec'].min(), religion_1_data['socio_ec'].max(), 100)
    predicted_values_religion_1 = predicted_values(4.6166, -0.2699, x_values)
    predicted_values_religion_2 = predicted_values(4.4631, -0.6842, x_values)
    sns.set_palette("colorblind")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='socio_ec', y='0_5', data=religion_1_data, label='Jewish', marker='o', edgecolor='none')
    plt.plot(x_values, predicted_values_religion_1, color='blue', linestyle='-', linewidth=2)
    sns.scatterplot(x='socio_ec', y='0_5', data=religion_2_data, label='Muslim', marker='o', edgecolor='none')
    plt.plot(x_values, predicted_values_religion_2, color='orange', linestyle='-', linewidth=2)# Plot data and regression line for religion 2
    plt.title('Differential Impact of Socioeconomic Index on Ages 0-5: Arab vs. Jewish Communities')
    plt.xlabel('Social Economic Index')
    plt.ylabel('Number of Children between 0-5')
    plt.legend(title='Religion', loc='upper right')
    plt.show()

    def predicted_values(constant, slope, x):
        return np.exp(constant + slope * x)

    x_values = np.linspace(religion_1_data['socio_ec'].min(), religion_1_data['socio_ec'].max(), 100)
    predicted_values_religion_1 = predicted_values(5.6098, -0.0964, x_values)
    predicted_values_religion_2 = predicted_values( 5.6243, -0.4116, x_values)
    sns.set_palette("colorblind")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='socio_ec', y='6_18', data=religion_1_data, label='Jewish', marker='o', edgecolor='none')
    plt.plot(x_values, predicted_values_religion_1, color='blue', linestyle='-', linewidth=2)
    sns.scatterplot(x='socio_ec', y='6_18', data=religion_2_data, label='Muslim', marker='o', edgecolor='none')
    plt.plot(x_values, predicted_values_religion_2, color='orange', linestyle='-', linewidth=2)
    plt.title('Differential Impact of Socioeconomic Index on Ages 6-18: Arab vs. Jewish Communities')
    plt.xlabel('Social Economic Index')
    plt.ylabel('Number of Children between 6-18')
    plt.legend(title='Religion', loc='upper right')
    plt.show()
# glm_regression()

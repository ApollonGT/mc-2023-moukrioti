import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from streamlit_option_menu import option_menu

from pathlib import Path
from PIL import Image

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

st.markdown(
    """
    <style>
    .main {
    background-color: #FFFFF0;
    color: #333333;
    }
    .css-xx {
    font-size: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


with st.sidebar:
    selected=option_menu(
        menu_title="Menu",
        options=["Project Overview","Exploratory Analysis","Clustering","Classification","Regression"],
        icons=["clipboard2-check","bar-chart-fill","people-fill","diagram-3-fill","graph-up"],
        menu_icon="list",
        default_index=0
    )


# Load preprocessed data

@st.cache_data
def load_data(filename):
    data = pd.read_csv(filename)
    return data

# Load preprocessed plots

def load_plot(filename):
    return plt.imread(filename)

# Project and Dataset Overview Page

if selected =="Project Overview":
    st.title(f"Welcome to my Thesis Dashboard: Consumer and Product Analysis in Business Desicion-Making")

    st.header(f"Project Overview")
    st.markdown('* The aim of this paper is to examine how consumer and product analysis using big data contributes significantly to making good business decisions.')
    st.markdown('* So first, we will apply exploratory analysis to identify hidden patterns and correlations between variables.')
    st.markdown('* Then, we will apply clustering techniques to group the data based on specific attributes and then we will apply classification algorithms to classify the new data into the already predefined groups.')
    st.markdown('* And finally, we will apply regression analysis to identify the factors that influence sales and also to predict the sales of new products.')

    st.header(f"Dataset Overview")
    st.subheader(f"Global Super Store Dataset")
    st.markdown('* The dataset used in this paper was found on Kaggle and contains details of online orders made by people around the world between 2011 and 2014.')

    Global_Superstore2_csv = Path(__file__).parents[1] / 'Dashboard' / 'Global_Superstore2.csv'
    project_data= pd.read_csv(Global_Superstore2_csv, encoding = 'latin-1')
    
    st.write(project_data.head())

    st.subheader(f"New Features")
    st.text('The features I created to assist with flow of analysis:')

    st.markdown('* **gender:** The new gender variable refers to the gender of the consumer and will be extracted from the Customer Name variable, which already exists in the dataset and contains the names of the customers.')
    st.markdown('* **Order_year:** The new Order_year variable refers to the year the order was submitted and will be extracted from the Order Date variable, which already exists in the dataset.')
    st.markdown('* **Order_month:** The new Order_month variable refers to the month the order was submitted and will be extracted from the Order Date variable, which already exists in the dataset.')
    st.markdown('* **Order_day:** The new Order_day variable refers to the day the order was submitted and will be extracted from the Order Date variable, which already exists in the dataset.')


# Exploratory Analysis Page 


if selected =="Exploratory Analysis":
    st.title(f"Exploratory Analysis")
    
    new_df_csv = Path(__file__).parents[1] / 'Dashboard' / 'new_df.csv'
    exploratory_data= load_data(new_df_csv)
    
    numeric_data = exploratory_data[['Sales','Quantity','Discount','Profit','Shipping Cost']]
    qualitative_data = exploratory_data[['Ship Mode','Segment','Region','Category','Sub-Category','gender','Order Priority']]

    # For Quantitative variables

    st.header(f"Quantitative Variables")

    # Dropdown to select quantitative variable

    selected_numeric_data = st.selectbox("Select Quantitative Variable", numeric_data.columns)
    
    if selected_numeric_data:
        st.subheader(f"{selected_numeric_data} Distribution")
        if st.checkbox("Show Histogram"):
            plt.figure(figsize=(8, 6))
            sns.histplot(exploratory_data[selected_numeric_data], bins=20, color='magenta', edgecolor='black', kde=True)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
        if st.checkbox("Show Boxplot"):
            plt.figure(figsize=(8, 6))
            sns.boxplot(data=exploratory_data, x=selected_numeric_data, color='skyblue')
            st.pyplot()

    st.subheader(f"Correlation Matrix")

    Correlation_Matrix_plot = 'Dashboard'/ 'Correlation Matrix.png'
    correlation_plot = Image.open(Correlation_Matrix_plot)

    # correlation_plot = load_plot(r'C:\Users\ελισαβετ\Documents\thesis\mc-2023-moukrioti\Dashboard\Correlation Matrix.png')
    st.image(correlation_plot, use_column_width=True)

    # For Qualitative variables
    st.header(f"Qualitative Variables")
    selected_qualitative_data = st.selectbox("Select Qualitative Variables", qualitative_data.columns)

    # Display Frequency Plot and Cross-tabulation based on selection

    if selected_qualitative_data:
        st.subheader(f"{selected_qualitative_data} Distribution")
        if st.checkbox("Show Frequency Plot"):
            plt.figure(figsize=(8, 6))
            sns.countplot(data=exploratory_data, x=selected_qualitative_data, color='violet')
            plt.xticks(rotation=45)
            st.pyplot()
        if st.checkbox("Show Cross-tabulation"):
            st.write(pd.crosstab(index=exploratory_data[selected_qualitative_data], columns="count"))

    st.header(f"Product and Consumer Analysis")

    # Sales by year

    st.subheader(f"Sales by Year")
    sales_year_plot = load_plot(r'C:\Users\ελισαβετ\Documents\thesis\mc-2023-moukrioti\Dashboard\Sales_year.png')
    st.image(sales_year_plot, use_column_width=True)


    # Sales by month of each year
    st.subheader(f"Sales by Month of Each Year")
    sales_month_year_plot =load_plot(r'C:\Users\ελισαβετ\Documents\thesis\mc-2023-moukrioti\Dashboard\Sales_month_year.png')
    st.image(sales_month_year_plot, use_column_width=True)

    # Sales by Month
    st.subheader(f"Sales by Month")
    sales_month_plot = load_plot(r'C:\Users\ελισαβετ\Documents\thesis\mc-2023-moukrioti\Dashboard\Sales_month.png')
    st.image(sales_month_plot, use_column_width=True)

    #Sales by Day 
    st.subheader(f"Sales by Day")
    sales_day_plot = load_plot(r'C:\Users\ελισαβετ\Documents\thesis\mc-2023-moukrioti\Dashboard\Sales_day.png')
    st.image(sales_day_plot, use_column_width=True)

    # Interactive plot for sales by year, month,day

    # Create the variables Order_year, Order_month, Order_day from Order Date


    exploratory_data['Order Date'] = pd.to_datetime(exploratory_data['Order Date'])


    exploratory_data['Order_year'] = exploratory_data['Order Date'].dt.year
    exploratory_data['Order_month'] = exploratory_data['Order Date'].dt.month
    exploratory_data['Order_day'] = exploratory_data['Order Date'].dt.day

    # Group by year, month, and day


    sales_by_year = exploratory_data.groupby('Order_year')['Sales'].sum()
    sales_by_month = exploratory_data.groupby('Order_month')['Sales'].sum()
    sales_by_day = exploratory_data.groupby('Order_day')['Sales'].sum()

    st.subheader(f"Interactive Plot for Sales by Year, Sales by Month and Sales by Day")

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                    subplot_titles=('Sales by Year', 'Sales by Month', 'Sales by Day'))
    fig.add_trace(go.Bar(x=sales_by_year.index, y=sales_by_year.values, name='Sales by Year'), row=1, col=1)
    fig.add_trace(go.Bar(x=sales_by_month.index, y=sales_by_month.values, name='Sales by Month'), row=2, col=1)
    fig.add_trace(go.Bar(x=sales_by_day.index, y=sales_by_day.values, name='Sales by Day'), row=3, col=1)
    fig.update_layout(
    updatemenus=[
        dict(
            buttons=list([
                dict(label="Year",
                     method="update",
                     args=[{"visible": [True, False, False]},
                           {"title": "Sales by Year"}]),
                dict(label="Month",
                     method="update",
                     args=[{"visible": [False, True, False]},
                           {"title": "Sales by Month"}]),
                dict(label="Day",
                     method="update",
                     args=[{"visible": [False, False, True]},
                           {"title": "Sales by Day"}]),
                        ]),
                        direction="down",
                        pad={"r": 10, "t": 10},
                        showactive=True,
                        x=0.1,
                        xanchor="left",
                        y=1.15,
                        yanchor="top"
            ),
        ]
    )
    st.plotly_chart(fig)


    # Which country has top 5 sales?

    st.subheader(f"Which countries contribute top 5 sales?")
    sales_countries_plot =load_plot(r'C:\Users\ελισαβετ\Documents\thesis\mc-2023-moukrioti\Dashboard\5Sales_countries.png')
    st.image(sales_countries_plot, use_column_width=True, caption="Top 5 Sales Countries")

    sales_countries_percentage_plot = load_plot(r'C:\Users\ελισαβετ\Documents\thesis\mc-2023-moukrioti\Dashboard\5Sales_countries_percentage.png')
    st.image(sales_countries_percentage_plot, use_column_width=True, caption= "Top 5 Sales Countries - Percentages")

    # Which are the top 6 profit-making product types on a yearly basis?

    st.subheader(f"Which are the 6 most profitable product types on a yearly basis?")
    profit_products_plot = load_plot(r'C:\Users\ελισαβετ\Documents\thesis\mc-2023-moukrioti\Dashboard\6Profit_products.png')
    st.image(profit_products_plot, use_column_width=True, caption="Top 6 products every year based on profit")

    # How the customers are distributed across the countries?

    st.subheader(f"How the customers are distributed across the countries?")

    country_group = exploratory_data.groupby(['Country'])
    customer_distribution = country_group.agg({'Customer ID':'count'})
    customer_distribution.columns = ['Customer_count']
    customer_distribution.reset_index(inplace=True)
    country_map = dict(type='choropleth',
           locations=customer_distribution['Country'],
           locationmode='country names',
           z=customer_distribution['Customer_count'],
            reversescale = True,
           text=customer_distribution['Country'],
           colorscale='rainbow',
           colorbar={'title':'Customer Count'})
    layout = dict(title='Customer Distribution over Countries',
             geo=dict(showframe=False,projection={'type':'mercator'}))
    choroplot = go.Figure(data = [country_map],layout = layout)
    st.plotly_chart(choroplot)
    
                            
                                 
#Clustering Page

if selected =="Clustering":
    st.title(f"Clustering")

    # Display elbow method plot
    st.subheader(f"Elbow Method Plot")
    elbow_plot = load_plot(r'C:\Users\ελισαβετ\Documents\thesis\mc-2023-moukrioti\Dashboard\Elbow method.png')
    st.image(elbow_plot, use_column_width=True)

    # Display PCA plot
    st.subheader(f"3D Plot")
    pca_plot = load_plot(r'C:\Users\ελισαβετ\Documents\thesis\mc-2023-moukrioti\Dashboard\3D Plot.png')
    st.image(pca_plot, use_column_width=True, caption="Plot with xlabel=PC1, ylabel=PC2, zlabel=PC3 and color= Cluster")

# Classification Page

if selected =="Classification":
    st.title(f"Classification")

    combined_data_csv = Path(__file__).parents[1] / 'Dashboard' / 'combined_data.csv'
    classification_data= load_data(combined_data_csv)

    #classification_data =load_data(r'C:\Users\ελισαβετ\Documents\thesis\mc-2023-moukrioti\Dashboard\combined_data.csv')

    st.header(f"Random Forest for Classification")

    # Variables for classification model (Category, Sub-Category, Region, Shipping Cost)

    selected_variables = [c for c in classification_data.columns 
                      if c.startswith('Category') or c.startswith('Sub-Category') or c.startswith('Region') ]
    selected_variables.append('Shipping Cost')
    X = classification_data[selected_variables]
    Y = classification_data['Cluster']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=24)
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=24)
    rf_classifier.fit(X_train, Y_train)
    y_pred = rf_classifier.predict(X_test)
    feature_importance_rf = rf_classifier.feature_importances_
    feature_names = X.columns
    sorted_indices = np.argsort(feature_importance_rf)[::-1]
    sorted_features = [feature_names[i] for i in sorted_indices]
    sorted_importances = feature_importance_rf[sorted_indices]
    # Create Plotly figure
    fig = go.Figure(go.Bar(x=sorted_features, y=sorted_importances, marker=dict(color='magenta')))
    fig.update_layout(title='Feature Importance Plot',
                      xaxis_title='Feature',
                      yaxis_title='Importance',
                      height=600)
    
    # Display the Plotly figure in the Streamlit app
    st.plotly_chart(fig)


# Regression Page

if selected =="Regression":
    st.title(f"Regression")

    encoded_data1_csv = Path(__file__).parents[1] / 'Dashboard' / 'encoded_data1.csv'
    regression_data= load_data(encoded_data1_csv)

    #regression_data =load_data(r'C:\Users\ελισαβετ\Documents\thesis\mc-2023-moukrioti\Dashboard\encoded_data1.csv')

    st.header(f"Random Forest for Regression")

    # Variables for regression model

    X= regression_data[['Profit','Discount','Quantity','Shipping Cost','Category_Technology','Category_Office Supplies','Category_Furniture','Order_year']]
    y=regression_data['Sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=24)
    rf_regressor.fit(X_train, y_train)
    y_pred = rf_regressor.predict(X_test)
    feature_importance_rf = rf_regressor.feature_importances_
    feature_names = X.columns
    sorted_indices = np.argsort(feature_importance_rf)[::-1]
    sorted_features = [feature_names[i] for i in sorted_indices]
    sorted_importances = feature_importance_rf[sorted_indices]
    fig = go.Figure(go.Bar(x=sorted_features, y=sorted_importances, marker=dict(color='magenta')))
    fig.update_layout(title='Feature Importance Plot',
                      xaxis_title='Feature',
                      yaxis_title='Importance',
                      height=600)
    
    # Display the Plotly figure in the Streamlit app
    st.plotly_chart(fig)

    # Evaluation of the model with interactive plot for predicted vs actual values

    st.header(f"Evaluation of the model")

    trace = go.Scatter(x=y_test, y=y_pred, mode='markers', marker=dict(color='magenta'))
    diag_line = go.Scatter(x=[min(y_test), max(y_test)], y=[min(y_test), max(y_test)], mode='lines', name='Perfect Predictions', line=dict(color='cyan'))
    layout = go.Layout(title='Predicted vs Actual Values',
                   xaxis=dict(title='Actual Values'),
                   yaxis=dict(title='Predicted Values'),
                   showlegend=True)
    fig = go.Figure(data=[trace, diag_line], layout=layout)
    hover_text = [f'Actual: {a}<br>Predicted: {p}' for a, p in zip(y_test, y_pred)]
    fig.data[0].update(text=hover_text, hoverinfo='text')
    st.plotly_chart(fig)

    # Predicted vs Actual Values for new products

    st.header(f"Predicted vs Actual Sales for New Products")
    newproducts_sales_plot = load_plot(r'C:\Users\ελισαβετ\Documents\thesis\mc-2023-moukrioti\Dashboard\newproducts_sales.png')
    st.image(newproducts_sales_plot, use_column_width=True, caption="Predicted Sales for Five New Products ID vs Actual Sales")


     





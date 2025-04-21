import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import base64
from PIL import Image
import io
import sys
import traceback

# Load the data
try:
    data = pd.read_csv(r'E:\Data-Visualization-Project\notebooks\diabetic_data.csv')
    print("Data loaded successfully")
except Exception as e:
    print(f"Error loading data: {str(e)}")
    data = pd.DataFrame()

# Load model metrics
try:
    metrics_path = r'E:\Data-Visualization-Project\src\models\visualizations\model_metrics.csv'
    model_metrics = pd.read_csv(metrics_path)
    print("Model metrics loaded successfully")
except Exception as e:
    print(f"Error loading model metrics: {str(e)}")
    model_metrics = pd.DataFrame()

# Function to load and encode images
def load_image(file_path):
    try:
        print(f"Attempting to load image from: {file_path}")
        if os.path.exists(file_path):
            with open(file_path, 'rb') as img_file:
                encoded = base64.b64encode(img_file.read()).decode('ascii')
                print(f"Successfully loaded image from: {file_path}")
                return f'data:image/png;base64,{encoded}'
        else:
            print(f"File not found: {file_path}")
            return None
    except Exception as e:
        print(f"Error loading image {file_path}: {str(e)}")
        return None

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Define the layout
app.layout = html.Div([
    # Navbar
    html.Nav([
        html.Div([
            html.H1("Diabetic Readmission Analysis", style={'color': 'white', 'margin': '0 20px'}),
            html.Ul([
                html.Li(dcc.Link('EDA', href='/eda', style={'margin': '0 20px'})),
                html.Li(dcc.Link('Model Visualization', href='/model-viz', style={'margin': '0 20px'})),
                html.Li(dcc.Link('Predictions', href='/predictions', style={'margin': '0 20px'})),
                html.Li(dcc.Link('Recommendations', href='/recommendations', style={'margin': '0 20px'}))
            ], style={
                'list-style': 'none',
                'display': 'flex',
                'margin': '0',
                'padding': '0',
                'align-items': 'center'
            })
        ], style={
            'display': 'flex',
            'align-items': 'center',
            'justify-content': 'space-between',
            'width': '100%',
            'max-width': '1200px',
            'margin': '0 auto'
        })
    ], style={
        'background-color': '#333',
        'padding': '15px',
        'margin-bottom': '20px',
        'box-shadow': '0 2px 4px rgba(0,0,0,0.1)'
    }),
    
    # Content
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content', style={'max-width': '1200px', 'margin': '0 auto', 'padding': '20px'})
])

# EDA Page Layout
eda_layout = html.Div([
    html.H2("Exploratory Data Analysis", style={'margin-bottom': '20px'}),
    
    # Data Overview
    html.Div([
        html.H3("Data Overview"),
        html.Div([
            html.P(f"Number of rows: {len(data)}"),
            html.P(f"Number of columns: {len(data.columns)}"),
            html.P(f"Columns: {', '.join(data.columns)}")
        ])
    ], style={'margin-bottom': '30px'}),
    
    # Missing Values Analysis
    html.Div([
        html.H3("Missing Values Analysis"),
        dcc.Graph(id='missing-values-plot')
    ], style={'margin-bottom': '30px'}),
    
    # Readmission Analysis
    html.Div([
        html.H3("Readmission Analysis"),
        dcc.Graph(id='readmission-distribution-plot'),
        dcc.Graph(id='readmission-categories-plot')
    ], style={'margin-bottom': '30px'}),
    
    # Medical Conditions Analysis
    html.Div([
        html.H3("Medical Conditions Analysis"),
        dcc.Graph(id='condition-readmission-plot')
    ], style={'margin-bottom': '30px'}),
    
    # Primary Diagnosis Analysis
    html.Div([
        html.H3("Primary Diagnosis Analysis"),
        dcc.Graph(id='diagnosis-distribution-plot'),
        dcc.Graph(id='diagnosis-readmission-plot')
    ], style={'margin-bottom': '30px'}),
    
    # Discharge Analysis
    html.Div([
        html.H3("Discharge Analysis"),
        dcc.Graph(id='discharge-readmission-plot'),
        dcc.Graph(id='admission-type-plot')
    ], style={'margin-bottom': '30px'}),
    
    # Previous Admissions Analysis
    html.Div([
        html.H3("Previous Admissions Analysis"),
        dcc.Graph(id='prev-admission-readmission-plot')
    ], style={'margin-bottom': '30px'}),
    
    # Readmission Distribution
    html.Div([
        html.H3("Readmission Distribution"),
        dcc.Graph(
            figure=px.pie(
                data,
                names='readmitted',
                title='Readmission Status Distribution'
            )
        )
    ], style={'margin-bottom': '30px'}),
    
    # Demographic Analysis
    html.Div([
        html.H3("Demographic Analysis"),
        dcc.Dropdown(
            id='demographic-dropdown',
            options=[
                {'label': 'Race', 'value': 'race'},
                {'label': 'Gender', 'value': 'gender'},
                {'label': 'Age', 'value': 'age'}
            ],
            value='race',
            style={'width': '50%', 'margin-bottom': '20px'}
        ),
        dcc.Graph(id='demographic-plot')
    ], style={'margin-bottom': '30px'}),
    
    # Medical Features
    html.Div([
        html.H3("Medical Features"),
        dcc.Dropdown(
            id='medical-dropdown',
            options=[
                {'label': 'Time in Hospital', 'value': 'time_in_hospital'},
                {'label': 'Number of Medications', 'value': 'num_medications'},
                {'label': 'Number of Procedures', 'value': 'num_procedures'}
            ],
            value='time_in_hospital',
            style={'width': '50%', 'margin-bottom': '20px'}
        ),
        dcc.Graph(id='medical-plot')
    ]),

    # Correlation Heatmap
    html.Div([
        html.H3("Feature Correlations"),
        dcc.Graph(
            figure=px.imshow(
                data.select_dtypes(include=['float64', 'int64']).corr().round(2),  # Round to 2 decimal places
                title='Correlation Matrix of Numerical Features',
                color_continuous_scale='RdBu',
                zmin=-1,
                zmax=1,
                text_auto=True,  # Show correlation values
                aspect="auto",   # Make the plot fill the available space
                height=600,      # Medium height
                width=800        # Medium width
            ).update_layout(
                font=dict(size=10),  # Adjust font size for better readability
                margin=dict(l=50, r=50, t=50, b=50),  # Add margins
                xaxis=dict(tickangle=45),  # Rotate x-axis labels
                yaxis=dict(tickangle=0)    # Keep y-axis labels horizontal
            )
        )
    ], style={'margin-bottom': '30px', 'overflow': 'auto'})  # Add scrolling for large plots
])

# Model Visualization Page Layout
model_viz_layout = html.Div([
    html.H2("Model Visualizations", style={'margin-bottom': '20px'}),
    
    # Model Comparison
    html.Div([
        html.H3("Model Comparison"),
        html.Img(
            src=load_image(r'E:\Data-Visualization-Project\src\models\visualizations\model_comparison.png'),
            style={'width': '100%', 'margin-bottom': '20px'}
        )
    ]),
    
    # Model Selection and Metrics
    html.Div([
        html.H3("Model Performance Metrics"),
        dcc.Dropdown(
            id='model-dropdown',
            options=[
                {'label': 'Logistic Regression', 'value': 'logistic_regression'},
                {'label': 'Random Forest', 'value': 'random_forest'},
                {'label': 'Gradient Boosting', 'value': 'gradient_boosting'}
            ],
            value='logistic_regression',
            style={'width': '50%', 'margin-bottom': '20px'}
        ),
        html.Div(id='model-metrics')
    ]),
    
    # Model Visualizations
    html.Div([
        html.H3("Model Performance Visualizations"),
        html.Div([
            html.Div([
                html.H4("Confusion Matrix"),
                html.Img(id='confusion-matrix', style={'width': '100%'})
            ], style={'width': '33%', 'display': 'inline-block', 'padding': '10px'}),
            html.Div([
                html.H4("ROC Curve"),
                html.Img(id='roc-curve', style={'width': '100%'})
            ], style={'width': '33%', 'display': 'inline-block', 'padding': '10px'}),
            html.Div([
                html.H4("PR Curve"),
                html.Img(id='pr-curve', style={'width': '100%'})
            ], style={'width': '33%', 'display': 'inline-block', 'padding': '10px'})
        ])
    ]),
    
    # Feature Importance
    html.Div([
        html.H3("Feature Importance"),
        html.Img(id='feature-importance', style={'width': '100%'})
    ])
])

# Prediction Page Layout
prediction_layout = html.Div([
    html.H2("Readmission Risk Prediction", style={'margin-bottom': '20px'}),
    
    # Model Selection and Risk Overview
    html.Div([
        html.Div([
            html.H3("Select Model for Prediction"),
            dcc.Dropdown(
                id='prediction-model-dropdown',
                options=[
                    {'label': 'Logistic Regression', 'value': 'logistic_regression'},
                    {'label': 'Random Forest', 'value': 'random_forest'},
                    {'label': 'Gradient Boosting', 'value': 'gradient_boosting'},
                    {'label': 'Best Model', 'value': 'best_model'}
                ],
                value='best_model',
                style={'width': '100%', 'margin-bottom': '20px'}
            )
        ], style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'top'}),
        
        html.Div([
            html.H3("Risk Level Distribution"),
            dcc.Graph(id='risk-level-pie')
        ], style={'width': '70%', 'display': 'inline-block', 'padding-left': '20px'})
    ], style={'margin-bottom': '30px'}),
    
    # Risk Threshold and Distribution
    html.Div([
        html.Div([
            html.H3("Risk Threshold Adjustment"),
            html.Div([
                dcc.Slider(
                    id='risk-threshold-slider',
                    min=0,
                    max=1,
                    step=0.01,
                    value=0.5,
                    marks={i/10: str(i/10) for i in range(11)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], style={'width': '100%', 'margin': '20px 0'}),
            html.Div(id='threshold-metrics')
        ], style={'width': '40%', 'display': 'inline-block', 'vertical-align': 'top'}),
        
        html.Div([
            html.H3("Probability Distribution"),
            dcc.Graph(id='prediction-distribution')
        ], style={'width': '60%', 'display': 'inline-block', 'padding-left': '20px'})
    ], style={'margin-bottom': '30px'}),
    
    # Patient Risk Analysis
    html.Div([
        html.H3("Patient Risk Analysis"),
        html.Div([
            html.Div([
                html.H4("High Risk Patients", style={'color': '#dc3545'}),
                html.Div(id='high-risk-patients', style={'padding': '15px', 'background': '#fff5f5', 'border-radius': '5px'})
            ], style={'width': '33%', 'display': 'inline-block', 'padding': '10px'}),
            html.Div([
                html.H4("Medium Risk Patients", style={'color': '#fd7e14'}),
                html.Div(id='medium-risk-patients', style={'padding': '15px', 'background': '#fff9f0', 'border-radius': '5px'})
            ], style={'width': '33%', 'display': 'inline-block', 'padding': '10px'}),
            html.Div([
                html.H4("Low Risk Patients", style={'color': '#28a745'}),
                html.Div(id='low-risk-patients', style={'padding': '15px', 'background': '#f0fff4', 'border-radius': '5px'})
            ], style={'width': '33%', 'display': 'inline-block', 'padding': '10px'})
        ]),
        dcc.Graph(id='risk-level-bar')
    ], style={'margin-bottom': '30px'}),
    
    # Add a container for prediction results
    html.Div(id='prediction-results', style={'margin-top': '20px'})
])

# Recommendations Page Layout
recommendations_layout = html.Div([
    html.H2("Resource Allocation Recommendations", style={'margin-bottom': '20px'}),
    
    # Resource Optimization Overview
    html.Div([
        html.H3("Resource Optimization Overview", style={'margin-bottom': '15px'}),
        html.Div([
            html.Div([
                html.H4("High Risk Patient Care", style={'color': '#dc3545'}),
                html.Div([
                    html.P("Recommended Resource Allocation", style={'fontWeight': 'bold'}),
                    html.Ul([
                        html.Li("Intensive monitoring protocols"),
                        html.Li("Frequent follow-up schedule"),
                        html.Li("Specialized care coordination")
                    ])
                ], style={'padding': '15px', 'background': '#fff5f5', 'borderRadius': '5px'})
            ], style={'width': '30%', 'display': 'inline-block', 'padding': '10px', 'verticalAlign': 'top'}),
            
            html.Div([
                html.H4("Medium Risk Patient Care", style={'color': '#fd7e14'}),
                html.Div([
                    html.P("Recommended Resource Allocation", style={'fontWeight': 'bold'}),
                    html.Ul([
                        html.Li("Regular monitoring schedule"),
                        html.Li("Standard follow-up care"),
                        html.Li("Preventive interventions")
                    ])
                ], style={'padding': '15px', 'background': '#fff9f0', 'borderRadius': '5px'})
            ], style={'width': '30%', 'display': 'inline-block', 'padding': '10px', 'verticalAlign': 'top'}),
            
            html.Div([
                html.H4("Low Risk Patient Care", style={'color': '#28a745'}),
                html.Div([
                    html.P("Recommended Resource Allocation", style={'fontWeight': 'bold'}),
                    html.Ul([
                        html.Li("Routine check-ups"),
                        html.Li("Self-management support"),
                        html.Li("Educational resources")
                    ])
                ], style={'padding': '15px', 'background': '#f0fff4', 'borderRadius': '5px'})
            ], style={'width': '30%', 'display': 'inline-block', 'padding': '10px', 'verticalAlign': 'top'})
        ], style={'margin-bottom': '30px'})
    ]),
    
    # Resource Distribution Visualization
    html.Div([
        html.H3("Recommended Resource Distribution", style={'margin-bottom': '15px'}),
        dcc.Graph(id='resource-allocation-chart')
    ], style={'margin-bottom': '30px'}),
    
    # Key Metrics and Impact
    html.Div([
        html.H3("Expected Impact Analysis", style={'margin-bottom': '15px'}),
        dcc.Graph(id='impact-analysis-chart')
    ], style={'margin-bottom': '30px'}),
    
    # Action Items Timeline
    html.Div([
        html.H3("Implementation Timeline", style={'margin-bottom': '15px'}),
        dcc.Graph(id='implementation-timeline')
    ])
])

# Add debug function
def debug_print(message):
    print(f"DEBUG: {message}", file=sys.stderr)
    sys.stderr.flush()

# Callback to switch between pages
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/model-viz':
        return model_viz_layout
    elif pathname == '/predictions':
        return prediction_layout
    elif pathname == '/recommendations':
        return recommendations_layout
    elif pathname == '/eda' or pathname == '/':
        return eda_layout
    else:
        return eda_layout  # Default to EDA page for any other path

# Callback for demographic plot
@app.callback(
    Output('demographic-plot', 'figure'),
    [Input('demographic-dropdown', 'value')],
    [dash.dependencies.State('url', 'pathname')]
)
def update_demographic_plot(selected_feature, pathname):
    if pathname not in ['/', '/eda']:
        return dash.no_update
    if data.empty:
        return go.Figure()
    return px.histogram(
        data,
        x=selected_feature,
        color='readmitted',
        title=f'{selected_feature.capitalize()} Distribution by Readmission Status'
    )

# Callback for medical plot
@app.callback(
    Output('medical-plot', 'figure'),
    [Input('medical-dropdown', 'value')],
    [dash.dependencies.State('url', 'pathname')]
)
def update_medical_plot(selected_feature, pathname):
    if pathname not in ['/', '/eda']:
        return dash.no_update
    if data.empty:
        return go.Figure()
    return px.box(
        data,
        x='readmitted',
        y=selected_feature,
        title=f'{selected_feature.replace("_", " ").title()} by Readmission Status'
    )

# Callback for missing values visualization
@app.callback(
    Output('missing-values-plot', 'figure'),
    [Input('url', 'pathname')]
)
def update_missing_values_plot(pathname):
    if pathname not in ['/', '/eda']:
        return dash.no_update
    if data.empty:
        return go.Figure()
    
    # Calculate missing values
    missing_values = data.isnull().sum()
    missing_values = missing_values[missing_values > 0].sort_values(ascending=False)
    
    # Create the plot
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=missing_values.index,
            y=missing_values.values,
            marker_color='tomato',
            text=missing_values.values,
            textposition='auto'
        )
    )
    
    fig.update_layout(
        title='Missing Values per Column (Only Non-Zero)',
        xaxis_title='Columns',
        yaxis_title='Number of Missing Entries',
        xaxis_tickangle=45,
        height=500,
        margin=dict(l=50, r=20, t=50, b=150),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# Callback for readmission distribution
@app.callback(
    Output('readmission-distribution-plot', 'figure'),
    [Input('url', 'pathname')]
)
def update_readmission_distribution(pathname):
    if pathname not in ['/', '/eda']:
        return dash.no_update
    if data.empty:
        return go.Figure()
    
    # Calculate readmission rate
    data['readmission'] = data['readmitted'].apply(lambda x: 1 if x in ['<30', '>30'] else 0)
    readmission_counts = data['readmission'].value_counts()
    readmission_rate = readmission_counts[1] / len(data) * 100
    
    # Create the plot
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=['No', 'Yes'],
            y=readmission_counts.values,
            marker_color=['steelblue', 'salmon'],
            text=[f'{count}<br>({count/len(data)*100:.1f}%)' for count in readmission_counts.values],
            textposition='auto'
        )
    )
    
    fig.update_layout(
        title=f'Readmission Distribution (Overall Rate: {readmission_rate:.1f}%)',
        xaxis_title='Readmitted (any time)',
        yaxis_title='Number of Patients',
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# Callback for readmission categories
@app.callback(
    Output('readmission-categories-plot', 'figure'),
    [Input('url', 'pathname')]
)
def update_readmission_categories(pathname):
    if pathname not in ['/', '/eda']:
        return dash.no_update
    if data.empty:
        return go.Figure()
    
    # Create the plot
    fig = go.Figure()
    category_order = ['NO', '>30', '<30']
    colors = {'NO': 'steelblue', '>30': 'lightgray', '<30': 'seagreen'}
    
    for category in category_order:
        count = len(data[data['readmitted'] == category])
        fig.add_trace(
            go.Bar(
                x=[category],
                y=[count],
                name=category,
                marker_color=colors[category],
                text=[f'{count}<br>({count/len(data)*100:.1f}%)'],
                textposition='auto'
            )
        )
    
    fig.update_layout(
        title='Distribution of Readmission Categories',
        xaxis_title='Readmission Status',
        yaxis_title='Number of Patients',
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    
    return fig

# Callback for condition readmission rates
@app.callback(
    Output('condition-readmission-plot', 'figure'),
    [Input('url', 'pathname')]
)
def update_condition_readmission(pathname):
    if pathname not in ['/', '/eda']:
        return dash.no_update
    if data.empty:
        return go.Figure()
    
    # Define ICD ranges
    conditions_icd = {
        'diabetes': [(250, 251)],
        'heart_failure': [(428, 429)],
        'copd': [(490, 496)],
        'hypertension': [(401, 405)],
        'renal_disease': [(585, 586)]
    }
    
    # Helper function
    def has_icd_range(diag, code_range):
        try:
            code = float(diag)
            return any(start <= code <= end for start, end in code_range)
        except:
            return False
    
    # Create binary columns
    for condition, code_range in conditions_icd.items():
        data[condition] = data[['diag_1', 'diag_2', 'diag_3']].apply(
            lambda row: any(has_icd_range(code, code_range) for code in row), axis=1
        ).astype(int)
    
    # Calculate readmission rates
    condition_readmission = {}
    for condition in conditions_icd:
        condition_readmission[condition] = [
            data[data[condition] == 0]['readmission'].mean() * 100,
            data[data[condition] == 1]['readmission'].mean() * 100
        ]
    
    # Create the plot
    fig = go.Figure()
    
    for i, condition in enumerate(conditions_icd.keys()):
        fig.add_trace(
            go.Bar(
                x=['No', 'Yes'],
                y=condition_readmission[condition],
                name=condition.replace('_', ' ').title(),
                marker_color=['lightblue', 'salmon'],
                text=[f'{rate:.1f}%' for rate in condition_readmission[condition]],
                textposition='auto'
            )
        )
    
    fig.update_layout(
        title='Readmission Rate by Medical Condition',
        xaxis_title='Condition Present',
        yaxis_title='Readmission Rate (%)',
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        barmode='group'
    )
    
    return fig

# Callback for diagnosis distribution
@app.callback(
    Output('diagnosis-distribution-plot', 'figure'),
    [Input('url', 'pathname')]
)
def update_diagnosis_distribution(pathname):
    if pathname not in ['/', '/eda']:
        return dash.no_update
    if data.empty:
        return go.Figure()
    
    # Function to categorize diagnosis codes
    def map_diagnosis(code):
        try:
            code = float(code)
            if 390 <= code <= 459 or code == 785:
                return 'Circulatory'
            elif 460 <= code <= 519 or code == 786:
                return 'Respiratory'
            elif 520 <= code <= 579 or code == 787:
                return 'Digestive'
            elif 250 <= code < 251:
                return 'Diabetes'
            elif 800 <= code <= 999:
                return 'Injury'
            elif 710 <= code <= 739:
                return 'Musculoskeletal'
            elif 580 <= code <= 629 or code == 788:
                return 'Genitourinary'
            elif 140 <= code <= 239:
                return 'Neoplasms'
            else:
                return 'Other'
        except:
            return 'Unknown'
    
    # Apply mapping and create primary_diagnosis column
    data['primary_diagnosis'] = data['diag_1'].apply(map_diagnosis)
    diagnosis_counts = data['primary_diagnosis'].value_counts()
    
    # Create the plot
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=diagnosis_counts.index,
            y=diagnosis_counts.values,
            marker_color='skyblue',
            text=[f'{count}<br>({count/len(data)*100:.1f}%)' for count in diagnosis_counts.values],
            textposition='auto'
        )
    )
    
    fig.update_layout(
        title='Distribution of Primary Diagnoses',
        xaxis_title='Primary Diagnosis Category',
        yaxis_title='Number of Patients',
        height=500,
        xaxis_tickangle=45,
        margin=dict(l=50, r=20, t=50, b=150),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# Callback for diagnosis readmission rates
@app.callback(
    Output('diagnosis-readmission-plot', 'figure'),
    [Input('url', 'pathname')]
)
def update_diagnosis_readmission(pathname):
    if pathname not in ['/', '/eda']:
        return dash.no_update
    if data.empty:
        return go.Figure()
    
    # Function to categorize diagnosis codes
    def map_diagnosis(code):
        try:
            code = float(code)
            if 390 <= code <= 459 or code == 785:
                return 'Circulatory'
            elif 460 <= code <= 519 or code == 786:
                return 'Respiratory'
            elif 520 <= code <= 579 or code == 787:
                return 'Digestive'
            elif 250 <= code < 251:
                return 'Diabetes'
            elif 800 <= code <= 999:
                return 'Injury'
            elif 710 <= code <= 739:
                return 'Musculoskeletal'
            elif 580 <= code <= 629 or code == 788:
                return 'Genitourinary'
            elif 140 <= code <= 239:
                return 'Neoplasms'
            else:
                return 'Other'
        except:
            return 'Unknown'
    
    # Apply mapping and create primary_diagnosis column
    data['primary_diagnosis'] = data['diag_1'].apply(map_diagnosis)
    
    # Calculate readmission rates by diagnosis
    diagnosis_readmission = data.groupby('primary_diagnosis')['readmission'].mean() * 100
    diagnosis_readmission = diagnosis_readmission.sort_values(ascending=False)
    
    # Create the plot
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=diagnosis_readmission.index,
            y=diagnosis_readmission.values,
            marker_color='lightcoral',
            text=[f'{rate:.1f}%' for rate in diagnosis_readmission.values],
            textposition='auto'
        )
    )
    
    fig.update_layout(
        title='Readmission Rate by Primary Diagnosis',
        xaxis_title='Primary Diagnosis Category',
        yaxis_title='Readmission Rate (%)',
        height=500,
        xaxis_tickangle=45,
        margin=dict(l=50, r=20, t=50, b=150),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# Callback for discharge readmission rates
@app.callback(
    Output('discharge-readmission-plot', 'figure'),
    [Input('url', 'pathname')]
)
def update_discharge_readmission(pathname):
    if pathname not in ['/', '/eda']:
        return dash.no_update
    if data.empty:
        return go.Figure()
    
    # Map discharge dispositions
    disposition_mapping = {
        1: 'Home',
        2: 'Skilled Nursing',
        3: 'Home Health',
        6: 'Home Health',
        7: 'Left AMA',
        20: 'Other',
        30: 'Expired',
        99: 'Unknown'
    }
    
    data['discharge_disposition_name'] = data['discharge_disposition_id'].map(disposition_mapping).fillna('Other')
    disposition_readmission = data.groupby('discharge_disposition_name')['readmission'].mean() * 100
    disposition_readmission = disposition_readmission.sort_values(ascending=False)
    
    # Create the plot
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=disposition_readmission.index,
            y=disposition_readmission.values,
            marker_color='skyblue',
            text=[f'{rate:.1f}%' for rate in disposition_readmission.values],
            textposition='auto'
        )
    )
    
    fig.update_layout(
        title='Readmission Rate by Discharge Disposition',
        xaxis_title='Discharge Disposition',
        yaxis_title='Readmission Rate (%)',
        height=500,
        xaxis_tickangle=45,
        margin=dict(l=50, r=20, t=50, b=150),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# Callback for admission type distribution
@app.callback(
    Output('admission-type-plot', 'figure'),
    [Input('url', 'pathname')]
)
def update_admission_type(pathname):
    if pathname not in ['/', '/eda']:
        return dash.no_update
    if data.empty:
        return go.Figure()
    
    # Map admission types
    admission_type_mapping = {
        1: 'Emergency',
        2: 'Urgent',
        3: 'Elective',
        4: 'Newborn',
        5: 'Not Available',
        6: 'NULL',
        7: 'Trauma Center',
        8: 'Not Mapped'
    }
    
    data['admission_type'] = data['admission_type_id'].map(admission_type_mapping)
    admission_counts = data['admission_type'].value_counts()
    
    # Create the plot
    fig = go.Figure()
    fig.add_trace(
        go.Pie(
            labels=admission_counts.index,
            values=admission_counts.values,
            textinfo='label+percent',
            insidetextorientation='radial',
            marker=dict(colors=px.colors.qualitative.Set3)
        )
    )
    
    fig.update_layout(
        title='Distribution of Admission Types',
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# Callback for previous admissions readmission rates
@app.callback(
    Output('prev-admission-readmission-plot', 'figure'),
    [Input('url', 'pathname')]
)
def update_prev_admission_readmission(pathname):
    if pathname not in ['/', '/eda']:
        return dash.no_update
    if data.empty:
        return go.Figure()
    
    # Calculate readmission rates by number of previous admissions
    prev_admission_readmission = data.groupby('number_inpatient')['readmission'].mean() * 100
    
    # Create the plot
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=prev_admission_readmission.index,
            y=prev_admission_readmission.values,
            marker_color='skyblue',
            text=[f'{rate:.1f}%' for rate in prev_admission_readmission.values],
            textposition='auto'
        )
    )
    
    fig.update_layout(
        title='Readmission Rate by Number of Previous Inpatient Admissions',
        xaxis_title='Number of Previous Inpatient Admissions',
        yaxis_title='Readmission Rate (%)',
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# Callback for model visualizations
@app.callback(
    [Output('model-metrics', 'children'),
     Output('confusion-matrix', 'src'),
     Output('roc-curve', 'src'),
     Output('pr-curve', 'src'),
     Output('feature-importance', 'src')],
    [Input('model-dropdown', 'value')]
)
def update_model_visualizations(selected_model):
    if not selected_model:
        return dash.no_update

    try:
        print(f"\nUpdating visualizations for model: {selected_model}")
        base_path = r'E:\Data-Visualization-Project\src\models\visualizations'
        
        # Load classification report
        report_path = f'{base_path}/{selected_model}_classification_report.csv'
        try:
            print(f"Loading classification report from: {report_path}")
            report = pd.read_csv(report_path)
            metrics_html = html.Div([
                html.H4("Classification Report"),
                html.Table([
                    html.Thead(html.Tr([html.Th(col) for col in report.columns])),
                    html.Tbody([
                        html.Tr([html.Td(report.iloc[i][col]) for col in report.columns])
                        for i in range(len(report))
                    ])
                ], style={'width': '100%', 'margin-bottom': '20px'})
            ])
        except Exception as e:
            print(f"Error loading classification report: {str(e)}")
            metrics_html = html.Div("Error loading metrics", 
                                  style={'color': 'red', 'padding': '10px', 'background': '#fff3f3'})

        # Load all visualization images
        confusion_matrix_src = load_image(f'{base_path}/{selected_model}_confusion_matrix.png')
        roc_curve_src = load_image(f'{base_path}/{selected_model}_roc_curve.png')
        pr_curve_src = load_image(f'{base_path}/{selected_model}_pr_curve.png')

        # Load and create feature importance plot
        feature_importance_src = None
        try:
            feature_importance_path = f'{base_path}/{selected_model}_feature_importance.csv'
            if os.path.exists(feature_importance_path):
                feature_importance = pd.read_csv(feature_importance_path)
                
                # Sort features by importance
                feature_importance = feature_importance.sort_values('importance', ascending=True)
                
                # Create the plot
                fig_importance = go.Figure()
                fig_importance.add_trace(
                    go.Bar(
                        x=feature_importance['importance'],
                        y=feature_importance['feature'],
                        orientation='h',
                        marker_color='#1f77b4'
                    )
                )
                
                fig_importance.update_layout(
                    title='Feature Importance',
                    xaxis_title='Importance Score',
                    yaxis_title='Feature',
                    height=600,
                    margin=dict(l=200, r=20, t=50, b=20),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    yaxis=dict(
                        tickfont=dict(size=10),
                        automargin=True
                    )
                )
                
                # Convert the figure to an image
                img_bytes = fig_importance.to_image(format="png")
                encoded = base64.b64encode(img_bytes).decode('ascii')
                feature_importance_src = f'data:image/png;base64,{encoded}'
            else:
                print(f"Feature importance file not found: {feature_importance_path}")
        except Exception as e:
            print(f"Error creating feature importance plot: {str(e)}")
            print(traceback.format_exc())

        # Create placeholder for missing images
        placeholder_src = None

        # Force update by ensuring we're returning new values
        return (
            metrics_html,
            confusion_matrix_src or placeholder_src,
            roc_curve_src or placeholder_src,
            pr_curve_src or placeholder_src,
            feature_importance_src or placeholder_src
        )

    except Exception as e:
        print(f"Error in update_model_visualizations: {str(e)}")
        error_div = html.Div(f"Error loading visualizations: {str(e)}", 
                           style={'color': 'red', 'padding': '10px', 'background': '#fff3f3'})
        return error_div, None, None, None, None

# Callback for prediction page
@app.callback(
    [Output('prediction-results', 'children'),
     Output('prediction-distribution', 'figure'),
     Output('threshold-metrics', 'children'),
     Output('risk-level-pie', 'figure'),
     Output('risk-level-bar', 'figure'),
     Output('high-risk-patients', 'children'),
     Output('medium-risk-patients', 'children'),
     Output('low-risk-patients', 'children')],
    [Input('prediction-model-dropdown', 'value'),
     Input('risk-threshold-slider', 'value')],
    [dash.dependencies.State('url', 'pathname')]
)

def update_predictions(selected_model, threshold, pathname):
    try:
        print(f"Updating predictions for model: {selected_model}, threshold: {threshold}, pathname: {pathname}")
        
        if pathname != '/predictions':
            print("Not on predictions page, returning no_update")
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
            
        base_path = r'E:\Data-Visualization-Project\src\models\predictions'
        pred_path = os.path.join(base_path, f'{selected_model}_predictions.csv')
        print(f"Loading predictions from: {pred_path}")
        
        if not os.path.exists(pred_path):
            raise FileNotFoundError(f"Predictions file not found: {pred_path}")
            
        predictions = pd.read_csv(pred_path)
        print(f"Loaded predictions from {pred_path} with shape {predictions.shape}")
        
        if 'probability' not in predictions.columns:
            raise ValueError("Probability column not found in predictions file")
            
        # Calculate risk levels
        predictions['risk_level'] = pd.cut(
            predictions['probability'],
            bins=[0, 0.3, 0.7, 1.0],
            labels=['Low', 'Medium', 'High']
        )
        
        # Create distribution plot
        fig_dist = px.histogram(
            predictions,
            x='probability',
            nbins=50,
            title='Predicted Probability Distribution',
            labels={'probability': 'Readmission Probability'},
            color='risk_level',
            color_discrete_map={'Low': '#28a745', 'Medium': '#fd7e14', 'High': '#dc3545'},
            template='plotly_white'
        )
        fig_dist.update_layout(
            showlegend=True,
            legend_title_text='Risk Level',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400
        )
        
        # Create risk level pie chart
        risk_counts = predictions['risk_level'].value_counts()
        fig_pie = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title='Risk Level Distribution',
            color=risk_counts.index,
            color_discrete_map={'Low': '#28a745', 'Medium': '#fd7e14', 'High': '#dc3545'},
            template='plotly_white'
        )
        fig_pie.update_layout(showlegend=True, height=400)
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        
        # Create risk level bar chart
        fig_bar = px.bar(
            x=risk_counts.index,
            y=risk_counts.values,
            title='Number of Patients by Risk Level',
            labels={'x': 'Risk Level', 'y': 'Number of Patients'},
            color=risk_counts.index,
            color_discrete_map={'Low': '#28a745', 'Medium': '#fd7e14', 'High': '#dc3545'},
            template='plotly_white'
        )
        fig_bar.update_layout(
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400
        )
        
        # Create metrics display
        metrics_html = html.Div([
            html.H4(f"Metrics at Threshold: {threshold:.2f}"),
            html.Table([
                html.Tr([html.Td("High Risk Patients"), html.Td(f"{(predictions['risk_level'] == 'High').sum()}")]),
                html.Tr([html.Td("Medium Risk Patients"), html.Td(f"{(predictions['risk_level'] == 'Medium').sum()}")]),
                html.Tr([html.Td("Low Risk Patients"), html.Td(f"{(predictions['risk_level'] == 'Low').sum()}")])
            ], style={'width': '100%', 'margin': '20px 0', 'border-collapse': 'collapse'})
        ], style={'padding': '20px', 'background-color': '#f8f9fa', 'border-radius': '5px'})
        
        # Create patient risk summaries
        def create_risk_summary(risk_level, bg_color):
            risk_data = predictions[predictions['risk_level'] == risk_level]
            return html.Div([
                html.P(f"Total: {len(risk_data)} patients"),
                html.P(f"Average Probability: {risk_data['probability'].mean():.2%}"),
                html.P(f"Readmission Rate: {risk_data['prediction'].mean():.2%}")
            ], style={'padding': '15px', 'background-color': bg_color, 'border-radius': '5px'})
        
        high_risk_html = create_risk_summary('High', '#fff5f5')
        medium_risk_html = create_risk_summary('Medium', '#fff9f0')
        low_risk_html = create_risk_summary('Low', '#f0fff4')
        
        # Create prediction results summary
        results_summary = html.Div([
            html.H4("Prediction Results Summary"),
            html.P(f"Total Patients: {len(predictions)}"),
            html.P(f"Average Readmission Probability: {predictions['probability'].mean():.2%}"),
            html.P(f"Predicted Readmission Rate: {predictions['prediction'].mean():.2%}")
        ], style={'padding': '20px', 'background-color': '#f8f9fa', 'border-radius': '5px'})
        
        print("Successfully generated all visualizations")
        return (
            results_summary,
            fig_dist,
            metrics_html,
            fig_pie,
            fig_bar,
            high_risk_html,
            medium_risk_html,
            low_risk_html
        )
        
    except Exception as e:
        print(f"Error in update_predictions: {str(e)}")
        error_div = html.Div([
            html.H4("Error Loading Predictions", style={'color': 'red'}),
            html.P(str(e))
        ], style={'color': 'red', 'padding': '20px', 'background-color': '#fff3f3', 'border-radius': '5px'})
        
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title='No data available',
            annotations=[{
                'text': str(e),
                'xref': 'paper',
                'yref': 'paper',
                'showarrow': False,
                'font': {'size': 14}
            }]
        )
        
        return tuple([error_div] + [empty_fig] * 4 + [error_div] * 3)

# Add callback for resource allocation chart
@app.callback(
    Output('resource-allocation-chart', 'figure'),
    [Input('url', 'pathname')]
)
def update_resource_allocation(pathname):
    if pathname != '/recommendations':
        return dash.no_update
        
    # Create resource allocation visualization
    categories = ['Staff Hours', 'Equipment', 'Follow-up Care', 'Monitoring Systems']
    high_risk = [40, 35, 45, 50]
    medium_risk = [35, 35, 35, 30]
    low_risk = [25, 30, 20, 20]
    
    fig = go.Figure(data=[
        go.Bar(name='High Risk', x=categories, y=high_risk, marker_color='#dc3545'),
        go.Bar(name='Medium Risk', x=categories, y=medium_risk, marker_color='#fd7e14'),
        go.Bar(name='Low Risk', x=categories, y=low_risk, marker_color='#28a745')
    ])
    
    fig.update_layout(
        barmode='stack',
        title='Resource Distribution by Risk Level',
        xaxis_title='Resource Category',
        yaxis_title='Resource Allocation (%)',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=400
    )
    
    return fig

# Add callback for impact analysis chart
@app.callback(
    Output('impact-analysis-chart', 'figure'),
    [Input('url', 'pathname')]
)
def update_impact_analysis(pathname):
    if pathname != '/recommendations':
        return dash.no_update
        
    # Create impact analysis visualization
    metrics = ['Readmission Rate', 'Patient Satisfaction', 'Resource Efficiency', 'Cost Savings']
    current = [24, 75, 60, 50]
    expected = [15, 90, 85, 75]
    
    fig = go.Figure(data=[
        go.Bar(name='Current', x=metrics, y=current, marker_color='#6c757d'),
        go.Bar(name='Expected', x=metrics, y=expected, marker_color='#007bff')
    ])
    
    fig.update_layout(
        barmode='group',
        title='Expected Impact of Resource Optimization',
        xaxis_title='Metrics',
        yaxis_title='Score',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=400
    )
    
    return fig

# Add callback for implementation timeline
@app.callback(
    Output('implementation-timeline', 'figure'),
    [Input('url', 'pathname')]
)
def update_implementation_timeline(pathname):
    if pathname != '/recommendations':
        return dash.no_update
        
    # Create implementation timeline visualization
    phases = ['Risk Assessment System', 'Resource Redistribution', 'Staff Training', 'Monitoring Implementation', 'Evaluation']
    start = ['2024-04-01', '2024-05-01', '2024-05-15', '2024-06-01', '2024-07-01']
    end = ['2024-04-30', '2024-05-31', '2024-06-15', '2024-06-30', '2024-07-31']
    
    fig = go.Figure()
    
    for idx, phase in enumerate(phases):
        fig.add_trace(go.Bar(
            name=phase,
            x=[30],  # Duration in days
            y=[phase],
            orientation='h',
            marker_color=['#17a2b8', '#28a745', '#fd7e14', '#dc3545', '#6610f2'][idx],
            base=[pd.Timestamp(start[idx]).dayofyear],
            width=0.3
        ))
    
    fig.update_layout(
        title='Implementation Timeline',
        xaxis_title='Timeline (Days)',
        yaxis_title='Implementation Phase',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=400,
        showlegend=False
    )
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True) 
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.offline as pyo
import plotly.graph_objects as go
import base64
import os

def load_image(path):
    with open(path, 'rb') as f:
        encoded = base64.b64encode(f.read()).decode()
    return f'data:image/png;base64,{encoded}'

# Load the data
data = pd.read_csv(r'E:\Data-Visualization-Project\notebooks\diabetic_data.csv')

# Ensure readmission categories ordered
if 'readmitted' in data.columns:
    data['readmitted'] = pd.Categorical(
        data['readmitted'],
        categories=['NO', '>30', '<30'],
        ordered=True
    )

# Admission type ID to description mapping
admission_mapping = {
    1: 'Emergency',
    2: 'Urgent',
    3: 'Elective',
    4: 'Newborn',
    5: 'Not Available',
    6: 'NULL',
    7: 'Trauma Center',
    8: 'Not Mapped'
}
# Map admission_type_id to names
data['admission_type_name'] = data['admission_type_id'].map(admission_mapping).fillna('Other')

# 1. Missing Values DataFrame
missing_counts = data.isnull().sum()
missing_counts = missing_counts[missing_counts > 0]
missing_df = pd.DataFrame({
    'column': missing_counts.index,
    'missing_count': missing_counts.values,
    'missing_pct': (missing_counts.values / len(data) * 100).round(2)
})

# 2. Readmission rates for donut
readmit_counts = data['readmitted'].value_counts().reindex(['NO', '>30', '<30']).reset_index()
readmit_counts.columns = ['readmitted', 'count']

# 3. Readmission Categories Treemap
treemap_df = readmit_counts.copy()

# 4. Readmission Rate by Gender
gender_df = data.groupby(['gender', 'readmitted']).size().reset_index(name='count')
gender_pct = (
    gender_df
    .groupby('gender')
    .apply(lambda df: df.assign(pct=(df['count'] / df['count'].sum() * 100).round(2)))
    .reset_index(drop=True)
)

# 5. Medications vs Hospital Stay (no prep required)

# 6. Admission Type Distribution
admit_df = data.groupby(['admission_type_name', 'readmitted']).size().reset_index(name='count')

# 7. Correlation of Numerical Features
df_num = data.select_dtypes(include=['int64', 'float64']).corr().round(2)

# 8. Distribution of Number of Medications by Age Group
med_age_df = data[['age', 'num_medications']].dropna()

# 9. Readmission Rate by Medical Condition
def has_icd_range(diag, code_range):
    try:
        code = float(diag)
        return any(start <= code <= end for start, end in code_range)
    except:
        return False

conditions_icd = {
    'diabetes': [(250, 251)],
    'heart_failure': [(428, 429)],
    'copd': [(490, 496)],
    'hypertension': [(401, 405)],
    'renal_disease': [(585, 586)]
}
for cond, ranges in conditions_icd.items():
    data[cond] = (
        data[['diag_1', 'diag_2', 'diag_3']]
        .apply(lambda row: any(has_icd_range(val, ranges) for val in row), axis=1)
        .astype(int)
    )
cond_rates = []
for cond in conditions_icd:
    for presence in [0, 1]:
        rate = data[data[cond] == presence]['readmitted'].apply(lambda x: x != 'NO').mean() * 100
        cond_rates.append({'condition': cond, 'present': presence, 'readmission_rate': round(rate, 2)})
cond_df = pd.DataFrame(cond_rates)

# 10 & 11. Primary Diagnosis mapping
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

data['primary_diagnosis'] = data['diag_1'].apply(map_diagnosis)
diag_counts = data['primary_diagnosis'].value_counts().reset_index()
diag_counts.columns = ['primary_diagnosis', 'count']
diag_rates = (
    data.groupby('primary_diagnosis')['readmitted']
    .apply(lambda s: (s != 'NO').mean() * 100)
    .reset_index(name='readmission_rate')
)

diag_rates = diag_rates.sort_values('readmission_rate', ascending=False)



# 12. Length of stay mapping to strings
data['readmitted_str'] = data['readmitted'].astype(str)

# 13. Readmission Rate by Emergency Admission Status
data['emergency'] = data['admission_type_id'].apply(lambda x: 'Emergency' if x == 1 else 'Non-Emergency')
emg_df = data.groupby(['emergency', 'readmitted']).size().reset_index(name='count')

# 14. Readmission Rate by Discharge Disposition
discharge_mapping = {1: 'Home', 2: 'Short Term Hospital', 3: 'Skilled Nursing Facility (SNF)',
                    4: 'Intermediate Care Facility (ICF)', 5: 'Other Inpatient Institution',
                    6: 'Home with Health Service', 7: 'Left AMA', 8: 'Home with IV Care',
                    9: 'Readmitted', 10: 'Neonate Transfer', 11: 'Expired', 12: 'Outpatient Return',
                    13: 'Hospice', 14: 'Hospice', 15: 'Swing Bed', 16: 'Outpatient', 17: 'Outpatient',
                    18: 'Unknown', 19: 'Expired', 20: 'Expired', 21: 'Expired', 22: 'Rehab Facility',
                    23: 'Long Term Care', 24: 'Nursing Facility (Medicaid Only)', 25: 'Unknown',
                    26: 'Unknown', 27: 'Federal Health Facility', 28: 'Psychiatric Facility',
                    29: 'Critical Access Hospital', 30: 'Other Institution'}
data['discharge_name'] = data['discharge_disposition_id'].map(discharge_mapping).fillna('Other')
disp_rates = (
    data.groupby('discharge_name')['readmitted']
    .apply(lambda s: (s != 'NO').mean() * 100)
    .reset_index(name='readmission_rate')
)

# 15 & 16. Medication Count by Readmission Status (Box & Violin)
prev_in_df = data.groupby(['number_inpatient', 'readmitted']).size().reset_index(name='count')
# 18. Readmission Rate by Number of Previous Inpatient Admissions
prev_in_pct = (
    prev_in_df.groupby('number_inpatient')['count']
    .apply(lambda x: (x / x.sum() * 100).round(2))
    .reset_index(name='pct')
)
# 19. Distribution of Discharge Disposition Categories (Treemap)
disp_count_df = data['discharge_name'].value_counts().reset_index()
disp_count_df.columns = ['disposition', 'count']

metrics_path=r'E:\Data-Visualization-Project\src\models\model_metrics.csv'
model_metrics=pd.read_csv(metrics_path)

# Initialize Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Serve EDA page with updated plots
def serve_eda():
    return html.Div([
        html.H2("Exploratory Data Analysis"),

        # 1. Missing Values
        html.H4("Columns with Missing Values"),
        dcc.Graph(
            figure=px.bar(
                missing_df, x='missing_count', y='column', orientation='h', text='missing_pct',
                title='Missing Values per Column'
            ).update_layout(yaxis={'categoryorder': 'total ascending'})
        ),

        # 2. Readmission Rate (Donut)
        html.H4("Readmission Rate (Any Time)"),
        dcc.Graph(
            figure=go.Figure(
                data=[go.Pie(
                    labels=readmit_counts['readmitted'],
                    values=readmit_counts['count'],
                    hole=0.4,
                    textinfo='label+percent'
                )]
            ).update_layout(title_text='Readmitted vs Non-Readmitted')
        ),

        # 3. Readmission Categories Treemap
        html.H4("Distribution of Readmission Categories"),
        dcc.Graph(
            figure=px.treemap(
                treemap_df, path=['readmitted'], values='count', title='Readmission Categories'
            )
        ),

        # 4. Readmission Rate by Gender
        html.H4("Readmission Rate by Gender"),
        dcc.Graph(
            figure=px.bar(
                gender_pct, x='gender', y='pct', color='readmitted', barmode='group',
                title='Readmission Percentage by Gender'
            )
        ),

        # 5. Medications vs Hospital Stay
        html.H4("Medications vs Length of Stay"),
        dcc.Graph(
            figure=px.scatter(
                data, x='num_medications', y='time_in_hospital', color='readmitted',
                color_discrete_sequence=px.colors.qualitative.Set1,
                title='Medications vs Hospital Stay'
            )
        ),

        # 6. Admission Type Distribution
        html.H4("Admission Type Distribution"),
        dcc.Graph(
            figure=px.bar(
                admit_df, x='admission_type_name', y='count', color='readmitted',
                barmode='stack', title='Admission Type Distribution'
            )
        ),

        # 7. Correlation of Numerical Features
        html.H4("Feature Correlations"),
        dcc.Graph(
            figure=px.imshow(
                df_num, text_auto=True, title='Correlation Matrix of Numerical Features'
            ).update_layout(
                title_x=0.5, title_y=0.95, title_font=dict(size=20), width=800, height=600
            )
        ),

        # 8. Distribution of Number of Medications by Age Group
        html.H4("Medications by Age Group"),
        dcc.Graph(
            figure=px.violin(
                med_age_df, x='age', y='num_medications', box=True, color='age',
                title='Number of Medications Distribution by Age Group'
            )
        ),

        # 9. Readmission Rate by Medical Condition
        html.H4("Readmission Rate by Medical Condition"),
        dcc.Graph(
            figure=px.bar(
                cond_df, x='readmission_rate', y='condition', color='present', orientation='h',
                barmode='group', title='Readmission % by Condition'
            )
        ),

        # 10. Distribution of Primary Diagnoses
        html.H4("Distribution of Primary Diagnoses"),
        dcc.Graph(
            figure=px.bar(
                diag_counts.head(10), x='count', y='primary_diagnosis', orientation='h',
                title='Top 10 Primary Diagnoses'
            )
        ),

        # 11. Readmission Rate by Primary Diagnosis (Lollipop Chart)
        html.H4("Readmission Rate by Primary Diagnosis"),
        dcc.Graph(
            figure=go.Figure(
                data=[
                    # vertical “heads” in crimson
                    go.Scatter(
                        x=diag_rates['primary_diagnosis'],
                        y=diag_rates['readmission_rate'],
                        mode='markers',
                        marker=dict(color='crimson', size=10),
                        name='Readmission Rate'
                    )
                ],
                layout=go.Layout(
                    title='Readmission % by Primary Diagnosis',
                    xaxis=dict(
                        title='Diagnosis',
                        tickangle=-45,           # rotate labels for readability
                        categoryorder='total descending'  # highest first
                    ),
                    yaxis=dict(title='Readmission Rate (%)'),
                    margin=dict(b=150, l=80),     # bottom margin for rotated labels
                    shapes=[
                        # vertical sticks from y=0 up to each rate
                        dict(
                            type='line',
                            xref='x',
                            yref='y',
                            x0=diag,
                            y0=0,
                            x1=diag,
                            y1=rate,
                            line=dict(color='black', width=2)
                        )
                        for diag, rate in zip(
                            diag_rates['primary_diagnosis'],
                            diag_rates['readmission_rate']
                        )
                    ]
                )
            )
        ),

        # 12. Length of Stay by Readmission Status
        html.H4("Length of Stay by Readmission Status"),
        dcc.Graph(
            figure=px.box(
                data, x='readmitted_str', y='time_in_hospital', color='readmitted_str',
                title='Length of Stay (Days) vs Readmission', labels={'readmitted_str':'Readmitted'}
            )
        ),

        # 13. Readmission Rate by Emergency Admission Status
        html.H4("Readmission Rate by Emergency Admission Status"),
        dcc.Graph(
            figure=px.bar(
                emg_df, x='emergency', y='count', color='readmitted', barmode='stack',
                title='Emergency vs Non-Emergency Readmission'
            )
        ),

        # 14. Readmission Rate by Discharge Disposition (Dot Plot)
        html.H4("Readmission Rate by Discharge Disposition"),
        dcc.Graph(
            figure=px.scatter(
                disp_rates, x='readmission_rate', y='discharge_name', size='readmission_rate',
                orientation='h', title='Readmission % by Discharge Disposition'
            )
        ),

        # 15. Medication Count by Readmission Status (Box)
        html.H4("Medication Count by Readmission Status"),
        dcc.Graph(
            figure=px.box(
                data, x='readmitted_str', y='num_medications', color='readmitted_str',
                title='Medication Count vs Readmission'
            )
        ),

        # 16. Medication Count by Readmission Status (Violin)
        html.H4("Medication Count Distribution (Violin Plot)"),
        dcc.Graph(
            figure=px.violin(
                data, x='readmitted_str', y='num_medications', box=True, color='readmitted_str',
                title='Medication Count Distribution by Readmission'
            )
        ),

        # 17. Previous Inpatient Admissions by Readmission Status
        html.H4("Previous Inpatient Admissions by Readmission Status"),
        dcc.Graph(
            figure=px.bar(
                prev_in_df, x='number_inpatient', y='count', color='readmitted', barmode='group',
                title='Previous Inpatient Admissions'
            )
        ),

        # 18. Readmission Rate by Number of Previous Inpatient Admissions
        html.H4("Readmission Rate by # of Previous Inpatient Admissions"),
        dcc.Graph(
            figure=px.line(
                prev_in_pct, x='number_inpatient', y='pct', title='Readmission % vs # Previous Inpatient Admissions'
            )
        ),

        # 19. Distribution of Discharge Disposition Categories (Treemap)
        html.H4("Distribution of Discharge Disposition Categories"),
        dcc.Graph(
            figure=px.treemap(
                disp_count_df, path=['disposition'], values='count', title='Discharge Disposition Categories'
            )
        ),

    ], style={'padding': '20px'})

# Main layout and routing
app.layout = html.Div([
    html.Nav([
        html.Div([
            html.H3("Diabetic Readmission Analysis", style={'color':'white','margin':'0 40px 0 0'}),
            html.Ul([
                html.Li(dcc.Link('Home', href='/')),html.Li(dcc.Link('EDA', href='/eda')),html.Li(dcc.Link('Model Viz', href='/model-viz')),
                html.Li(dcc.Link('Predictions', href='/predictions')),html.Li(dcc.Link('Recommendations', href='/recommendations'))
            ], style={'display':'flex','list-style':'none','gap':'20px'})
        ], style={'display':'flex','justify-content':'space-between','align-items':'center','width':'95%'})
    ], style={'background-color':'#333','padding':'15px','box-shadow':'0 2px 4px rgba(0,0,0,0.1)', 'position':'fixed', 'top':'0', 'width':'100%', 'z-index':'1000'}),
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content', style={'max-width':'1200px','margin':'0 auto', 'padding-top':'80px'})
])

def serve_home():
    return html.Div([
        # Hero Section
        html.Div([
            html.Div([
                html.H2("Diabetes Readmission Prediction Project", 
                       style={'color':'white', 'fontSize':'2.5rem', 'marginBottom':'20px'}),
                html.P("An end-to-end solution for predicting hospital readmissions for diabetes patients using machine learning.",
                      style={'color':'white', 'fontSize':'1.2rem', 'maxWidth':'800px'})
            ], style={'padding':'80px 20px', 'background':'linear-gradient(135deg, #3498db, #2c3e50)'})
        ], style={'margin':'-20px -20px 40px -20px'}),
        
        # Project Overview
        html.Div([
            html.H3("Project Overview", style={'color':'#2c3e50', 'marginBottom':'30px'}),
            html.Div([
                html.Div([
                    html.H4("Data Pipeline", style={'color':'#3498db'}),
                    html.P("Automated data cleaning, preprocessing, and feature engineering pipelines"),
                    html.H4("ML Models", style={'color':'#3498db'}),
                    html.P("Multiple machine learning models trained and evaluated for optimal performance"),
                    html.H4("Dashboard", style={'color':'#3498db'}),
                    html.P("Interactive visualization platform for monitoring readmission risks"),
                    html.H4("Resource Optimization", style={'color':'#3498db'}),
                    html.P("Data-driven recommendations for healthcare resource allocation")
                ], style={'flex':'1', 'padding':'20px', 'background':'#f8f9fa', 'borderRadius':'10px',
                          'marginRight':'20px'}),
                
                # Team Members
                html.Div([
                    html.H3("Team Members", style={'color':'#2c3e50', 'marginBottom':'20px'}),
                    html.Div([
                        html.Div([
                            html.Div([
                                html.H4("Shridhar Kumar", style={'margin':'0'}),
                                html.P("Team Lead", style={'color':'#7f8c8d', 'margin':'5px 0'})
                            ], style={'padding':'15px'})
                        ], style={'border':'1px solid #ecf0f1', 'borderRadius':'8px', 'marginBottom':'15px'}),
                        
                        html.Div([
                            html.Div([
                                html.H4("Kritika Gahlawat", style={'margin':'0'}),
                                
                            ], style={'padding':'15px'})
                        ], style={'border':'1px solid #ecf0f1', 'borderRadius':'8px', 'marginBottom':'15px'}),
                        
                        html.Div([
                            html.Div([
                                html.H4("Biswajit Gorai", style={'margin':'0'}),
                        
                            ], style={'padding':'15px'})
                        ], style={'border':'1px solid #ecf0f1', 'borderRadius':'8px', 'marginBottom':'15px'}),
                        
                        html.Div([
                            html.Div([
                                html.H4("Neha Rana", style={'margin':'0'}),
                                
                            ], style={'padding':'15px'})
                        ], style={'border':'1px solid #ecf0f1', 'borderRadius':'8px', 'marginBottom':'15px'}),
                        
                        html.Div([
                            html.Div([
                                html.H4("Saswata Ghosh", style={'margin':'0'}),
                        
                            ], style={'padding':'15px'})
                        ], style={'border':'1px solid #ecf0f1', 'borderRadius':'8px'})
                    ], style={'display':'flex', 'flexDirection':'column'})
                ], style={'flex':'0 0 300px', 'padding':'20px', 'background':'white', 
                         'borderRadius':'10px', 'boxShadow':'0 2px 4px rgba(0,0,0,0.1)'})
            ], style={'display':'flex', 'gap':'30px', 'marginBottom':'40px'}),
            
            # Key Features
            html.Div([
                html.H3("Key Features", style={'color':'#2c3e50', 'marginBottom':'30px'}),
                html.Div([
                    html.Div([
                        html.Img(src="https://cdn-icons-png.flaticon.com/512/1534/1534959.png",
                                style={'height':'60px', 'marginBottom':'15px'}),
                        html.H4("Predictive Analytics"),
                        html.P("Advanced ML models for accurate readmission prediction", 
                              style={'color':'#7f8c8d'})
                    ], style={'textAlign':'center', 'padding':'20px', 'flex':'1'}),
                    
                    html.Div([
                        html.Img(src="https://cdn-icons-png.flaticon.com/512/2103/2103787.png",
                                style={'height':'60px', 'marginBottom':'15px'}),
                        html.H4("Interactive Dashboard"),
                        html.P("Real-time visualization of patient data and predictions", 
                              style={'color':'#7f8c8d'})
                    ], style={'textAlign':'center', 'padding':'20px', 'flex':'1'}),
                    
                    html.Div([
                        html.Img(src="https://cdn-icons-png.flaticon.com/512/3594/3594465.png",
                                style={'height':'60px', 'marginBottom':'15px'}),
                        html.H4("Clinical Insights"),
                        html.P("Actionable recommendations for healthcare providers", 
                              style={'color':'#7f8c8d'})
                    ], style={'textAlign':'center', 'padding':'20px', 'flex':'1'})
                ], style={'display':'flex', 'gap':'30px', 'background':'#f8f9fa',
                          'padding':'30px', 'borderRadius':'10px'})
            ])
        ], style={'maxWidth':'1200px', 'margin':'0 auto'})
    ])

def serve_model_viz():
    base_vis=r'E:\Data-Visualization-Project\src\models\visualizations'
    default_model = model_metrics['model'].iloc[0]
    return html.Div([
        html.H2('Model Visualizations',style={'margin-bottom':'20px'}),
        html.Div([
            html.H3('Model Comparison'),
            html.Img(src=load_image(os.path.join(base_vis,'model_comparison.png')),style={'width':'100%','margin-bottom':'20px'})
        ]),
        html.Div([
            html.H3('Model Performance Metrics'),
            dcc.Dropdown(
                id='model-dropdown',
                options=[{'label':m,'value':m} for m in model_metrics['model']],
                value=model_metrics['model'].iloc[0],
                style={'width':'50%','margin-bottom':'20px'}
            ),
            html.Div(id='model-metrics')
        ]),
        html.Div([
            html.Div([html.H4('Confusion Matrix'),html.Img(id='confusion-matrix',
                                                           src= load_image(os.path.join(base_vis,f"{default_model}_confusion_matrix.png")),
                                                           style={'width':'100%'})],style={'width':'33%','display':'inline-block','padding':'10px'}),
            html.Div([html.H4('ROC Curve'),html.Img(id='roc-curve',
                                                    src=load_image(os.path.join(base_vis,f"{default_model}_roc_curve.png")),
                                                    style={'width':'100%'})],style={'width':'33%','display':'inline-block','padding':'10px'}),
            html.Div([html.H4('PR Curve'),html.Img(id='pr-curve',src=load_image(os.path.join(base_vis,f"{default_model}_pr_curve.png")),
                                                   style={'width':'100%'})],style={'width':'33%','display':'inline-block','padding':'10px'})
        ])
    ],style={'padding':'20px'})

def serve_predictions():
    return html.Div([
        html.H2('Predictions'),
        dcc.Dropdown(
            id='prediction-model-dropdown',
            options=[{'label': m, 'value': m} for m in model_metrics['model']],
            value=model_metrics['model'].iloc[0],
            style={'width': '50%', 'margin-bottom': '20px'}
        ),
        html.Div(id='prediction-results'),
        dcc.Graph(id='prediction-distribution'),
        html.Div(id='threshold-metrics'),
        dcc.Graph(id='risk-level-pie'),
        dcc.Graph(id='risk-level-bar'),
        html.Div(id='high-risk-patients'),
        html.Div(id='medium-risk-patients'),
        html.Div(id='low-risk-patients')
    ], style={'padding': '20px'})


@app.callback(Output('page-content','children'), [Input('url','pathname')])
def display_page(pathname):
    if pathname == '/':
        return serve_home()
    elif pathname == '/eda':
        return serve_eda()
    elif pathname == '/model-viz':
        return serve_model_viz()
    elif pathname == '/predictions':
        return serve_predictions()
    return serve_home()

@app.callback(
    Output('confusion-matrix', 'src'),
    Output('roc-curve', 'src'),
    Output('pr-curve', 'src'),
    Input('model-dropdown', 'value')
)
def update_model_visualizations(selected_model):
    base_vis = r'E:\Data-Visualization-Project\src\models\visualizations'
    confusion_path = os.path.join(base_vis, f"{selected_model}_confusion_matrix.png")
    roc_path = os.path.join(base_vis, f"{selected_model}_roc_curve.png")
    pr_path = os.path.join(base_vis, f"{selected_model}_pr_curve.png")

    return (
        load_image(confusion_path),
        load_image(roc_path),
        load_image(pr_path)
    )

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
    [Input('prediction-model-dropdown', 'value')],
    [dash.dependencies.State('url', 'pathname')]
)
def update_predictions(selected_model, pathname):
    try:
        threshold = 0.5
        print(f"Updating predictions for model: {selected_model}, threshold: {threshold}, pathname: {pathname}")
        
        if pathname != '/predictions':
            print("Not on predictions page, returning no_update")
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
        
        base_path = r'E:\Data-Visualization-Project\src\models\src\models\predictions'
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
            y='probability',
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
                html.P(f"Average Readmission Probability: {risk_data['probability'].mean():.2%}"),
                html.P(f"Predicted Readmission Rate: {risk_data['prediction'].mean():.2%}")
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
        
        # Create error display
        error_div = html.Div([
            html.H4("Error Loading Predictions", style={'color': 'red'}),
            html.P(str(e))
        ], style={'color': 'red', 'padding': '20px', 'background-color': '#fff3f3', 'border-radius': '5px'})
        
        # Empty figure placeholders
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
        
        # Return error messages and empty charts for each output
        return tuple([error_div] + [empty_fig] * 4 + [error_div] * 3)


if __name__ == '__main__':
    app.run_server(debug=True)

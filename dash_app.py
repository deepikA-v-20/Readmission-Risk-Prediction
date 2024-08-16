import pickle
import pandas as pd
from dash import dcc, html, Dash, dash_table
from dash.dependencies import Input, Output
import plotly.express as px

# Load the pretrained model
model_path = r'C:\Users\DEEPIKA\Downloads\readm fe\data\trained_ensemble_model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Load the dataset
data_path = r'C:\Users\DEEPIKA\Downloads\readm fe\data\updated_dataset_with_ids.csv'
df = pd.read_csv(data_path)

# Ensure the dataset has an 'Age' column for visualization
if 'Age' not in df.columns:
    raise ValueError("The dataset must contain an 'Age' column for visualization purposes.")

# Check if 'PatientID' is in the DataFrame columns before attempting to drop it
if 'PatientID' in df.columns:
    features = df.drop(columns=['PatientID', 'Outcome_Readmission_within_30days'])
else:
    features = df.drop(columns=['Outcome_Readmission_within_30days'])

# Handle missing values
features.fillna(0, inplace=True)

# Make predictions
df['risk_probability'] = model.predict_proba(features)[:, 1]
df['risk'] = df['risk_probability'].apply(lambda x: 'High' if x > 0.5 else 'Low')

# Get top 10 high and low-risk patients
top_10_high_risk = df[df['risk'] == 'High'].nlargest(10, 'risk_probability')[['PatientID', 'risk_probability']]
top_10_low_risk = df[df['risk'] == 'Low'].nsmallest(10, 'risk_probability')[['PatientID', 'risk_probability']]

# Initialize Dash app
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Risk Prediction Dashboard"),
    
    # Risk Type Dropdown
    html.Div([
        dcc.Dropdown(
            id='view-dropdown',
            options=[
                {'label': 'All Patients', 'value': 'All'},
                {'label': 'High Risk', 'value': 'High'},
                {'label': 'Low Risk', 'value': 'Low'}
            ],
            value='All',
            clearable=False
        ),
        dcc.Dropdown(
            id='patient-dropdown',
            options=[
                {'label': pid, 'value': pid} for pid in df['PatientID'].unique()
            ],
            value=None,
            placeholder="Select a Patient ID",
            clearable=True
        )
    ]),
    
    html.Div([
        dcc.Graph(id='risk-distribution-pie'),
        dcc.Graph(id='age-distribution-pie'),
        dcc.Graph(id='risk-distribution-bubble'),
        dcc.Graph(id='age-distribution-bubble'),
        html.Div([
            html.H2("Top 10 High-Risk Patients"),
            dash_table.DataTable(
                id='top-high-risk-table',
                columns=[{"name": i, "id": i} for i in top_10_high_risk.columns],
                data=top_10_high_risk.to_dict('records'),
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left'},
                page_size=10
            ),
            html.H2("Top 10 Low-Risk Patients"),
            dash_table.DataTable(
                id='top-low-risk-table',
                columns=[{"name": i, "id": i} for i in top_10_low_risk.columns],
                data=top_10_low_risk.to_dict('records'),
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left'},
                page_size=10
            ),
        ])
    ])
])

@app.callback(
    [Output('risk-distribution-pie', 'figure'),
     Output('age-distribution-pie', 'figure'),
     Output('risk-distribution-bubble', 'figure'),
     Output('age-distribution-bubble', 'figure')],
    [Input('view-dropdown', 'value'),
     Input('patient-dropdown', 'value')]
)
def update_graph(view, selected_patient):
    if selected_patient:
        filtered_df = df[df['PatientID'] == selected_patient]
    else:
        if view == 'High':
            filtered_df = df[df['risk'] == 'High']
        elif view == 'Low':
            filtered_df = df[df['risk'] == 'Low']
        else:
            filtered_df = df

    # Risk Distribution Pie Chart
    risk_distribution_pie = px.pie(
        filtered_df,
        names='risk',
        title='Risk Distribution Pie Chart',
        color='risk',
        color_discrete_map={'High': 'red', 'Low': 'blue'}
    )

    # Age Distribution Pie Chart
    age_distribution_pie = px.pie(
        filtered_df.groupby('risk')['Age'].mean().reset_index(),
        names='risk',
        values='Age',
        title='Average Age Distribution Pie Chart',
        color='risk',
        color_discrete_map={'High': 'red', 'Low': 'blue'}
    )

    # Risk Distribution Bubble Chart
    risk_distribution_bubble = px.scatter(
        filtered_df,
        x='PatientID',
        y='risk_probability',
        size='risk_probability',
        color='risk',
        title='Risk Distribution Bubble Chart',
        hover_data=['PatientID', 'risk_probability'],
        color_discrete_map={'High': 'red', 'Low': 'blue'},
        size_max=20  # Adjust the max size of bubbles
    )

    # Age Distribution Bubble Chart
    age_distribution_bubble = px.scatter(
        filtered_df,
        x='PatientID',
        y='Age',
        size='Age',
        color='risk',
        title='Age Distribution Bubble Chart',
        hover_data=['PatientID', 'risk_probability'],
        color_discrete_map={'High': 'red', 'Low': 'blue'},
        size_max=20  # Adjust the max size of bubbles
    )

    return risk_distribution_pie, age_distribution_pie, risk_distribution_bubble, age_distribution_bubble

if __name__ == '__main__':
    app.run_server(debug=True)

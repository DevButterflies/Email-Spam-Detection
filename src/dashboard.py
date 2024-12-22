import dash
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from preprocess import EmailPreprocessor
from model import SpamDetectionModel, ModelEvaluator
import joblib
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc
import base64

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Global variables for models and data
models = {}
preprocessor = None
current_results = {}
test_data = None

def load_and_process_data(file_path):
    """Load and process the email dataset."""
    global preprocessor, test_data
    
    # Initialize preprocessor
    preprocessor = EmailPreprocessor()
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(file_path)
    
    # Train models and get results
    results = ModelEvaluator.compare_models(X_train, X_test, y_train, y_test)
    
    # Store test data for later use
    test_data = (X_test, y_test)
    
    return results

def create_model_comparison_plot(results):
    """Create a bar plot comparing model accuracies."""
    models_df = pd.DataFrame([
        {
            'Model': model,
            'Accuracy': metrics['accuracy'],
            'Precision (Spam)': metrics['class_metrics']['spam']['precision'],
            'Recall (Spam)': metrics['class_metrics']['spam']['recall'],
            'F1 Score (Spam)': metrics['class_metrics']['spam']['f1']
        }
        for model, metrics in results.items()
    ])
    
    fig = go.Figure()
    metrics = ['Accuracy', 'Precision (Spam)', 'Recall (Spam)', 'F1 Score (Spam)']
    
    for metric in metrics:
        fig.add_trace(go.Bar(
            name=metric,
            x=models_df['Model'],
            y=models_df[metric],
            text=models_df[metric].round(3),
            textposition='auto',
        ))
    
    fig.update_layout(
        title='Model Performance Comparison',
        barmode='group',
        yaxis_range=[0, 1]
    )
    
    return fig

def create_confusion_matrix_plot(results):
    """Create confusion matrix plots for all models."""
    n_models = len(results)
    fig = make_subplots(
        rows=(n_models + 1) // 2,
        cols=2,
        subplot_titles=list(results.keys())
    )
    
    for i, (model_name, metrics) in enumerate(results.items()):
        row = i // 2 + 1
        col = i % 2 + 1
        
        cm = metrics['confusion_matrix']
        
        heatmap = go.Heatmap(
            z=cm,
            x=['Ham', 'Spam'],
            y=['Ham', 'Spam'],
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 16},
            colorscale='Blues'
        )
        
        fig.add_trace(heatmap, row=row, col=col)
    
    fig.update_layout(
        title='Confusion Matrices',
        height=300 * ((n_models + 1) // 2),
        showlegend=False
    )
    
    return fig

def create_roc_curves(results):
    """Create ROC curves for all models."""
    fig = go.Figure()
    
    for model_name, metrics in results.items():
        if 'probability_scores' in metrics:
            fpr, tpr, _ = roc_curve(test_data[1], metrics['probability_scores']['spam_proba'])
            roc_auc = auc(fpr, tpr)
            
            fig.add_trace(go.Scatter(
                x=fpr,
                y=tpr,
                name=f'{model_name} (AUC = {roc_auc:.3f})',
                mode='lines'
            ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        name='Random',
        mode='lines',
        line=dict(dash='dash')
    ))
    
    fig.update_layout(
        title='ROC Curves',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'),
        width=800,
        height=800
    )
    
    return fig

# App layout
app.layout = dbc.Container([
    html.H1("Spam Email Detection Dashboard", className="text-center my-4"),
    
    # File Upload
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False
    ),
    
    # Status and Model Selection
    dbc.Row([
        dbc.Col(html.Div(id='upload-status'), width=8),
        dbc.Col(
            dcc.Dropdown(
                id='model-selector',
                placeholder="Select a model for predictions"
            ),
            width=4
        )
    ], className="mb-4"),
    
    # Tabs for different visualizations
    dcc.Tabs([
        # Model Comparison Tab
        dcc.Tab(label='Model Comparison', children=[
            dbc.Row([
                dbc.Col(dcc.Graph(id='model-comparison-plot'), width=12)
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id='confusion-matrices'), width=12)
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id='roc-curves'), width=12)
            ])
        ]),
        
        # Detailed Metrics Tab
        dcc.Tab(label='Detailed Metrics', children=[
            html.Div(id='detailed-metrics')
        ]),
        
        # Prediction Interface Tab
        dcc.Tab(label='Make Predictions', children=[
            dbc.Row([
                dbc.Col([
                    dbc.Input(
                        id='input-message',
                        placeholder='Enter an email message...',
                        type='text',
                        className="mb-2"
                    ),
                    dbc.Button(
                        "Predict",
                        id='predict-button',
                        color="primary",
                        className="mb-2"
                    ),
                    html.Div(id='prediction-output')
                ])
            ])
        ])
    ])
], fluid=True)

@app.callback(
    [Output('upload-status', 'children'),
     Output('model-comparison-plot', 'figure'),
     Output('confusion-matrices', 'figure'),
     Output('roc-curves', 'figure'),
     Output('detailed-metrics', 'children'),
     Output('model-selector', 'options')],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def update_output(contents, filename):
    if contents is None:
        return "Please upload your dataset.", {}, {}, {}, "", []
    
    try:
        # Decode and load the file
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        # Save temporarily and process
        temp_path = "temp_dataset.csv"
        with open(temp_path, 'wb') as f:
            f.write(decoded)
        
        # Load and process data
        results = load_and_process_data(temp_path)
        os.remove(temp_path)
        
        # Store results globally
        global current_results
        current_results = results
        
        # Create visualizations
        comparison_plot = create_model_comparison_plot(results)
        confusion_plot = create_confusion_matrix_plot(results)
        roc_plot = create_roc_curves(results)
        
        # Create detailed metrics table
        metrics_tables = []
        for model_name, metrics in results.items():
            metrics_tables.append(html.H4(f"{model_name} Metrics"))
            metrics_tables.append(html.Pre(metrics['classification_report']))
        
        # Create model selector options
        model_options = [{'label': model, 'value': model} for model in results.keys()]
        
        return (
            f"Dataset '{filename}' processed successfully!",
            comparison_plot,
            confusion_plot,
            roc_plot,
            metrics_tables,
            model_options
        )
    
    except Exception as e:
        return f"Error processing file: {str(e)}", {}, {}, {}, "", []

@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [State('input-message', 'value'),
     State('model-selector', 'value')]
)
def make_prediction(n_clicks, message, selected_model):
    if n_clicks is None or message is None or selected_model is None:
        return ""
    
    try:
        # Preprocess the message
        X = preprocessor.transform_text(message)
        
        # Get the selected model
        model = SpamDetectionModel(selected_model)
        model.model = current_results[selected_model]['model']
        
        # Make prediction
        prediction = model.predict(X)[0]
        proba = model.predict_proba(X)
        
        # Create output
        result = html.Div([
            html.H5(
                f"Prediction: {'SPAM' if prediction == 1 else 'HAM'}", 
                style={'color': 'red' if prediction == 1 else 'green'}
            )
        ])
        
        if proba is not None:
            result.children.append(html.P(
                f"Confidence: {proba[0][1]:.2%} chance of being spam"
            ))
        
        return result
    
    except Exception as e:
        return f"Error making prediction: {str(e)}"

if __name__ == '__main__':
    app.run_server(debug=True)

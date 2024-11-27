"""Create a Dash app to visualize 3D t-SNE plots of the embedding and latent spaces."""
import dash
from dash import dcc, html, Input, Output
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from database_management.database_managers.tsne_database_manager import (
    TsneDatabaseManager,
    EMBEDDING_TSNE_KEY,
    LATENT_TSNE_KEY
)
from settings import SAMPLE_PATH_DB_KEY


class TsneVisualizer:
    def __init__(self):
        """Constructor"""
        self._df = TsneDatabaseManager(read_from_file=True).dd.compute()
        self._app = None

    @staticmethod
    def create_3d_scatter(df, column_key, title, color_map):
            """
            Creates a 3D scatter plot using Plotly.
            :param df: DataFrame containing the data to be plotted.
            :type df: pandas.DataFrame
            :param column_key: The key in the DataFrame to access the coordinates for the scatter plot.
            :type column_key: str
            :param title: The title of the scatter plot.
            :type title: str
            :param color_map: A dictionary mapping sample paths to colors.
            :type color_map: dict
            :return: A Plotly Figure object representing the 3D scatter plot.
            :rtype: plotly.graph_objs._figure.Figure
            """
            
            fig = go.Figure(data=[go.Scatter3d(
                x=[coords[0] for coords in df[column_key]],
                y=[coords[1] for coords in df[column_key]],
                z=[coords[2] for coords in df[column_key]],
                mode='markers',
                marker=dict(size=5, opacity=0.8, color=[color_map[path] for path in df[SAMPLE_PATH_DB_KEY]]),
                text=df[SAMPLE_PATH_DB_KEY],  # Sample path displayed on hover
                hoverinfo='text'
            )])
            fig.update_layout(title=title, margin=dict(l=0, r=0, b=0, t=40))
            return fig

    def _prepare_dash(self):
        """
        Prepare the Dash application layout and callbacks for t-SNE visualization.
        This method initializes the Dash application, sets up the layout with two 3D t-SNE plots
        (embedding t-SNE and latent t-SNE), and adds a callback to display the file path when a point
        on either plot is clicked.
        """
        
        # Initialize the Dash application
        self._app = dash.Dash(__name__)

        color_map = {path: f'rgba({np.random.randint(0, 255)}, {np.random.randint(0, 255)}, {np.random.randint(0, 255)}, 0.8)' for path in self._df[SAMPLE_PATH_DB_KEY]}

        self._app.layout = html.Div([
            html.H1("3D t-SNE Visualization"),
            # Section for embedding_tsne
            html.Div([
                html.H3("Embedding t-SNE"),
                dcc.Graph(id='embedding-tsne', figure=self.create_3d_scatter(self._df, EMBEDDING_TSNE_KEY, "Embedding t-SNE", color_map))
            ], style={'width': '48%', 'display': 'inline-block', 'padding': '0 20'}),

            # Section for latent_tsne
            html.Div([
                html.H3("Latent t-SNE"),
                dcc.Graph(id='latent-tsne', figure=self.create_3d_scatter(self._df, LATENT_TSNE_KEY, "Latent t-SNE", color_map))
            ], style={'width': '48%', 'display': 'inline-block', 'padding': '0 20'}),

            # Section displaying the file path after clicking
            html.Div(id='file-path-output', style={'marginTop': 20, 'fontSize': 18})
        ])

        # Callback to display the file path after clicking a point on the plot
        @self._app.callback(
            Output('file-path-output', 'children'),
            [Input('embedding-tsne', 'clickData'),
            Input('latent-tsne', 'clickData')]
        )
        def display_file_path(embedding_click, latent_click):
            """
            Display the file path when a point on the t-SNE plot is clicked.

            :param embedding_click: Data from clicking a point on the embedding t-SNE plot.
            :type embedding_click: dict
            :param latent_click: Data from clicking a point on the latent t-SNE plot.
            :type latent_click: dict
            :return: The file path of the clicked point or a prompt to click a point.
            :rtype: str
            """
            clicked_data = embedding_click or latent_click
            if clicked_data:
                sample_path = clicked_data['points'][0]['text']
                return f"File Path: {sample_path}"
            return "Click on a point to see the file path."

    def visualize(self):
        """Prepare and run the Dash server to visualize the t-SNE plots."""
        self._prepare_dash()
        self._app.run_server(debug=True, host='0.0.0.0', port=8050)

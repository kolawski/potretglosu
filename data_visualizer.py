import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.graph_objects as go

from tsne_database_manager import TsneDatabaseManager, EMBEDDING_TSNE_KEY, LATENT_TSNE_KEY, SAMPLE_PATH_KEY

class DataVisualizer:
    def __init__(self):
        self._df = TsneDatabaseManager(read_from_file=True).dd.compute()
        self._app = None

    @staticmethod
    def create_3d_scatter(df, column_key, title):
            fig = go.Figure(data=[go.Scatter3d(
                x=[coords[0] for coords in df[column_key]],
                y=[coords[1] for coords in df[column_key]],
                z=[coords[2] for coords in df[column_key]],
                mode='markers',
                marker=dict(size=5, color='blue', opacity=0.8),
                text=df[SAMPLE_PATH_KEY],  # Sample_path wyświetlany po najechaniu
                hoverinfo='text'
            )])
            fig.update_layout(title=title, margin=dict(l=0, r=0, b=0, t=40))
            return fig

    def _prepare_dash(self):
        # Inicjalizacja aplikacji Dash
        self._app = dash.Dash(__name__)

        self._app.layout = html.Div([
            html.H1("3D t-SNE Visualization"),
            # Sekcja dla embedding_tsne
            html.Div([
                html.H3("Embedding t-SNE"),
                dcc.Graph(id='embedding-tsne', figure=self.create_3d_scatter(self._df, EMBEDDING_TSNE_KEY, "Embedding t-SNE"))
            ], style={'width': '48%', 'display': 'inline-block', 'padding': '0 20'}),

            # Sekcja dla latent_tsne
            html.Div([
                html.H3("Latent t-SNE"),
                dcc.Graph(id='latent-tsne', figure=self.create_3d_scatter(self._df, LATENT_TSNE_KEY, "Latent t-SNE"))
            ], style={'width': '48%', 'display': 'inline-block', 'padding': '0 20'}),

            # Sekcja wyświetlająca ścieżkę do pliku po kliknięciu
            html.Div(id='file-path-output', style={'marginTop': 20, 'fontSize': 18})
        ])

        # Callback do wyświetlania ścieżki do pliku po kliknięciu punktu na wykresie
        @self._app.callback(
            Output('file-path-output', 'children'),
            [Input('embedding-tsne', 'clickData'),
            Input('latent-tsne', 'clickData')]
        )
        def display_file_path(embedding_click, latent_click):
            # Wybierz odpowiedni punkt kliknięcia, jeśli istnieje
            clicked_data = embedding_click or latent_click
            if clicked_data:
                sample_path = clicked_data['points'][0]['text']
                return f"File Path: {sample_path}"
            return "Click on a point to see the file path."

    def visualize(self):
        self._prepare_dash()
        self._app.run_server(debug=True, host='0.0.0.0', port=8050)

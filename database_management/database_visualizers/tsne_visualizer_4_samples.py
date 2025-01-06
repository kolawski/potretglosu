"""Create a Dash app to visualize 3D t-SNE plots of the embedding and latent spaces."""
import dash
from dash import dcc, html, Input, Output
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

    def _prepare_dash(self, highlighted_paths):
        """
        Prepare the Dash application layout and callbacks for t-SNE visualization.

        :param highlighted_paths: List of file paths to highlight in the plots.
        :type highlighted_paths: list of str
        """
        # Initialize the Dash application
        self._app = dash.Dash(__name__)

        # Generate the color map
        color_map = {
            path: 'red' if path in highlighted_paths else 'rgba(169, 169, 169, 0.5)'
            # Red for highlighted, gray for others
            for path in self._df[SAMPLE_PATH_DB_KEY]
        }

        # Define the layout
        self._app.layout = html.Div([
            html.H1("3D t-SNE Visualization"),
            # Embedding t-SNE plot
            html.Div([
                html.H3("Embedding t-SNE"),
                dcc.Graph(id='embedding-tsne', figure=self.create_3d_scatter(
                    self._df, EMBEDDING_TSNE_KEY, "Embedding t-SNE", color_map))
            ], style={'width': '48%', 'display': 'inline-block', 'padding': '0 20'}),

            # Latent t-SNE plot
            html.Div([
                html.H3("Latent t-SNE"),
                dcc.Graph(id='latent-tsne', figure=self.create_3d_scatter(
                    self._df, LATENT_TSNE_KEY, "Latent t-SNE", color_map))
            ], style={'width': '48%', 'display': 'inline-block', 'padding': '0 20'}),

            # Display selected file path
            html.Div(id='file-path-output', style={'marginTop': 20, 'fontSize': 18})
        ])

        # Callback to display file path
        @self._app.callback(
            Output('file-path-output', 'children'),
            [Input('embedding-tsne', 'clickData'),
             Input('latent-tsne', 'clickData')]
        )
        def display_file_path(embedding_click, latent_click):
            clicked_data = embedding_click or latent_click
            if clicked_data:
                sample_path = clicked_data['points'][0]['text']
                return f"File Path: {sample_path}"
            return "Click on a point to see the file path."

    def visualize(self, highlighted_paths):
        """
        Prepare and run the Dash server to visualize the t-SNE plots.

        :param highlighted_paths: List of file paths to highlight in the plots.
        :type highlighted_paths: list of str
        """
        self._prepare_dash(highlighted_paths)
        self._app.run_server(debug=True, host='0.0.0.0', port=8050)


if __name__ == "__main__":
    # Lista ścieżek do wyróżnionych próbek
    # Są wybrane przez algorytm szukający najbardziej odległych punktów w oryginalnej przestrzeni
    highlighted_paths = [
        "/app/Resources/ready_audio_samples/common_voice_pl_22072821.wav",
        "/app/Resources/ready_audio_samples/common_voice_pl_25483212.wav",
        "/app/Resources/ready_audio_samples/common_voice_pl_20808053.wav",
        "/app/Resources/ready_audio_samples/common_voice_pl_32106416.wav"
    ]

    # Utwórz i uruchom wizualizację
    visualizer = TsneVisualizer()
    visualizer.visualize(highlighted_paths)


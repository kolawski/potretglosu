from dash import Dash, html, Input, Output, State, ALL, dcc
import dash
import os
from flask import send_from_directory
from data_loader import get_initial_samples
from scipy.spatial.distance import cdist
import numpy as np
from dash.exceptions import PreventUpdate

# Katalog współdzielony dla próbek audio
SHARED_DIR = "/app/shared"

# Inicjalizacja aplikacji Dash
app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# Pobierz początkowe dane
paths, indices, embedding_matrix = get_initial_samples()

# Zmienna globalna do przechowywania stanu iteracji
remaining_indices = list(range(len(embedding_matrix)))


# Funkcja do eksportu plików do katalogu współdzielonego
def export_audio_files(selected_files):
    exported_files = []
    for idx, file_path in enumerate(selected_files):
        output_name = f"sample_{idx}.wav"
        output_path = os.path.join(SHARED_DIR, output_name)
        os.makedirs(SHARED_DIR, exist_ok=True)
        os.system(f"cp {file_path} {output_path}")
        exported_files.append(output_name)
    return exported_files


# Layout aplikacji
app.layout = html.Div([
    html.H1("Iteracyjne Wybieranie Próbek Audio"),
    html.Div(id='audio-list'),  # Placeholder na listę audio
    html.Button("Rozpocznij Iterację", id='start-button', n_clicks=0),
    html.Div(id='selected-sample', style={'marginTop': '20px'}),
    html.Button("Dalej", id='next-button', n_clicks=0, disabled=True),
    dcc.Store(id='remaining-samples', data=remaining_indices),  # Przechowywanie stanu
])


# Serwowanie plików audio
@server.route('/audio/<path:filename>')
def serve_audio(filename):
    return send_from_directory(SHARED_DIR, filename)


# Callback do generowania listy audio
@app.callback(
    Output('audio-list', 'children'),
    Input('start-button', 'n_clicks'),
    State('remaining-samples', 'data')
)
def update_audio_list(n_clicks, remaining_indices):
    global embedding_matrix

    if n_clicks == 0:
        return html.P("Kliknij 'Rozpocznij Iterację', aby rozpocząć.")

    k = 4
    if len(remaining_indices) <= k:
        selected_indices = remaining_indices
    else:
        selected_indices = np.random.choice(remaining_indices, k, replace=False)

    selected_files = [paths[idx] for idx in selected_indices]
    exported_files = export_audio_files(selected_files)

    # Tworzenie listy audio w interfejsie
    return html.Div([
        html.Div([
            html.Audio(controls=True, src=f"/audio/{file}"),
            html.Button(f"Wybierz próbkę {idx + 1}", id={'type': 'choose-button', 'index': idx})
        ]) for idx, file in enumerate(exported_files)
    ])


# Callback do obsługi wyboru i przejścia do następnego kroku
@app.callback(
    [
        Output('remaining-samples', 'data'),
        Output('selected-sample', 'children'),
        Output('next-button', 'disabled')
    ],
    [
        Input({'type': 'choose-button', 'index': ALL}, 'n_clicks'),
        Input('next-button', 'n_clicks')
    ],
    [
        State('remaining-samples', 'data'),
        State('selected-sample', 'children')
    ]
)
def handle_buttons(n_clicks_choose, n_clicks_next, remaining_indices, chosen_sample_text):
    global embedding_matrix

    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if 'choose-button' in triggered_id:
        chosen_idx = n_clicks_choose.index(max(n_clicks_choose))
        chosen_index = remaining_indices[chosen_idx]
        chosen_sample_text = f"Wybrano próbkę: {paths[chosen_index]}"
        return remaining_indices, chosen_sample_text, False

    elif triggered_id == 'next-button':
        chosen_index = remaining_indices[0]  # Wybierz pierwszą próbkę jako domyślną
        chosen_embedding = embedding_matrix[chosen_index]
        remaining_embeddings = np.array([embedding_matrix[idx] for idx in remaining_indices])
        distances = cdist([chosen_embedding], remaining_embeddings, metric="euclidean")[0]

        half_size = len(remaining_indices) // 2
        nearest_indices = np.argsort(distances)[:half_size]
        remaining_indices = [remaining_indices[i] for i in nearest_indices]

        return remaining_indices, "Przejdź do następnego kroku", True

    raise PreventUpdate


# Uruchomienie serwera
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)

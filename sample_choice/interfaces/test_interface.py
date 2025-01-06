import dash
from dash import Dash, html, Input, Output, State, ctx, dcc
from flask import send_from_directory
import os
from pathlib import Path
import math

from database_management.database_managers.embedding_database_manager import EmbeddingDatabaseManager, EMBEDDING_KEY, \
    LATENT_KEY, SAMPLE_PATH_DB_KEY
#from utils.xtts_handler import XTTSHandler
from sample_choice.algorithm.gender_list import man_indexes, women_indexes

edb = EmbeddingDatabaseManager()
#xtts_handler = XTTSHandler()
embeddings = edb.get_all_values_from_column(EMBEDDING_KEY)
latents = edb.get_all_values_from_column(LATENT_KEY)
paths = edb.get_all_values_from_column(SAMPLE_PATH_DB_KEY)
points = [tuple(row) for row in embeddings]

NUM_STEPS = 10

# Inicjalizacja aplikacji Dash
app = Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([
    dcc.Store(id="user-input", data=""),  # Przechowywanie tekstu wpisanego przez użytkownika
    dcc.Store(id="user-gender", data=""),  # Przechowywanie wybranej płci
    dcc.Store(id="current-view", data="input"),  # Przechowywanie aktualnego widoku
    dcc.Store(id="current-step", data=1),  # Bieżący krok (zaczynamy od kroku 1)
    dcc.Store(id="selections-store", data=[None] * NUM_STEPS),  # Przechowywanie wyborów
    html.Div(id="app-content")  # Dynamiczne treści aplikacji
])
def choose_n_points_farthest(points, subset_indexes, n):

    # 1. Tworzymy "lokalny" zbiór punktów - tylko z subset_indexes
    points_subset = [points[i] for i in subset_indexes]
    N = len(points_subset)

    if n < 2:
        raise ValueError("n musi być co najmniej 2.")
    if n > N:
        raise ValueError("n nie może być większe niż liczba dostępnych punktów w podzbiorze.")

    # a) Znajdź najbardziej oddaloną parę w 'points_subset'
    max_dist = -1
    p1_idx, p2_idx = None, None
    for i in range(N):
        for j in range(i + 1, N):
            dist = math.dist(points_subset[i], points_subset[j])
            if dist > max_dist:
                max_dist = dist
                p1_idx, p2_idx = i, j

    chosen_local_idx = [p1_idx, p2_idx]

    # b) Dodawaj kolejne punkty (max-min distance)
    while len(chosen_local_idx) < n:
        best_candidate_idx = None
        best_min_dist = -1

        for k in range(N):
            if k in chosen_local_idx:
                continue
            d_min = min(math.dist(points_subset[k], points_subset[idx])
                        for idx in chosen_local_idx)
            if d_min > best_min_dist:
                best_min_dist = d_min
                best_candidate_idx = k

        chosen_local_idx.append(best_candidate_idx)

    # 3. Mapujemy indeksy lokalne (w podzbiorze) na indeksy globalne
    chosen_global_idx = [subset_indexes[i] for i in chosen_local_idx]
    return chosen_global_idx

# Callback do obsługi widoku aplikacji
@app.callback(
    Output("app-content", "children"),
    [Input("current-view", "data"),
     State("user-input", "data"),
     State("user-gender", "data"),
     State("current-step", "data")]
)
def render_view(current_view, user_input, user_gender, current_step):
    if current_view == "input":
        return html.Div([
            html.H1("Wprowadź swój tekst"),
            dcc.Input(
                id="user-text",
                type="text",
                placeholder="Wpisz tekst...",
                style={"margin-right": "10px"}
            ),
            html.Br(),
            html.H3("Tekst wypowiedział..."),
            dcc.RadioItems(
                id="user-gender-radio",
                options=[
                    {"label": "Mężczyzna", "value": "mężczyzna"},
                    {"label": "Kobieta", "value": "kobieta"}
                ],
                value="",
                style={"margin-top": "10px"}
            ),
            html.Button("OK", id="submit-text", n_clicks=0, style={"margin-top": "20px"})
        ])
    elif current_view == "main":
        # Załadowanie plików audio z katalogu krok0 (pierwszy krok = current_step=1 -> katalog krok0)
        audio_files_step_1 = [f for f in os.listdir("/app/shared/krok0") if f.endswith(".wav")]

        return html.Div([
            html.H1("Odtwarzanie Próbek Audio"),
            html.Div(f"Wprowadzony tekst: {user_input}", style={"font-size": "16px", "font-style": "italic", "margin-bottom": "5px"}),
            html.Div(f"Płeć: {user_gender}", style={"font-size": "16px", "font-style": "italic", "margin-bottom": "20px"}),
            html.H3(id="step-header", children="Krok 1"),
            html.Div(id="main-content", children=[
                html.P("Kliknij przycisk, aby wybrać próbkę:"),
                html.Ul([
                    html.Li([
                        f"Próbka {idx + 1}: ",
                        html.Audio(
                            controls=True,
                            src=f"/audio/1/{file}"
                        ),
                        html.Button(
                            "Wybierz tę próbkę",
                            id={"type": "select-button", "index": idx + 1},
                            n_clicks=0
                        )
                    ], style={"margin-bottom": "10px"}) for idx, file in enumerate(audio_files_step_1)
                ]),
                html.H3("Wybrana próbka:"),
                html.Div(id="selected-sample", style={"font-size": "18px", "font-weight": "bold"}),
                html.Br(),
                html.Button("Dalej", id="next-button", n_clicks=0, style={"margin-left": "10px"}, disabled=True)
            ])
        ])

# Callback do obsługi przejścia do głównego widoku
@app.callback(
    [Output("current-view", "data"),
     Output("user-input", "data"),
     Output("user-gender", "data")],
    Input("submit-text", "n_clicks"),
    [State("user-text", "value"),
     State("user-gender-radio", "value")],
    prevent_initial_call=True
)
def submit_user_text(n_clicks, user_text, user_gender):
    #global remaining_indices
    #global ratio
    if user_text and user_gender:
        remaining_indices = []
        ratio = 0
        if user_gender == 'mężczyzna':
            remaining_indices = man_indexes.copy()
            ratio = 0.5
        elif user_gender == 'kobieta':
            remaining_indices = women_indexes.copy()
            ratio = 0.65
        # selected1 = choose_n_points_farthest(points, remaining_indices, 8)
        # for i in range(len(selected1)):
        #     embedding = embeddings.iloc[selected1[i]]
        #     latent = latents.iloc[selected1[i]]
        #     folder_path = Path(f"/app/shared/krok11")
        #     folder_path.mkdir(parents=True, exist_ok=True)
        #     xtts_handler.inference(embedding, latent, f"/app/shared/krok11/test{i}.wav", user_text)
        return "main", user_text, user_gender
    return dash.no_update, dash.no_update, dash.no_update

# Serwowanie plików audio z dynamiczną ścieżką
server = app.server

@server.route('/audio/<int:step>/<path:filename>')
def serve_audio(step, filename):
    return send_from_directory(f"/app/shared/krok{step-1}", filename)

# Callback do obsługi wyboru próbki, nawigacji i aktywacji przycisku "Dalej"
@app.callback(
    [Output("main-content", "children"),
     Output("step-header", "children"),
     Output("current-step", "data"),
     Output("selections-store", "data"),
     Output("selected-sample", "children"),
     Output("next-button", "disabled")],
    [Input("next-button", "n_clicks"),
     Input({"type": "select-button", "index": dash.ALL}, "n_clicks")],
    [State("selections-store", "data"),
     State("current-step", "data")]
)
def manage_steps(n_clicks_next, select_clicks, selections_data, current_step):
    triggered = ctx.triggered_id
    selections = selections_data.copy()
    selected_sample_text = ""
    next_disabled = True  # Domyślnie przycisk "Dalej" jest nieaktywny

    if current_step <= NUM_STEPS:
        audio_files_current = [f for f in os.listdir(f"/app/shared/krok{current_step-1}") if f.endswith(".wav")]
    else:
        audio_files_current = []

    # Obsługa wyboru próbki
    if isinstance(triggered, dict) and "index" in triggered:
        sample_index = triggered["index"]
        # Zapamiętujemy wybraną próbkę (numer)
        selections[current_step - 1] = sample_index
        selected_sample_text = f"Próbka {sample_index}"
        next_disabled = False

    # Obsługa przycisku "Dalej"
    if triggered == "next-button" and current_step < NUM_STEPS + 1:
        current_step += 1
        if current_step <= NUM_STEPS:
            audio_files_current = [f for f in os.listdir(f"/app/shared/krok{current_step-1}") if f.endswith(".wav")]

    if current_step == NUM_STEPS:
        next_button_text = "Zakończ"
    else:
        next_button_text = "Dalej"

    # Jeśli przekroczono liczbę kroków, pokaż tylko ostatnią wybraną próbkę
    if current_step > NUM_STEPS:
        last_selection = selections[-1]
        audio_files_last_step = [f for f in os.listdir(f"/app/shared/krok{NUM_STEPS-1}") if f.endswith(".wav")]

        if last_selection:
            final_sample_index = last_selection
            final_sample_filename = audio_files_last_step[final_sample_index - 1]
            summary_content = html.Div([
                html.H1("Ostatecznie wybrana próbka"),
                html.P(f"Krok {NUM_STEPS}, Próbka {final_sample_index}: {final_sample_filename}"),
                html.Audio(
                    controls=True,
                    src=f"/audio/{NUM_STEPS}/{final_sample_filename}"
                )
            ])
        else:
            summary_content = html.Div([
                html.H1("Brak wybranej próbki"),
                html.P("Nie dokonano wyboru w ostatnim kroku.")
            ])
        return summary_content, "Podsumowanie", current_step, selections, "", True

    # Ustal, czy wybrano już próbkę w bieżącym kroku
    selected_sample_number = selections[current_step - 1] if current_step <= NUM_STEPS else None
    if selected_sample_number:
        selected_sample_text = f"Próbka {selected_sample_number}"
        next_disabled = False

    content = html.Div([
        html.P("Kliknij przycisk, aby wybrać próbkę:"),
        html.Ul([
            html.Li([
                f"Próbka {idx + 1}: ",
                html.Audio(
                    controls=True,
                    src=f"/audio/{current_step}/{file}"
                ),
                html.Button(
                    "Wybierz tę próbkę",
                    id={"type": "select-button", "index": idx + 1},
                    n_clicks=0,
                    style={"background-color": "#d3d3d3"} if selected_sample_number == idx + 1 else {}
                )
            ], style={"margin-bottom": "10px"}) for idx, file in enumerate(audio_files_current)
        ]),
        html.H3("Wybrana próbka:"),
        html.Div(
            selected_sample_text,
            id="selected-sample",
            style={"font-size": "18px", "font-weight": "bold"}
        ),
        html.Br(),
        html.Button(next_button_text, id="next-button", n_clicks=n_clicks_next, style={"margin-left": "10px"},
                    disabled=next_disabled)
    ])

    return content, f"Krok {current_step if current_step <= NUM_STEPS else NUM_STEPS}", current_step, selections, selected_sample_text, next_disabled

if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0', port=8050)

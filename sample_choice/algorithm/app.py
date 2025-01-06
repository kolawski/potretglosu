import dash
from dash import Dash, html, Input, Output, State, ctx, dcc
from flask import send_from_directory
import os
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from pathlib import Path

from database_management.database_managers.embedding_database_manager import EmbeddingDatabaseManager, EMBEDDING_KEY, LATENT_KEY
from settings import SAMPLE_PATH_DB_KEY
from utils.xtts_handler import XTTSHandler

# Liczba kroków
NUM_STEPS = 10

# Inicjalizacja aplikacji Dash
app = Dash(__name__, suppress_callback_exceptions=True)

# Serwer Flask do serwowania audio
server = app.server

# Inicjalizacja baz i handlerów
edb = EmbeddingDatabaseManager()
xtts_handler = XTTSHandler()

embeddings = edb.get_all_values_from_column(EMBEDDING_KEY)
latents = edb.get_all_values_from_column(LATENT_KEY)
paths = edb.get_all_values_from_column(SAMPLE_PATH_DB_KEY)

# Listy indeksów dla mężczyzn i kobiet
man_indexes = [...]  # tutaj wstawiamy dokładnie tę samą listę co w drugim kodzie
women_indexes = [...]  # analogicznie

# Funkcje pomocnicze

def get_centroids(iteration, min_b, max_b):
    mid_x = (min_b[0] + max_b[0]) / 2
    mid_y = (min_b[1] + max_b[1]) / 2
    mid_z = (min_b[2] + max_b[2]) / 2

    # Zgodnie z logiką z drugiego kodu
    if iteration == 0:
        centroids = [
            [min_b[0], min_b[1], min_b[2]],
            [min_b[0], min_b[1], max_b[2]],
            [min_b[0], max_b[1], min_b[2]],
            [min_b[0], max_b[1], max_b[2]],
            [max_b[0], min_b[1], min_b[2]],
            [max_b[0], min_b[1], max_b[2]],
            [max_b[0], max_b[1], min_b[2]],
            [max_b[0], max_b[1], max_b[2]],
        ]
    elif iteration == 1:
        centroids = [
            [min_b[0], min_b[1], mid_z],
            [min_b[0], max_b[1], mid_z],
            [mid_x, min_b[1], mid_z],
            [mid_x, max_b[1], mid_z],
            [max_b[0], min_b[1], mid_z],
            [max_b[0], max_b[1], mid_z],
        ]
    elif 2 <= iteration <= 6:
        centroids = [
            [mid_x, min_b[1], min_b[2]],
            [mid_x, max_b[1], min_b[2]],
            [mid_x, min_b[1], max_b[2]],
            [mid_x, max_b[1], max_b[2]],
        ]
    elif 7 <= iteration <= 8:
        centroids = [
            [min_b[0], mid_y, mid_z],
            [mid_x, max_b[1], mid_z],
            [max_b[0], mid_y, mid_z],
        ]
    else:
        centroids = [
            [mid_x, min_b[1], mid_z],
            [mid_x, max_b[1], mid_z],
        ]
    return centroids

def find_nearest_indices(remaining_indices, centroids, pca, latents_matrix):
    remaining_latents_3d = pca.transform(latents_matrix[remaining_indices])
    selected_indices = []
    for centroid in centroids:
        distances = cdist(remaining_latents_3d, [centroid])
        nearest_index = np.argmin(distances)
        selected_indices.append(nearest_index)
    return selected_indices

def generate_samples_for_step(step, selected_indices, remaining_indices, embeddings, latents, user_text):
    # Tworzymy folder dla kroków
    folder_path = Path(f"/app/shared/krok{step}")
    folder_path.mkdir(parents=True, exist_ok=True)

    # Czyszczimy folder z poprzednich plików, jeśli takie są
    for f in folder_path.glob("*.wav"):
        f.unlink()

    current_points = [remaining_indices[idx] for idx in selected_indices]
    # Generujemy próbki audio
    for i, point_idx in enumerate(current_points):
        embedding = embeddings.iloc[point_idx]
        latent = latents.iloc[point_idx]
        xtts_handler.inference(embedding, latent, f"/app/shared/krok{step}/test{i}.wav", user_text)
    return current_points

def update_remaining_indices(chosen_point, remaining_indices, latents_matrix, ratio):
    distances = cdist([latents_matrix[chosen_point]], latents_matrix[remaining_indices]).flatten()
    sorted_indices = np.argsort(distances)
    count = int(len(sorted_indices) * ratio)
    closest_indices = sorted_indices[:count]
    remaining_indices = [remaining_indices[i] for i in closest_indices]
    return remaining_indices

# Layout aplikacji
app.layout = html.Div([
    dcc.Store(id="user-input", data=""),        # Przechowywanie tekstu wpisanego przez użytkownika
    dcc.Store(id="user-gender", data=""),       # Przechowywanie wybranej płci
    dcc.Store(id="current-view", data="input"), # Przechowywanie aktualnego widoku
    dcc.Store(id="remaining-indices", data=[]), # Przechowywanie pozostałych indeksów do przetwarzania
    dcc.Store(id="iteration", data=0),          # Bieżąca iteracja (0-9)
    dcc.Store(id="ratio", data=0.5),            # Domyślny ratio dla mężczyzn, 0.65 dla kobiet ustawiany dynamicznie
    dcc.Store(id="selected-indices", data=[]),  # Indeksy wybrane w danym kroku
    dcc.Store(id="current-points", data=[]),    # Aktualnie wybrane punkty do wyświetlenia
    dcc.Store(id="final-chosen", data=None),    # Ostatecznie wybrana próbka
    dcc.Store(id="latents-matrix", data=[]),    # Przechowywanie latents_matrix jako listy
    dcc.Store(id="pca-components", data={}),     # Przechowywanie PCA (w praktyce tylko składowe i mean)
    html.Div(id="app-content")  # Dynamiczne treści aplikacji
])

@app.callback(
    Output("app-content", "children"),
    [Input("current-view", "data"),
     State("user-input", "data"),
     State("user-gender", "data")]
)
def render_view(current_view, user_input, user_gender):
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
        return html.Div([
            html.H1("Odtwarzanie Próbek Audio"),
            html.Div(f"Wprowadzony tekst: {user_input}", style={"font-size": "16px", "font-style": "italic", "margin-bottom": "5px"}),
            html.Div(f"Płeć: {user_gender}", style={"font-size": "16px", "font-style": "italic", "margin-bottom": "20px"}),
            html.H3(id="step-header", children="Krok 1"),
            dcc.Store(id="selections-store", data=[None] * NUM_STEPS),  # Przechowywanie wyborów użytkownika
            dcc.Store(id="current-step", data=1),  # Bieżący krok od 1 do 10
            html.Div(id="main-content")
        ])

    return html.Div("")

@app.callback(
    [Output("current-view", "data"),
     Output("user-input", "data"),
     Output("user-gender", "data"),
     Output("remaining-indices", "data"),
     Output("ratio", "data"),
     Output("latents-matrix", "data"),
     Output("pca-components", "data")],
    Input("submit-text", "n_clicks"),
    [State("user-text", "value"),
     State("user-gender-radio", "value")],
    prevent_initial_call=True
)
def submit_user_text(n_clicks, user_text, user_gender):
    if user_text and user_gender:
        # Inicjujemy dane zgodnie z wyborem płci
        if user_gender == "mężczyzna":
            remaining_indices = man_indexes.copy()
            ratio = 0.5
        else:
            remaining_indices = women_indexes.copy()
            ratio = 0.65

        # Przygotowanie PCA dla wybranej grupy
        latents_matrix = np.vstack(latents.values)
        # PCA wstępne
        pca = PCA(n_components=3)
        pca.fit(latents_matrix[remaining_indices])
        # Przechowamy składowe PCA do późniejszego użycia (transformacja)
        pca_components = {
            "mean_": pca.mean_.tolist(),
            "components_": pca.components_.tolist()
        }

        return "main", user_text, user_gender, remaining_indices, ratio, latents_matrix.tolist(), pca_components
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

@server.route('/audio/<int:step>/<path:filename>')
def serve_audio(step, filename):
    directory = f"/app/shared/krok{step}"
    return send_from_directory(directory, filename)

@app.callback(
    [Output("main-content", "children"),
     Output("step-header", "children"),
     Output("current-step", "data"),
     Output("selections-store", "data"),
     Output("selected-sample", "children"),
     Output("next-button", "disabled"),
     Output("remaining-indices", "data"),
     Output("iteration", "data"),
     Output("pca-components", "data"),
     Output("current-points", "data"),
     Output("final-chosen", "data")],
    [Input("next-button", "n_clicks"),
     Input({"type": "select-button", "index": dash.ALL}, "n_clicks")],
    [State("selections-store", "data"),
     State("current-step", "data"),
     State("remaining-indices", "data"),
     State("iteration", "data"),
     State("ratio", "data"),
     State("latents-matrix", "data"),
     State("pca-components", "data"),
     State("user-input", "data"),
     State("user-gender", "data")]
, prevent_initial_call=True)
def manage_steps(n_clicks_next, select_clicks, selections_data, current_step, remaining_indices, iteration, ratio, latents_matrix_list, pca_components, user_text, user_gender):
    triggered = ctx.triggered_id
    selections = selections_data.copy()
    selected_sample_text = ""
    next_disabled = True  # Domyślnie przycisk "Dalej" jest nieaktywny
    final_chosen = dash.no_update

    # Odtworzenie PCA
    pca = PCA(n_components=3)
    pca.mean_ = np.array(pca_components["mean_"])
    pca.components_ = np.array(pca_components["components_"])
    latents_matrix = np.array(latents_matrix_list)

    # Funkcja do aktualizacji centroidów i wybranych indeksów
    def update_selection(iteration, remaining_indices, latents_matrix):
        # Aktualizacja PCA dla aktualnych remaining_indices
        pca_local = PCA(n_components=3)
        pca_local.fit(latents_matrix[remaining_indices])
        group_latents_3d = pca_local.transform(latents_matrix[remaining_indices])
        min_bounds = np.min(group_latents_3d, axis=0)
        max_bounds = np.max(group_latents_3d, axis=0)
        centroids = get_centroids(iteration, min_bounds, max_bounds)
        selected_indices = find_nearest_indices(remaining_indices, centroids, pca_local, latents_matrix)
        # Zwracamy też komponenty nowego PCA
        pca_local_components = {
            "mean_": pca_local.mean_.tolist(),
            "components_": pca_local.components_.tolist()
        }
        return selected_indices, pca_local_components

    # Jeżeli użytkownik wybrał próbkę (przycisk "Wybierz tę próbkę")
    if isinstance(triggered, dict) and "index" in triggered:
        sample_index = triggered["index"]
        selections[current_step - 1] = sample_index
        selected_sample_text = f"Próbka {sample_index}"
        next_disabled = False
        return (dash.no_update, dash.no_update, dash.no_update, selections,
                selected_sample_text, next_disabled, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update)

    # Jeżeli użytkownik kliknął "Dalej"
    if triggered == "next-button":
        # Sprawdzamy, czy to nie koniec
        if current_step == NUM_STEPS:
            # To był ostatni krok, pokazujemy podsumowanie
            summary_content = html.Div([
                html.H1("Podsumowanie wybranych próbek"),
                html.Ul([
                    html.Li(f"Krok {idx + 1}: Próbka {sample}" if sample else f"Krok {idx + 1}: Nie wybrano")
                    for idx, sample in enumerate(selections)
                ]),
                html.Br(),
                html.Div("Proces zakończony. Wybrana próbka to ta z ostatniego kroku.")
            ])
            # Ostatnia wybrana próbka:
            last_choice = selections[-1]
            # Mamy current_points w tym kroku, musimy ustalić który indeks globalny został wybrany
            # Niestety current_points będzie wyliczane w logice przed generacją UI.
            # Załóżmy, że za każdym razem generujemy current_points przed renderem.
            # Skoro user przeszedł "dalej" bez nowego wyboru, current_points jest z poprzedniego stanu
            # Musimy więc mieć current_points zapisane w stanie. Mieliśmy w store "current-points", aktualizujemy go przy każdym kroku.
            # last_choice to numer próbki (1-based) wybranej przez usera w danym kroku
            # sprawdzimy w selections co jest w ostatnim elemencie:
            final_choice_number = selections[-1]
            if not final_choice_number:
                final_chosen = None
            else:
                # Musimy mieć current_points z poprzedniego kroku. Zanim user kliknął "Dalej", current_points zostało wygenerowane.
                # Niestety w tym momencie jest już za późno - user jest na summary_view.
                # Rozwiązanie: final_chosen ustawimy w poprzednim kroku, gdy user wybierze próbkę i kliknie "Dalej" w ostatnim kroku.
                # Zakładamy, że final_chosen zostanie ustalone poniżej przy zamknięciu pętli (iteration == 9).
                pass

            return (summary_content,
                    "Podsumowanie",
                    current_step,
                    selections,
                    "",
                    True,
                    remaining_indices,
                    iteration,
                    pca_components,
                    [],
                    final_chosen)

        # Jeśli to nie koniec, przechodzimy do kolejnej iteracji
        # Najpierw aktualizujemy remaining_indices na podstawie wyboru usera w poprzednim kroku
        chosen_sample_number = selections[current_step - 1]
        if chosen_sample_number is None:
            # Użytkownik nie wybrał próbki, nie możemy iść dalej
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, True, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

        # Mamy chosen_sample_number, to numer próbki (1-based) wybranej w aktualnym kroku
        # Musimy mapować go na globalny indeks. current_points zawiera globalne indeksy próbek
        # current_points należy wczytać przed generacją contentu:
        # Ale my generujemy current_points na podstawie wybranych indeksów. Zrobimy to teraz.

        # Odzyskanie poprzedniego kroku iteration
        # Każdy krok to iteration od 0 do 9
        # current_step to np. 1 => iteration=0, current_step=2 => iteration=1, ...
        # iteration odpowiada current_step - 1
        chosen_iteration = current_step - 1

        # W iteration mamy bieżącą iterację, ale "Dalej" kliknięto po wyborze, więc iteration jest jeszcze tą starą
        # Spróbujemy odtworzyć current_points za pomocą saved "selected_indices" i "remaining_indices"
        # Musimy je jednak mieć. Wyliczymy je tuż przed renderem zawartości w tym callbacku.

        # Jednak logika jest taka:
        # 1. Przy wejściu do kroku generujemy audio i current_points.
        # 2. Użytkownik wybiera próbkę -> zapamiętujemy w selections.
        # 3. Po kliknięciu "Dalej" wykonujemy filtrację remaining_indices.

        # Załóżmy, że current_points jest już aktualne w stanie (zapewniamy to przy generowaniu contentu niżej)

        # Odczytujemy current_points ze stanu (z poprzedniego renderu)
        current_points_state = dash.get_triggered()[0]["state"] if dash.get_triggered() else None
        # Niestety ctx nie daje nam direct state w prosty sposób - mamy go w state arguments
        # Ale mamy current-points store i updates poniżej, więc skorzystamy z parametru stanu callbacka.

        # Przypisujemy current_points z parametru stanu callbacka:
        # W tym momencie callback ma stan i wejścia w arguments - weźmy je z arguments:
        # W definicji callbacka jest State("current-points", "data") - zapomnieliśmy dodać w definicji.
        # Dodajemy w definicji callbacka State("current-points", "data") aby mieć dostęp do current_points.
        # Uaktualniamy callback o ten stan:

    # Poprawka w definicji callbacka:
    # Dodamy do states w definicji callbacka:
    # State("current-points", "data")

# ZMIANA: Dodajemy do callbacka State("current-points", "data") i State("final-chosen", "data")

@app.callback(
    [Output("main-content", "children"),
     Output("step-header", "children"),
     Output("current-step", "data"),
     Output("selections-store", "data"),
     Output("selected-sample", "children"),
     Output("next-button", "disabled"),
     Output("remaining-indices", "data"),
     Output("iteration", "data"),
     Output("pca-components", "data"),
     Output("current-points", "data"),
     Output("final-chosen", "data")],
    [Input("next-button", "n_clicks"),
     Input({"type": "select-button", "index": dash.ALL}, "n_clicks")],
    [State("selections-store", "data"),
     State("current-step", "data"),
     State("remaining-indices", "data"),
     State("iteration", "data"),
     State("ratio", "data"),
     State("latents-matrix", "data"),
     State("pca-components", "data"),
     State("user-input", "data"),
     State("user-gender", "data"),
     State("current-points", "data"),
     State("final-chosen", "data")],
    prevent_initial_call=True
)
def manage_steps(n_clicks_next, select_clicks, selections_data, current_step, remaining_indices, iteration, ratio, latents_matrix_list, pca_components, user_text, user_gender, current_points, final_chosen):
    triggered = ctx.triggered_id
    selections = selections_data.copy()
    selected_sample_text = ""
    next_disabled = True

    # Odtworzenie PCA
    pca = PCA(n_components=3)
    pca.mean_ = np.array(pca_components["mean_"])
    pca.components_ = np.array(pca_components["components_"])
    latents_matrix = np.array(latents_matrix_list)

    # Obsługa wyboru próbki
    if isinstance(triggered, dict) and "index" in triggered:
        sample_index = triggered["index"]
        selections[current_step - 1] = sample_index
        selected_sample_text = f"Próbka {sample_index}"
        next_disabled = False
        return (dash.no_update, dash.no_update, dash.no_update, selections,
                selected_sample_text, next_disabled, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update)

    # Obsługa przycisku "Dalej"
    if triggered == "next-button":
        # Jeśli to ostatni krok, wyświetlamy podsumowanie
        if current_step == NUM_STEPS:
            # Ostateczna próbka:
            chosen_sample_number = selections[-1]
            if chosen_sample_number is not None and current_points:
                final_global_idx = current_points[chosen_sample_number - 1]
                final_chosen = final_global_idx

            summary_content = html.Div([
                html.H1("Podsumowanie wybranych próbek"),
                html.Ul([
                    html.Li(f"Krok {idx + 1}: Próbka {sample}" if sample else f"Krok {idx + 1}: Nie wybrano")
                    for idx, sample in enumerate(selections)
                ]),
                html.Br(),
                html.H2("Wybrana finalna próbka:"),
                html.Div(f"Indeks próbki: {final_chosen}", style={"font-weight": "bold"}),
                html.Div(f"Ścieżka: {paths.iloc[final_chosen]}", style={"font-weight": "bold"})
            ])
            return (summary_content, "Podsumowanie", current_step, selections, "", True, remaining_indices,
                    iteration, pca_components, [], final_chosen)

        # Inaczej przechodzimy do następnej iteracji
        chosen_sample_number = selections[current_step - 1]
        if chosen_sample_number is None:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, True, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

        chosen_global_idx = current_points[chosen_sample_number - 1]
        # Aktualizujemy remaining_indices
        remaining_indices = update_remaining_indices(chosen_global_idx, remaining_indices, latents_matrix, ratio)

        # Aktualizacja PCA i centroidów dla nowej iteracji
        new_iteration = iteration + 1
        pca_local = PCA(n_components=3)
        pca_local.fit(latents_matrix[remaining_indices])
        group_latents_3d = pca_local.transform(latents_matrix[remaining_indices])
        min_bounds = np.min(group_latents_3d, axis=0)
        max_bounds = np.max(group_latents_3d, axis=0)
        centroids = get_centroids(new_iteration, min_bounds, max_bounds)
        selected_indices = find_nearest_indices(remaining_indices, centroids, pca_local, latents_matrix)

        # Zapisujemy nowe PCA składowe
        pca_local_components = {
            "mean_": pca_local.mean_.tolist(),
            "components_": pca_local.components_.tolist()
        }

        # Generujemy nowe próbki audio dla kolejnego kroku (current_step + 1)
        new_current_points = generate_samples_for_step(current_step, selected_indices, remaining_indices, embeddings, latents, user_text)

        # Ustawiamy next_button domyślnie na disabled, dopóki user nie wybierze próbki
        next_disabled = True
        selected_sample_text = ""

        new_step = current_step + 1
        if new_step == NUM_STEPS:
            next_button_text = "Zakończ"
        else:
            next_button_text = "Dalej"

        # Tworzymy content dla nowego kroku
        # Wczytujemy pliki audio dla nowego kroku
        current_step_dir = f"/app/shared/krok{current_step}"
        audio_files = [f for f in os.listdir(current_step_dir) if f.endswith(".wav")]

        # Generujemy listę próbek dla bieżącego kroku
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
                        style={"background-color": "#d3d3d3"}
                    )
                ], style={"margin-bottom": "10px"}) for idx, file in enumerate(audio_files)
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

        return (content, f"Krok {new_step}", new_step, selections, selected_sample_text, next_disabled,
                remaining_indices, new_iteration, pca_local_components, new_current_points, final_chosen)

    # Jeśli żadna akcja nie zaszła (pierwsze wywołanie "main")
    # To znaczy renderujemy pierwszy krok
    # Iteracja = 0
    iteration = 0
    pca_local = PCA(n_components=3)
    pca_local.fit(latents_matrix[remaining_indices])
    group_latents_3d = pca_local.transform(latents_matrix[remaining_indices])
    min_bounds = np.min(group_latents_3d, axis=0)
    max_bounds = np.max(group_latents_3d, axis=0)
    centroids = get_centroids(iteration, min_bounds, max_bounds)
    selected_indices = find_nearest_indices(remaining_indices, centroids, pca_local, latents_matrix)

    # Zapisujemy nowe PCA składowe
    pca_local_components = {
        "mean_": pca_local.mean_.tolist(),
        "components_": pca_local.components_.tolist()
    }

    # Generacja pierwszych próbek
    new_current_points = generate_samples_for_step(1, selected_indices, remaining_indices, embeddings, latents, user_text)
    current_step_dir = f"/app/shared/krok1"
    audio_files = [f for f in os.listdir(current_step_dir) if f.endswith(".wav")]

    content = html.Div([
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
                    n_clicks=0,
                    style={"background-color": "#d3d3d3"}
                )
            ], style={"margin-bottom": "10px"}) for idx, file in enumerate(audio_files)
        ]),
        html.H3("Wybrana próbka:"),
        html.Div(
            "",
            id="selected-sample",
            style={"font-size": "18px", "font-weight": "bold"}
        ),
        html.Br(),
        html.Button("Dalej", id="next-button", n_clicks=0, style={"margin-left": "10px"}, disabled=True)
    ])

    return (content, "Krok 1", 1, [None]*NUM_STEPS, "", True,
            remaining_indices, iteration, pca_local_components, new_current_points, final_chosen)


if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0', port=8050)

import dash
from dash import Dash, html, Input, Output, State, ctx, dcc
from flask import send_from_directory
import os

# Katalog z próbkami audio
SHARED_DIR = "/app/shared/krok0"

# Liczba kroków
NUM_STEPS = 5

# Inicjalizacja aplikacji Dash
app = Dash(__name__, suppress_callback_exceptions=True)

# Pobierz listę plików audio
audio_files = [f for f in os.listdir(SHARED_DIR) if f.endswith(".wav")]

# Layout aplikacji
app.layout = html.Div([
    dcc.Store(id="user-input", data=""),  # Przechowywanie tekstu wpisanego przez użytkownika
    dcc.Store(id="user-gender", data=""),  # Przechowywanie wybranej płci
    dcc.Store(id="current-view", data="input"),  # Przechowywanie aktualnego widoku
    html.Div(id="app-content")  # Dynamiczne treści aplikacji
])

# Callback do obsługi widoku aplikacji
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
            dcc.Store(id="selections-store", data=[None] * NUM_STEPS),  # Przechowywanie wyborów (5 kroków)
            dcc.Store(id="current-step", data=1),  # Bieżący krok
            html.Div(id="main-content", children=[
                html.P("Kliknij przycisk, aby wybrać próbkę:"),
                html.Ul([
                    html.Li([
                        f"Próbka {idx + 1}: ",
                        html.Audio(
                            controls=True,
                            src=f"/audio/{file}"
                        ),
                        html.Button(
                            "Wybierz tę próbkę",
                            id={"type": "select-button", "index": idx + 1},
                            n_clicks=0
                        )
                    ], style={"margin-bottom": "10px"}) for idx, file in enumerate(audio_files)
                ]),
                html.H3("Wybrana próbka:"),
                html.Div(id="selected-sample", style={"font-size": "18px", "font-weight": "bold"}),
                html.Br(),
                html.Button("Dalej", id="next-button", n_clicks=0, style={"margin-left": "10px"}, disabled=True)  # Początkowo dezaktywowany
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
    if user_text and user_gender:
        return "main", user_text, user_gender
    return dash.no_update, dash.no_update, dash.no_update

# Serwowanie plików audio
server = app.server

@server.route('/audio/<path:filename>')
def serve_audio(filename):
    return send_from_directory(SHARED_DIR, filename)


# Callback do obsługi wyboru próbki, nawigacji i aktywacji przycisku "Dalej"
@app.callback(
    [Output("main-content", "children"),
     Output("step-header", "children"),
     Output("current-step", "data"),
     Output("selections-store", "data"),
     Output("selected-sample", "children"),
     Output("next-button", "disabled")],  # Dynamiczne zarządzanie przyciskiem
    [Input("next-button", "n_clicks"),
     Input({"type": "select-button", "index": dash.ALL}, "n_clicks")],
    [State("selections-store", "data"),
     State("current-step", "data")],
    prevent_initial_call=True
)
def manage_steps(n_clicks_next, select_clicks, selections_data, current_step):
    triggered = ctx.triggered_id
    selections = selections_data.copy()
    selected_sample_text = ""
    next_disabled = True  # Domyślnie przycisk "Dalej" jest nieaktywny

    # Obsługa wyboru próbki
    if isinstance(triggered, dict) and "index" in triggered:
        sample_index = triggered["index"]
        selections[current_step - 1] = sample_index
        selected_sample_text = f"Próbka {sample_index}"
        next_disabled = False  # Aktywuj przycisk po wyborze próbki
        return dash.no_update, dash.no_update, dash.no_update, selections, selected_sample_text, next_disabled

    # Obsługa przycisków "Dalej" i "Poprzedni krok"
    if triggered == "next-button" and current_step < NUM_STEPS + 1:
        current_step += 1

    if current_step == NUM_STEPS:
        next_button_text = "Zakończ"
    else:
        next_button_text = "Dalej"

    # Jeśli zakończono wybieranie (krok 6), pokaż podsumowanie
    if current_step > NUM_STEPS:
        summary_content = html.Div([
            html.H1("Podsumowanie wybranych próbek"),
            html.Ul([
                html.Li(f"Krok {idx + 1}: Próbka {sample}" if sample else f"Krok {idx + 1}: Nie wybrano")
                for idx, sample in enumerate(selections)
            ]),
            html.Br()
        ])
        return summary_content, "Podsumowanie", current_step, selections, "", True

    # Generuj listę próbek z zaznaczoną ostatnio wybraną próbką
    selected_sample_number = selections[current_step - 1] if current_step <= NUM_STEPS else None
    if selected_sample_number:
        selected_sample_text = f"Próbka {selected_sample_number}"
        next_disabled = False  # Aktywuj przycisk, jeśli próbka jest wybrana

    content = html.Div([
        html.P("Kliknij przycisk, aby wybrać próbkę:"),
        html.Ul([
            html.Li([
                f"Próbka {idx + 1}: ",
                html.Audio(
                    controls=True,
                    src=f"/audio/{file}"
                ),
                html.Button(
                    "Wybierz tę próbkę",
                    id={"type": "select-button", "index": idx + 1},
                    n_clicks=0,
                    style={"background-color": "#d3d3d3"} if selected_sample_number == idx + 1 else {}
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

    return content, f"Krok {current_step if current_step <= NUM_STEPS else NUM_STEPS}", current_step, selections, selected_sample_text, next_disabled


# Uruchom serwer
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)

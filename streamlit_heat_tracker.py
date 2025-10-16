# streamlit_heat_tracker.py
# Streamlit app to track and forecast heating consumption with persistent local CSV and config.json storage

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, date
import os, json

st.set_page_config(page_title="Heat Consumption Tracker", layout="wide")

DATA_FILE = 'data.csv'
CONFIG_FILE = 'config.json'

# ----- Helpers -----

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {'prev_season_total': 0.0}
    else:
        return {'prev_season_total': 0.0}


def save_config(config):
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def init_session_state():
    config = load_config()
    if 'areas' not in st.session_state:
        st.session_state.areas = ['Room 1', 'Room 2', 'Room 3']
    if 'readings' not in st.session_state:
        if os.path.exists(DATA_FILE):
            try:
                df = pd.read_csv(DATA_FILE)
                st.session_state.readings = df.to_dict(orient='records')
                st.success('Dane wczytane z pliku data.csv')
            except Exception as e:
                st.session_state.readings = []
                st.warning(f'Nie udało się wczytać danych: {e}')
        else:
            st.session_state.readings = []
    if 'prev_season_total' not in st.session_state:
        st.session_state.prev_season_total = config.get('prev_season_total', 0.0)


def save_data_to_file():
    df = pd.DataFrame(st.session_state.readings)
    df.to_csv(DATA_FILE, index=False)


@st.cache_data
def to_dataframe(readings):
    if not readings:
        return pd.DataFrame(columns=['area', 'date', 'value'])
    df = pd.DataFrame(readings)
    df['date'] = pd.to_datetime(df['date']).dt.date
    df = df.sort_values(['area','date']).reset_index(drop=True)
    return df


def add_area(name):
    name = name.strip()
    if name and name not in st.session_state.areas:
        st.session_state.areas.append(name)


def remove_area(name):
    if name in st.session_state.areas:
        st.session_state.areas.remove(name)
        st.session_state.readings = [r for r in st.session_state.readings if r['area'] != name]
        save_data_to_file()
        st.success(f'Usunięto obszar: {name}')


def add_reading(area, reading_date, value):
    try:
        val = float(value)
    except Exception:
        st.error('Wartość odczytu musi być liczbą.')
        return
    st.session_state.readings.append({'area': area, 'date': reading_date.isoformat(), 'value': val})
    save_data_to_file()
    st.success(f'Dodano odczyt dla {area} — {reading_date} : {val}')


def clear_all_data():
    st.session_state.readings = []
    if os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)
    st.success('Wszystkie odczyty usunięte i plik data.csv skasowany.')


def compute_consumption(df, season_start, season_end):
    results = {}
    for area, g in df.groupby('area'):
        g = g.sort_values('date')
        g = g[g['date'] <= season_end]
        if g.empty:
            results[area] = {'consumed': 0.0, 'first_date': None, 'last_date': None, 'current_value': None}
            continue
        before_start = g[g['date'] <= season_start]
        if not before_start.empty:
            v_start = before_start.iloc[-1]['value']
            date_start = before_start.iloc[-1]['date']
        else:
            v_start = g.iloc[0]['value']
            date_start = g.iloc[0]['date']
        v_end = g.iloc[-1]['value']
        date_end = g.iloc[-1]['date']
        consumed = max(0.0, v_end - v_start)
        results[area] = {'consumed': consumed, 'first_date': date_start, 'last_date': date_end, 'current_value': v_end}
    return results


def forecast_linear(results, season_end, today=None):
    if today is None:
        today = date.today()
    days_remaining = (season_end - today).days
    if days_remaining < 0:
        days_remaining = 0
    forecast = {}
    total_now = 0.0
    total_forecast = 0.0
    for area, r in results.items():
        consumed = r['consumed']
        first = r['first_date']
        last = r['last_date']
        if first is None or last is None:
            avg_daily = 0.0
        else:
            days_used = (last - first).days
            if days_used <= 0:
                days_used = 1
            avg_daily = consumed / days_used
        forecast_end = consumed + avg_daily * days_remaining
        forecast[area] = {'now': consumed, 'avg_daily': avg_daily, 'forecast_end': forecast_end}
        total_now += consumed
        total_forecast += forecast_end
    return forecast, total_now, total_forecast

# ----- App -----
init_session_state()

st.title('Śledzenie i prognoza zużycia ciepła')

# Sidebar
with st.sidebar:
    st.header('Ustawienia sezonu')
    season_start = st.date_input('Początek sezonu grzewczego', value=date(date.today().year, 10, 1))
    season_end = st.date_input('Koniec sezonu grzewczego', value=date(date.today().year + 1, 4, 30))

    prev_season_total = st.number_input('Wartość wyjściowa (całkowita) z poprzedniego okresu', value=st.session_state.prev_season_total, step=0.1)
    if prev_season_total != st.session_state.prev_season_total:
        st.session_state.prev_season_total = prev_season_total
        save_config({'prev_season_total': prev_season_total})

    st.markdown('---')
    st.header('Obszary (pomieszczenia)')
    new_area = st.text_input('Dodaj nowy obszar', '')
    if st.button('Dodaj obszar'):
        if new_area.strip() == '':
            st.error('Nazwa obszaru nie może być pusta')
        else:
            add_area(new_area)

    remove_choice = st.selectbox('Usuń obszar', ['(Wybierz)'] + st.session_state.areas)
    if st.button('Usuń wybrany obszar'):
        if remove_choice != '(Wybierz)':
            remove_area(remove_choice)

    st.write('Aktywne obszary:')
    for a in st.session_state.areas:
        st.write('- ' + a)

st.markdown('---')

# Main layout
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader('Dodaj odczyt (ręcznie)')
    area_choice = st.selectbox('Wybierz obszar', st.session_state.areas)
    rd_date = st.date_input('Data odczytu', value=date.today())
    rd_value = st.text_input('Wartość odczytu (licznik) — liczba', '')
    if st.button('Dodaj odczyt'):
        add_reading(area_choice, rd_date, rd_value)

    st.markdown('---')
    st.subheader('Szybkie akcje')
    if st.button('Wyczyść wszystkie odczyty'):
        clear_all_data()

with col2:
    st.subheader('Tabela odczytów')
    df = to_dataframe(st.session_state.readings)
    st.dataframe(df)
    if not df.empty:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button('Pobierz odczyty (CSV)', data=csv, file_name='readings.csv', mime='text/csv')

st.markdown('---')

# Calculations and visualization
st.header('Analiza i prognoza')
if df.empty:
    st.info('Brak odczytów — dodaj odczyty dla wybranych obszarów.')
else:
    results = compute_consumption(df, season_start, season_end)
    forecast, total_now, total_forecast = forecast_linear(results, season_end)

    st.subheader('Podsumowanie na obszar')
    summary_rows = []
    for area in st.session_state.areas:
        r = results.get(area, {'consumed': 0.0, 'first_date': None, 'last_date': None})
        f = forecast.get(area, {'now': 0.0, 'avg_daily': 0.0, 'forecast_end': 0.0})
        summary_rows.append({
            'area': area,
            'consumed_so_far': round(f['now'], 3),
            'avg_daily': round(f['avg_daily'], 6),
            'forecast_end_of_season': round(f['forecast_end'], 3)
        })
    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df)

    st.subheader('Suma wszystkich obszarów')
    st.metric('Zużycie do teraz (suma)', f"{round(total_now,3)}")
    st.metric('Prognoza na koniec sezonu (suma)', f"{round(total_forecast,3)}")
    st.write('Wartość wyjściowa z poprzedniego okresu (podana):', st.session_state.prev_season_total)

    st.subheader('Wykres zużycia (skumulowane)')
    plt.figure(figsize=(10,5))
    ax = plt.gca()
    for area, g in df.groupby('area'):
        g = g.sort_values('date')
        before = g[g['date'] <= season_start]
        if not before.empty:
            baseline = before.iloc[-1]['value']
        else:
            baseline = g.iloc[0]['value']
        dates = pd.to_datetime(g['date'])
        values = g['value'] - baseline
        if len(dates) > 0:
            ax.plot(dates, values, label=area)

    # Add comparison line for previous season total
    ax.plot([pd.to_datetime(season_start), pd.to_datetime(season_end)], [0, st.session_state.prev_season_total],
            label='Poprzedni okres', linestyle='--', color='red')

    ax.set_xlabel('Data')
    ax.set_ylabel('Skumulowane zużycie (jednostki)')
    ax.legend()
    ax.grid(True)
    st.pyplot(plt)

st.markdown('\n---\n')
st.caption('Dane zapisywane są automatycznie do plików: odczyty → data.csv, konfiguracja → config.json. Prosta prognoza liniowa: prognoza = zużycie + średnie dzienne * dni do końca sezonu.')
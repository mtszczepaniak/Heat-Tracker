# streamlit_heat_tracker.py
# A Streamlit app to track and forecast heating consumption for multiple areas (rooms)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, date

st.set_page_config(page_title="Heat Consumption Tracker", layout="wide")

# ----- Helpers -----

def init_session_state():
    if 'areas' not in st.session_state:
        # default three rooms
        st.session_state.areas = ['Room 1', 'Room 2', 'Room 3']
    if 'readings' not in st.session_state:
        # store readings as list of dicts: {area, date (iso), value}
        st.session_state.readings = []


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


def add_reading(area, reading_date, value):
    try:
        val = float(value)
    except Exception:
        st.error('Wartość odczytu musi być liczbą.')
        return
    st.session_state.readings.append({'area': area, 'date': reading_date.isoformat(), 'value': val})
    st.success(f'Dodano odczyt dla {area} — {reading_date} : {val}')


# compute consumptions across season

def compute_consumption(df, season_start, season_end):
    # df: area, date, value (date as datetime.date)
    results = {}
    # filter readings within some window (we will consider readings before season end for proper diffs)
    for area, g in df.groupby('area'):
        g = g.sort_values('date')
        # Only keep readings up to season_end
        g = g[g['date'] <= season_end]
        if g.empty:
            results[area] = {'consumed': 0.0, 'first_date': None, 'last_date': None, 'current_value': None}
            continue
        # We'll compute consumption as difference between earliest reading on/after season_start (or the first reading before) and latest reading up to season_end
        # Find reading at or before season_start
        before_start = g[g['date'] <= season_start]
        if not before_start.empty:
            v_start = before_start.iloc[-1]['value']
            date_start = before_start.iloc[-1]['date']
        else:
            # use earliest reading available (may be after season_start)
            v_start = g.iloc[0]['value']
            date_start = g.iloc[0]['date']
        v_end = g.iloc[-1]['value']
        date_end = g.iloc[-1]['date']
        consumed = max(0.0, v_end - v_start)
        results[area] = {'consumed': consumed, 'first_date': date_start, 'last_date': date_end, 'current_value': v_end}
    return results


def forecast_linear(results, season_end, today=None):
    # results: dict per area with consumed, first_date, last_date
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

# Sidebar: season settings and area management
with st.sidebar:
    st.header('Ustawienia sezonu')
    season_start = st.date_input('Początek sezonu grzewczego', value=date(date.today().year, 10, 1))
    season_end = st.date_input('Koniec sezonu grzewczego', value=date(date.today().year + 1, 4, 30))
    prev_season_total = st.number_input('Wartość wyjściowa (całkowita) z poprzedniego okresu', value=0.0, step=0.1)
    st.markdown('---')
    st.header('Obszary (pomieszczenia)')
    new_area = st.text_input('Dodaj nowy obszar', '')
    if st.button('Dodaj obszar'):
        if new_area.strip() == '':
            st.error('Nazwa obszaru nie może być pusta')
        else:
            add_area(new_area)
    st.write('Aktywne obszary:')
    for a in st.session_state.areas:
        st.write('- ' + a)

st.markdown('---')

# Main layout: left column for data entry per area, right column for tables and plots
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
        st.session_state.readings = []
        st.success('Wszystkie odczyty usunięte')

with col2:
    st.subheader('Tabela odczytów')
    df = to_dataframe(st.session_state.readings)
    st.dataframe(df)

    # allow export
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

    # Show per-area summary
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

    # Total summary
    st.subheader('Suma wszystkich obszarów')
    st.metric('Zużycie do teraz (suma)', f"{round(total_now,3)}")
    st.metric('Prognoza na koniec sezonu (suma)', f"{round(total_forecast,3)}")
    st.write('Wartość wyjściowa z poprzedniego okresu (podana):', prev_season_total)

    # Plot: cumulative consumption over time per area and total
    st.subheader('Wykres zużycia (skumulowane)')
    # prepare cumulative consumption per area over time
    plt.figure(figsize=(10,5))
    ax = plt.gca()
    all_dates = pd.date_range(start=season_start, end=min(season_end, date.today()))

    # plot per area
    for area, g in df.groupby('area'):
        # create series of cumulative consumption starting from season start baseline
        g = g.sort_values('date')
        # determine baseline value at or before season_start
        baseline = None
        before = g[g['date'] <= season_start]
        if not before.empty:
            baseline = before.iloc[-1]['value']
        else:
            baseline = g.iloc[0]['value']
        # build series for plotting
        dates = pd.to_datetime(g['date'])
        values = g['value'] - baseline
        # only plot dates up to today
        dates_plot = dates[dates <= pd.to_datetime(date.today())]
        values_plot = values[:len(dates_plot)]
        if len(dates_plot) > 0:
            ax.plot(dates_plot, values_plot, label=area)

    # total line: sum of area consumptions at each date (approx by reindexing)
    # build a daily total series
    daily_index = pd.date_range(start=season_start, end=min(season_end, date.today()))
    total_series = pd.Series(0.0, index=daily_index)
    for area, g in df.groupby('area'):
        g = g.sort_values('date')
        # baseline
        before = g[g['date'] <= season_start]
        if not before.empty:
            baseline = before.iloc[-1]['value']
        else:
            baseline = g.iloc[0]['value']
        s = pd.Series(g['value'].values - baseline, index=pd.to_datetime(g['date']))
        s = s.reindex(daily_index, method='ffill').fillna(0.0)
        total_series = total_series + s
    ax.plot(total_series.index, total_series.values, label='Total', linewidth=2, linestyle='--')

    ax.set_xlabel('Data')
    ax.set_ylabel('Skumulowane zużycie (jednostki)')
    ax.legend()
    ax.grid(True)
    st.pyplot(plt)

    # Show forecast details
    st.subheader('Szczegóły prognozy')
    fc_rows = []
    for area, f in forecast.items():
        fc_rows.append({'area': area, 'now': round(f['now'],3), 'avg_daily': round(f['avg_daily'],6), 'forecast_end': round(f['forecast_end'],3)})
    fc_df = pd.DataFrame(fc_rows)
    st.dataframe(fc_df)

st.markdown('\n---\n')
st.caption('Prosta prognoza liniowa: prognoza = dotychczasowe zużycie + średnie dzienne * dni do końca sezonu')

# EOF

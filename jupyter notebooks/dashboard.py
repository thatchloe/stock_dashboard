import pandas as pd
from glob import glob
from time import strftime, sleep
import numpy as np
from datetime import datetime

import dash
from dash import dcc
from dash import html
from dash.dependencies import Output, Input
import plotly.express as px
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import dash_table
from jupyter_dash import JupyterDash
from plotly.subplots import make_subplots
from dash import Dash

def clean_header(df):
    df.columns = df.columns.str.strip().str.lower().str.replace('.', '', regex=False).str.replace('(', \
                '', regex=False).str.replace(')', '', regex=False).str.replace(' ', '_', regex=False).str.replace('_/_', '/', regex=False)
    
def get_now():
    now = datetime.now().strftime('%Y-%m-%d_%Hh%Mm')
    return now

def datetime_maker(df, datecol):
    df[datecol] = pd.to_datetime(df[datecol])

last_file = sorted(glob('../outputs/transactions_all/*finaldf*.xlsx'))[-1] # path to file in the folder

all_transactions = pd.read_excel(last_file, engine='openpyxl')
all_transactions.date = pd.to_datetime(all_transactions.date, format='%d/%m/%Y')

last_file = sorted(glob('../outputs/portfolio_df/portfolio_df_*.csv'))[-1] # path to file in the folder

portf_allvalues = pd.read_csv(last_file)
portf_allvalues = portf_allvalues.set_index('date')

last_file = sorted(glob('../outputs/final_current_positions/final_current_positions_*.csv'))[-1] # path to file in the folder

current_positions = pd.read_csv(last_file)
current_positions = current_positions.sort_values(by='current_value', ascending=False).round(2)

initial_date = '2020-01-09'
plotlydf_portfval = portf_allvalues[portf_allvalues.index > initial_date]

# plotlydf_portfval = portf_allvalues.copy()
plotlydf_portfval = plotlydf_portfval[['portf_value', 'sp500_mktvalue', 'ptf_value_pctch',
                                     'sp500_pctch', 'ptf_value_diff', 'sp500_diff']].reset_index().round(2)

# calculating cumulative growth since initial date
plotlydf_portfval.rename(columns={'index': 'date'}, inplace=True)  # needed for later
plotlydf_portfval.date = pd.to_datetime(plotlydf_portfval.date)
# Going to use the column cashflow to calculate a net return on the assets
invested_df = (all_transactions.groupby('date').sum()['cashflow']*-1)
idx = pd.date_range(all_transactions.date.min(), plotlydf_portfval.date.max())
invested_df = invested_df.reindex(idx, fill_value=0).reset_index()
invested_df.rename(columns={'index': 'date'}, inplace=True)
invested_df['alltime_cashflow'] = invested_df['cashflow'].cumsum()
plotlydf_portfval = pd.merge(plotlydf_portfval, invested_df, on='date', how='inner')
# net invested will let us know how much we invested during the period in analysis
# then we take this out of the portfolio value, to calculate the returns
plotlydf_portfval['net_invested'] = plotlydf_portfval['cashflow'].cumsum()
plotlydf_portfval['net_value'] = plotlydf_portfval.portf_value - plotlydf_portfval.net_invested
plotlydf_portfval['ptf_growth'] = plotlydf_portfval.net_value/plotlydf_portfval['net_value'].iloc[0]
plotlydf_portfval['sp500_growth'] = plotlydf_portfval.sp500_mktvalue/plotlydf_portfval['sp500_mktvalue'].iloc[0]
# adjusted ptfchg will be the accurate variation (net of investments)
plotlydf_portfval['adjusted_ptfchg'] = (plotlydf_portfval['net_value'].pct_change()*100).round(2)
plotlydf_portfval['highvalue'] = plotlydf_portfval['net_value'].cummax()
plotlydf_portfval['drawdownpct'] = (plotlydf_portfval['net_value']/plotlydf_portfval['highvalue']-1).round(4)*100

CHART_THEME = 'plotly_dark'  # others include seaborn, ggplot2, plotly_white, plotly_dark
chart_ptfvalue = go.Figure() # generating a figure that will be updated in the following lines

chart_ptfvalue.add_trace(
    go.Scatter(
        x=plotlydf_portfval.date,
        y=plotlydf_portfval.portf_value,
        mode='lines',  # you can also use "lines+markers", or just "markers"
        name='Portfolio Value',
        hovertemplate = '$ %{y:,.0f}'
    )
)

chart_ptfvalue.add_trace(
    go.Scatter(
        x=invested_df.date,
        y=invested_df.alltime_cashflow,
        fill='tozeroy',
        fillcolor='rgba(255, 150, 20, 0.3)', # https://www.w3schools.com/css/css_colors_rgb.asp
        line = dict(
            color='orangered',
            width=2,
            dash='dash'),
        mode='lines',  # you can also use "lines+markers", or just "markers"
        name='Net Invested',
        hovertemplate = '$ %{y:,.0f}'
    )
)


chart_ptfvalue.update_layout(
    margin = dict(t=50, b=50, l=25, r=25), # this will help you optimize the chart space
    xaxis_tickfont_size=12,
    yaxis=dict(
        title='Value: $ USD',
        titlefont_size=14,
        tickfont_size=12,
        ),
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01),
    showlegend=False,
#     title='Global Portfolio Value (USD $)',
    title_x=0.5, # title centered
    paper_bgcolor="#272b30",
    plot_bgcolor="#272b30"
)


# # # Time Series with Range Selector Buttons - https://plotly.com/python/time-series/
chart_ptfvalue.update_xaxes(
    rangeslider_visible=False,
        rangeselector=dict(
            buttons=list([
                dict(count=7, label="1w", step="day", stepmode="backward"),
                dict(count=14, label="2w", step="day", stepmode="backward"),
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=12, label="12m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(label='All', step="all"),
            ]),
            bgcolor="#272b30",
            activecolor='tomato',
#             y=1.22,
#             x=0.25
        )
)

chart_ptfvalue.update_layout(hovermode='x unified')
chart_ptfvalue.layout.template = CHART_THEME
chart_ptfvalue.layout.height=500

drawdown_chart = go.Figure()  # generating a figure that will be updated in the following lines
drawdown_chart.add_trace(
    go.Scatter(
        x=plotlydf_portfval.date,
        y=plotlydf_portfval.drawdownpct,
        fill='tozeroy',
        fillcolor='tomato',
        line = dict(
            color='firebrick',
            width=2),
        mode='lines',  # you can also use "lines+markers", or just "markers"
        name='Drawdown %'))

drawdown_chart.update_layout(
    margin = dict(t=45, b=30, l=25, r=25),
    yaxis=dict(
        title='%',
        titlefont_size=14,
        tickfont_size=12,
        ),
    title='Drawdown',
    title_x=0.5,
    paper_bgcolor="#272b30",
    plot_bgcolor="#272b30"
)

drawdown_chart.update_xaxes(
    rangeslider_visible=False,
        rangeselector=dict(
            buttons=list([
                dict(count=7, label="1w", step="day", stepmode="backward"),
                dict(count=14, label="2w", step="day", stepmode="backward"),
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=12, label="12m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(label='All', step="all"),
            ]),
            bgcolor="#272b30",
            activecolor='tomato',
#             y=1.02,
#             x=0.05
        )
)

drawdown_chart.layout.template = CHART_THEME
drawdown_chart.layout.height=250

portfolio_cashflow = go.Figure()  # generating a figure that will be updated in the following lines
portfolio_cashflow.add_trace(
    go.Bar(
        x=plotlydf_portfval.date,
        y=plotlydf_portfval.cashflow.replace(0,np.nan),
        name='Drawdown %',
        xperiod="M1",
    )
)

portfolio_cashflow.update_layout(
    margin = dict(t=50, b=30, l=25, r=25),
    yaxis=dict(
        title='$ Value',
        titlefont_size=14,
        tickfont_size=12,
        ),
    title='Monthly Buy & Sell Orders',
    title_x=0.5,
    paper_bgcolor="#272b30",
    plot_bgcolor="#272b30"
)

portfolio_cashflow.update_xaxes(
    rangeslider_visible=False,
        rangeselector=dict(
            buttons=list([
                dict(count=7, label="1w", step="day", stepmode="backward"),
                dict(count=14, label="2w", step="day", stepmode="backward"),
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=12, label="12m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(label='All', step="all"),
            ]),
            bgcolor="#272b30",
            activecolor='tomato',
        )
)

portfolio_cashflow.layout.template = CHART_THEME
portfolio_cashflow.layout.height=250
df = plotlydf_portfval[['date', 'net_value', 'sp500_mktvalue']].copy()
df['month'] = df.date.dt.month_name()
df['weekday'] = df.date.dt.day_name()
df['year'] = df.date.dt.year
df['weeknumber'] = df.date.dt.isocalendar().week
df['timeperiod'] = df.year.astype(str) + ' - ' + df.date.dt.month.astype(str).str.zfill(2)
sp = df.reset_index().groupby('timeperiod').last()['sp500_mktvalue'].pct_change()*100
ptf = df.reset_index().groupby('timeperiod').last()['net_value'].pct_change()*100
plotlydf_growth_compare = pd.merge(ptf, sp, on='timeperiod').reset_index()
plotlydf_growth_compare.head()
fig_growth2 = go.Figure()
fig_growth2.layout.template = CHART_THEME
fig_growth2.add_trace(go.Bar(
    x=plotlydf_growth_compare.timeperiod,
    y=plotlydf_growth_compare.net_value.round(2),
    name='Portfolio'
))
fig_growth2.add_trace(go.Bar(
    x=plotlydf_growth_compare.timeperiod,
    y=plotlydf_growth_compare.sp500_mktvalue.round(2),
    name='S&P 500',
))
fig_growth2.update_layout(barmode='group')
fig_growth2.layout.height=300
fig_growth2.update_layout(margin = dict(t=50, b=50, l=25, r=25))
fig_growth2.update_layout(
    xaxis_tickfont_size=12,
    yaxis=dict(
        title='% change',
        titlefont_size=13,
        tickfont_size=12,
        ))

fig_growth2.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="right",
    x=0.99))
fig_growth2.update_layout(paper_bgcolor="#272b30", plot_bgcolor="#272b30")
t = 'AMZN'
t_candles = pd.read_csv('../outputs/price_hist/{}_price_hist.csv'.format(t))
# Create subplots and mention plot grid size
fig_main = make_subplots(rows=2, cols=1, shared_xaxes=True, 
               vertical_spacing=0.03, subplot_titles=('OHLC', 'Volume'), 
               row_width=[0.2, 0.7])
fig_main.layout.template = CHART_THEME
# Plot OHLC on 1st row
fig_main.add_trace(go.Candlestick(x=t_candles["date"], open=t_candles["open"], high=t_candles["high"],
                low=t_candles['low'], close=t_candles['close'], name="OHLC", showlegend=False), 
                row=1, col=1
)

tx_df = all_transactions[all_transactions.ticker==t]

fig_main.add_trace(go.Scatter(
    x=tx_df[tx_df.type=='Buy'].date,
    y=tx_df[tx_df.type=='Buy'].price,
    mode='markers',
    name='Buy Orders',
    marker=dict(
        color='rgba(63, 255, 56, 0.6)',
        size=12,
        line=dict(
            color='black',
            width=1
        )), showlegend=False))

fig_main.add_trace(go.Scatter(
    x=tx_df[tx_df.type=='Sell'].date,
    y=tx_df[tx_df.type=='Sell'].price,
    mode='markers',
    name='Sell Orders',
    marker=dict(
            color='rgba(255, 13, 17, 0.6)',
            size=12,
            line=dict(
                color='black',
                width=1
            )), showlegend=False))

avg_price_df = current_positions.set_index('ticker')
avg_price = avg_price_df.loc[t].avg_price
res = round((avg_price_df.loc[t].price/avg_price-1)*100, 2)

fig_main.update_layout(
    yaxis_title='Price $',
    shapes = [dict(
        x0=0, x1=1, y0=avg_price, y1=avg_price, xref='paper', yref='y',
        line_width=1)],
    annotations=[dict(
        x=0.05, y=avg_price*0.90, xref='paper', yref='y',
        showarrow=False, xanchor='left', bgcolor="black",
        opacity=0.30, text='Average Price: $ {}<br>Result: {} %'.format(avg_price, res), font={'size':12})]
)

# Bar trace for volumes on 2nd row without legend
fig_main.add_trace(go.Bar(x=t_candles['date'], y=t_candles['volume'], showlegend=False), row=2, col=1)

# Do not show OHLC's rangeslider plot 
fig_main.update(layout_xaxis_rangeslider_visible=False)
fig_main.update_layout(margin = dict(t=50, b=50, l=25, r=25))
fig_main.update_layout(paper_bgcolor="#272b30", plot_bgcolor="#272b30")
tables_df = current_positions.copy()
tables_df.columns = ['Ticker', 'Company', 'Sector', 'Industry', 'P/E', 'Perf Week', 'Perf Month', 'Perf Quart',
             'Perf Half', 'Perf Year', 'Perf YTD', 'Volatility Week', 'Volatility Month', 'Recom', 'ATR',
             'SMA20', 'SMA50', 'SMA200', '52W High', '52W Low', 'RSI', 'Insider Own', 'Insider Trans',
             'Inst Own', 'Inst Trans', 'Float Short', 'Short Ratio', 'Dividend', 'LTDebt/Eq', 'Debt/Eq', 
             'Cumulative Units', 'Cumulative Cost ($)', 'Realized G/L ($)', 'Open Cashflow ($)',
             'Price ($)', 'Current Value ($)', 'Average Cost', 'Weight (%)', 'Unrealized ($)', 'Unrealized (%)']
table_dict = {}
for tick in tables_df.Ticker:
    table = tables_df[tables_df.Ticker==tick].T.reset_index()
    table.columns = ['indicator', tick]
    table_dict[tick] = table
table_dict['AAPL'].iloc[[34,35,37,31,36,39,38,33,32],]
datatabletotal = tables_df.to_dict('records')
cols_total = [{"name": i, "id": i} for i in tables_df.columns[:10]]

tableview_table = dash_table.DataTable(
    id='total-table',
    columns=cols_total,
    data=datatabletotal,
    filter_action="native",
    sort_action="native",
    fixed_columns={
        'headers': True,
        'data': 2
    },
    style_table={
        'minWidth': '100%',
        'overflowX': 'auto'
    },
    style_header={
        'backgroundColor': 'rgb(30, 30, 30)',
        'color': 'white'
    },
    style_data={
        'backgroundColor': 'rgb(50, 50, 50)',
        'color': 'white'
    },
    style_data_conditional=[
        {
            'if': {'row_index': 'odd'},
            'backgroundColor': 'rgb(80, 80, 80)',
        },
        {
           'if': {'column_id': 'Ticker'},
           'width': '25px',
           'textAlign': 'left',
           'fontWeight' : 'bold'
       },
       {
           'if': {'column_id': 'Company'},
           'width': '140px',
           'textAlign': 'left',
       }
    ],
)


kpi_portfolio7d_abs = portf_allvalues.tail(7).ptf_value_diff.sum().round(2)
kpi_portfolio15d_abs = portf_allvalues.tail(15).ptf_value_diff.sum().round(2)
kpi_portfolio30d_abs = portf_allvalues.tail(30).ptf_value_diff.sum().round(2)
kpi_portfolio200d_abs = portf_allvalues.tail(200).ptf_value_diff.sum().round(2)
kpi_portfolio7d_pct = round(kpi_portfolio7d_abs/portf_allvalues.tail(7).portf_value[0]*100,2)
kpi_portfolio15d_pct = round(kpi_portfolio15d_abs/portf_allvalues.tail(15).portf_value[0]*100,2)
kpi_portfolio30d_pct = round(kpi_portfolio30d_abs/portf_allvalues.tail(30).portf_value[0]*100,2)
kpi_portfolio200d_pct = round(kpi_portfolio200d_abs/portf_allvalues.tail(200).portf_value[0]*100,2)

kpi_sp500_7d_abs = portf_allvalues.tail(7).sp500_diff.sum().round(2)
kpi_sp500_15d_abs = portf_allvalues.tail(15).sp500_diff.sum().round(2)
kpi_sp500_30d_abs = portf_allvalues.tail(30).sp500_diff.sum().round(2)
kpi_sp500_200d_abs = portf_allvalues.tail(200).sp500_diff.sum().round(2)
kpi_sp500_7d_pct = round(kpi_sp500_7d_abs/portf_allvalues.tail(7).sp500_mktvalue.iloc[0]*100,2)
kpi_sp500_15d_pct = round(kpi_sp500_15d_abs/portf_allvalues.tail(15).sp500_mktvalue.iloc[0]*100,2)
kpi_sp500_30d_pct = round(kpi_sp500_30d_abs/portf_allvalues.tail(30).sp500_mktvalue.iloc[0]*100,2)
kpi_sp500_200d_pct = round(kpi_sp500_200d_abs/portf_allvalues.tail(200).sp500_mktvalue.iloc[0]*100,2)

indicators_ptf = go.Figure()
indicators_ptf.layout.template = CHART_THEME
indicators_ptf.add_trace(go.Indicator(
    mode = "number+delta",
    value = kpi_portfolio7d_pct,
    number = {'suffix': " %"},
    title = {"text": "<br><span style='font-size:0.7em;color:gray'>7 Days</span>"},
    delta = {'position': "bottom", 'reference': kpi_sp500_7d_pct, 'relative': False},
    domain = {'row': 0, 'column': 0}))

indicators_ptf.add_trace(go.Indicator(
    mode = "number+delta",
    value = kpi_portfolio15d_pct,
    number = {'suffix': " %"},
    title = {"text": "<span style='font-size:0.7em;color:gray'>15 Days</span>"},
    delta = {'position': "bottom", 'reference': kpi_sp500_15d_pct, 'relative': False},
    domain = {'row': 1, 'column': 0}))

indicators_ptf.add_trace(go.Indicator(
    mode = "number+delta",
    value = kpi_portfolio30d_pct,
    number = {'suffix': " %"},
    title = {"text": "<span style='font-size:0.7em;color:gray'>30 Days</span>"},
    delta = {'position': "bottom", 'reference': kpi_sp500_30d_pct, 'relative': False},
    domain = {'row': 2, 'column': 0}))

indicators_ptf.add_trace(go.Indicator(
    mode = "number+delta",
    value = kpi_portfolio200d_pct,
    number = {'suffix': " %"},
    title = {"text": "<span style='font-size:0.7em;color:gray'>200 Days</span>"},
    delta = {'position': "bottom", 'reference': kpi_sp500_200d_pct, 'relative': False},
    domain = {'row': 3, 'column': 1}))

indicators_ptf.update_layout(
    grid = {'rows': 4, 'columns': 1, 'pattern': "independent"},
    margin=dict(l=50, r=50, t=30, b=30)
)
indicators_ptf.update_layout(paper_bgcolor="#272b30")

indicators_sp500 = go.Figure()
indicators_sp500.layout.template = CHART_THEME
indicators_sp500.add_trace(go.Indicator(
    mode = "number+delta",
    value = kpi_sp500_7d_pct,
    number = {'suffix': " %"},
    title = {"text": "<br><span style='font-size:0.7em;color:gray'>7 Days</span>"},
    domain = {'row': 0, 'column': 0}))

indicators_sp500.add_trace(go.Indicator(
    mode = "number+delta",
    value = kpi_sp500_15d_pct,
    number = {'suffix': " %"},
    title = {"text": "<span style='font-size:0.7em;color:gray'>15 Days</span>"},
    domain = {'row': 1, 'column': 0}))

indicators_sp500.add_trace(go.Indicator(
    mode = "number+delta",
    value = kpi_sp500_30d_pct,
    number = {'suffix': " %"},
    title = {"text": "<span style='font-size:0.7em;color:gray'>30 Days</span>"},
    domain = {'row': 2, 'column': 0}))

indicators_sp500.add_trace(go.Indicator(
    mode = "number+delta",
    value = kpi_sp500_200d_pct,
    number = {'suffix': " %"},
    title = {"text": "<span style='font-size:0.7em;color:gray'>200 Days</span>"},
    domain = {'row': 3, 'column': 1}))

indicators_sp500.update_layout(
    grid = {'rows': 4, 'columns': 1, 'pattern': "independent"},
    margin=dict(l=50, r=50, t=30, b=30)
)

indicators_sp500.update_layout(paper_bgcolor="#272b30")
levels = ['ticker', 'industry', 'sector'] # levels used for the hierarchical chart
color_columns = ['current_value', 'cml_cost']
value_column = 'current_value'
current_ptfvalue = "${:,.2f}".format(portf_allvalues.portf_value[-1]) # To use later with Dash
def build_hierarchical_dataframe(df, levels, value_column, color_columns=None, total_name='total'):
    """
    Build a hierarchy of levels for Sunburst or Treemap charts.

    Levels are given starting from the bottom to the top of the hierarchy,
    ie the last level corresponds to the root.
    """
    df_all_trees = pd.DataFrame(columns=['id', 'parent', 'value', 'color'])
    for i, level in enumerate(levels):
        df_tree = pd.DataFrame(columns=['id', 'parent', 'value', 'color'])
        dfg = df.groupby(levels[i:]).sum()
        dfg = dfg.reset_index()
        df_tree['id'] = dfg[level].copy()
        if i < len(levels) - 1:
            df_tree['parent'] = dfg[levels[i+1]].copy()
        else:
            df_tree['parent'] = total_name
        df_tree['value'] = dfg[value_column]
        df_tree['color'] = round((dfg[color_columns[0]] / dfg[color_columns[1]]-1)*100, 2)
        df_all_trees = pd.concat([df_all_trees, pd.DataFrame(df_tree)], ignore_index=True)
    total = pd.Series(dict(id=total_name, parent='',
                              value=df[value_column].sum(),
                              color=100*round(df[color_columns[0]].sum() / df[color_columns[1]].sum()-1,2)))
    df_all_trees = pd.concat([df_all_trees, pd.DataFrame(total)], ignore_index=True)
    return df_all_trees
df_all_trees = build_hierarchical_dataframe(current_positions, levels, value_column, color_columns, total_name='Portfolio')
average_score = current_positions['current_value'].sum() / current_positions['cml_cost'].sum()-1
sunburst_fig2 = go.Figure()
sunburst_fig2.layout.template = CHART_THEME
sunburst_fig2.add_trace(go.Sunburst(
    labels=df_all_trees['id'],
    parents=df_all_trees['parent'],
    values=df_all_trees['value'],
    branchvalues='total',
    marker=dict(
        colors=df_all_trees['color'],
        colorscale='mrybm',
        cmid=average_score),
    hovertemplate='<b>%{label} </b> <br> Size: $ %{value}<br> Variation: %{color:.2f}%',
    maxdepth=2,
    name=''
    ))

sunburst_fig2.update_layout(margin=dict(t=10, b=10, r=10, l=10))
sunburst_fig2.update_layout(paper_bgcolor="#272b30")

ticker_dict = [{'label': current_positions.company[i], 'value': current_positions.ticker[i]} for i in range(current_positions.shape[0])]
first_stock = current_positions.ticker[0]

SIDEBAR_STYLE = {
    'position': 'fixed',
    'top': 0,
    'left': 0,
    'bottom': 0,
    'width': '12rem',
    'padding': '2rem 1rem',
    'background-color': 'rgba(120, 120, 120, 0.4)',
}
CONTENT_STYLE = {
    'margin-left': '15rem',
    'margin-right': '2rem',
    'padding': '2rem' '1rem',
}

sidebar = html.Div(
    [
        html.Hr(),
        html.P('Investment Tracker v0.95', className='text-center p-3 border border-dark'),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink('Portfolio', href="/", active='exact'),
                dbc.NavLink('Ticker View', href="/tickerpage", active='exact'),
                dbc.NavLink('Sunburst Chart', href="/sunburst", active='exact'),
                dbc.NavLink('Table View', href="/tableview", active='exact')
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)
homepage = [
    dbc.Row(dbc.Col(html.H2('PORTFOLIO OVERVIEW', className='text-center mb-3 p-3'))),
    dbc.Row([
        dbc.Col([
            html.H5('Portfolio Value vs Net Invested ($USD)', className='text-center'),
            html.Div(children=f"Portfolio Value: {current_ptfvalue}", className='text-left mb-2'),
            dcc.Graph(id='chrt-portfolio-main',
                      figure=chart_ptfvalue,
                      style={'height': 450},
                      className='shadow-lg'
                     ),
            html.Hr(),

        ],
            width={'size': 8, 'offset': 0, 'order': 1}),
        dbc.Col([
            html.H5('Portfolio', className='text-center'),
            html.Div(children="KPI's", className='text-center fs-4'),
            dcc.Graph(id='indicators-ptf',
                      figure=indicators_ptf,
                      style={'height': 450},
                      className='shadow-lg'),
            html.Hr()
        ],
            width={'size': 2, 'offset': 0, 'order': 2}),
        dbc.Col([
            html.H5('S&P500', className='text-center'),
            html.Div(children="KPI's", className='text-center fs-4'),            
            dcc.Graph(id='indicators-sp',
                      figure=indicators_sp500,
                      style={'height': 450},
                      className='shadow-lg'),
            html.Hr()
        ],
            width={'size': 2, 'offset': 0, 'order': 3}),
    ]),  # end of second row
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='chrt-portfolio-secondary',
                      figure=drawdown_chart,
                      style={'height': 300},
                      className='shadow-lg'),
            html.Hr(),
            dcc.Graph(id='chrt-portfolio-third',
                      figure=portfolio_cashflow,
                      style={'height': 300},
                      className='shadow-lg'),
        ],
            width={'size': 8, 'offset': 0, 'order': 1}),
        dbc.Col([
            dcc.Graph(id='pie-top15',
                      figure=sunburst_fig2,
                      style={'height': 630},
                      className='shadow-lg'),
        ],
            width={'size': 4, 'offset': 0, 'order': 2}),
    ])

]
tickerpage = [
    dbc.Row(dbc.Col(html.H2('TICKER VIEW', className='text-center mb-3 p-3'))),
    dbc.Row([
        dbc.Col([
            html.H5('Candlestick chart', className='text-center'),
            dcc.Dropdown(
                id='ticker-selector',
                options=ticker_dict,
                value=first_stock,
                clearable=False,
            ),
            dcc.Graph(id='chrt-ticker-main',
                      figure=fig_main,
                      style={'height': 920},
                      className='shadow-lg'),
            html.Hr(),

        ],
            width={'size': 9, 'offset': 0, 'order': 1}),
        dbc.Col([
#             html.H5('Metrics', className='text-center'),
            dash_table.DataTable(id='first-table',
                                 columns=[],
                                 data=[],
                                 style_header = {'display': 'none'},
                                 style_data = {'whiteSpace': 'normal',
                                               'height': 'auto',
                                               'lineHeight': '15px',
                                               'border': 'none',
                                               'backgroundColor': '#272b30',
                                               'color': 'white'
#                                                'textAlign': 'center'
                                              },
                                style_cell_conditional=[
                                    {'if': {'column_id': 'indicator'},
                                     'width': '40%',
                                     'textAlign': 'left',
                                     'fontWeight' : 'bold'
#                                      'backgroundColor': 'rgba(0, 116, 217, 0.3)',
#                                      'color': 'rgba(0,20,80,1)'
                                    },
                                    ]),
            html.Hr(),
            dash_table.DataTable(id='second-table',
                                 columns=[],
                                 data=[],
                                 style_header = {'display': 'none'},
                                 style_data = {'whiteSpace': 'normal',
                                               'height': 'auto',
                                               'lineHeight': '15px',
                                               'border': 'none',
                                               'backgroundColor': '#272b30',
                                               'color': 'white'
#                                                'textAlign': 'center'
                                              },
                                style_cell_conditional = [
                                     {
                                         'if': {'column_id': 'indicator'},
                                         'width': '40%',
                                         'textAlign': 'left',
                                         'fontWeight' : 'bold'
                                     }]
                                ),
            html.Hr(),
            dash_table.DataTable(id='third-table',
                                 columns=[],
                                 data=[],
                                 style_header = {'display': 'none'},
                                 style_data = {'whiteSpace': 'normal',
                                               'height': 'auto',
                                               'lineHeight': '15px',
                                               'border': 'none',
                                               'backgroundColor': '#272b30',
                                               'color': 'white'
#                                                'textAlign': 'center'
                                              },
                                 style_cell_conditional = [
                                     {
                                         'if': {'column_id': 'indicator'},
                                         'width': '40%',
                                         'textAlign': 'left',
                                         'fontWeight' : 'bold'
                                     }]
                                 ),
            html.Hr(),
        ],
            width={'size': 3, 'offset': 0, 'order': 2}),
    ])
]
sunburstpage = [
    dbc.Row(dbc.Col(html.H2('SUNBURST VIEW', className='text-center mb-3 p-3'))),
    dbc.Row([
        dbc.Col([
            html.H5('Explore your portfolio interactively', className='text-left'),
            html.Div(children=f"Portfolio Value: {current_ptfvalue}", className='text-left'),
            html.Hr(),
            dcc.Graph(id='chrt-sunburstpage',
                      figure=sunburst_fig2,
                      style={'height': 800}),
            html.Hr(),

        ],
            width={'size': 12, 'offset': 0, 'order': 1}),
    ]),
]
tablepage = [
    dbc.Row(dbc.Col(html.H2('FULL TABLE VIEW', className='text-center mb-3 p-3'))),
    dbc.Row([
        dbc.Col([
            html.H5('Detailed view about every stock', className='text-left'),
            html.Hr(),
            tableview_table,
            html.Hr(),

        ],
            width={'size': 12, 'offset': 0, 'order': 1}),
    ]),
]
content = html.Div(id='page-content', children=[], style=CONTENT_STYLE)
content = html.Div(id='page-content', children=[], style=CONTENT_STYLE)
app = Dash(__name__, external_stylesheets=[dbc.themes.SLATE], suppress_callback_exceptions=True)

app.layout = dbc.Container(
    [
        
        dcc.Location(id='url', refresh=False),
        sidebar,
        content
        
    ], fluid=True)



@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return homepage

    elif pathname == "/tickerpage":
        return tickerpage

    elif pathname == "/sunburst":
        return sunburstpage

    elif pathname == "/tableview":
        return tablepage


@app.callback(
    [Output("chrt-ticker-main", "figure"),
     Output("first-table", "columns"), Output("first-table", "data"),
     Output("second-table", "columns"), Output("second-table", "data"),
     Output("third-table", "columns"), Output("third-table", "data")
    ],
    [Input("ticker-selector", "value")])
def render_tickerchart(value):
    t_candles = pd.read_csv('../outputs/price_hist/{}_price_hist.csv'.format(value))
    fig_main = make_subplots(rows=2, cols=1, shared_xaxes=True,
                             vertical_spacing=0.03, subplot_titles=('OHLC', 'Volume'),
                             row_width=[0.2, 0.7])

    # Plot OHLC on 1st row
    fig_main.add_trace(go.Candlestick(x=t_candles["date"], open=t_candles["open"], high=t_candles["high"],
                                      low=t_candles['low'], close=t_candles['close'], name="OHLC", showlegend=False),
                       row=1, col=1
                       )
    avg_price_df = current_positions.set_index('ticker')
    avg_price = avg_price_df.loc[value].avg_price
    res = round((avg_price_df.loc[value].price / avg_price - 1) * 100, 2)
    fig_main.update_layout(
        yaxis_title='Price $',
        shapes=[dict(
            x0=0, x1=1, y0=avg_price, y1=avg_price, xref='paper', yref='y', line_width=1)],
        annotations=[dict(
            x=0.05, y=avg_price * 0.90, xref='paper', yref='y',
            showarrow=False, xanchor='left', bgcolor="black",
            opacity=0.35, text='Average Price: $ {}<br>Result: {} %'.format(avg_price, res), font={'size': 12})]
    )

    # Bar trace for volumes on 2nd row without legend
    fig_main.add_trace(go.Bar(x=t_candles['date'], y=t_candles['volume'], showlegend=False), row=2, col=1)
    tx_df = all_transactions[all_transactions.ticker==value]
    fig_main.add_trace(go.Scatter(
        x=tx_df[tx_df.type=='Buy'].date,
        y=tx_df[tx_df.type=='Buy'].price,
        mode='markers',
        name='Buy Orders',
        marker=dict(
            color='rgba(60, 255, 75, 0.8)',
            size=12,
            line=dict(
                color='white',
                width=1
            )), showlegend=False))
    fig_main.add_trace(go.Scatter(
        x=tx_df[tx_df.type=='Sell'].date,
        y=tx_df[tx_df.type=='Sell'].price,
        mode='markers',
        name='Sell Orders',
        marker=dict(
                color='rgba(255, 20, 40, 0.9)',
                size=12,
                line=dict(
                    color='white',
                    width=1
                )), showlegend=False))

    # Do not show OHLC's rangeslider plot
    fig_main.update(layout_xaxis_rangeslider_visible=False)
    fig_main.update_layout(margin=dict(t=50, b=50, l=25, r=25))
    fig_main.update_layout(paper_bgcolor="#272b30", plot_bgcolor='#272b30')
    fig_main.layout.template = CHART_THEME
    
    datatabletwo = table_dict[value][5:11].to_dict('records')
    datatable = table_dict[value].iloc[[0,1,2,3,34,35,37,31,36,39,38,33,32],].to_dict('records')
    datatablethree = table_dict[value][18:25].to_dict('records')
    cols = [{"name": i, "id": i} for i in table_dict[value].columns]

    
    
    return fig_main, cols, datatable, cols, datatabletwo, cols, datatablethree

if __name__ == "__main__":
    app.run_server(debug=True, port='8058')



################################ PACKAGES ################################
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import scipy.optimize as sop

##########################################################################


######################## INPUT ########################
Tickers = ["ES=F", "BND", "GC=F", "DX=F"]
start_date = "1990-01-01"
end_date = "2025-04-30"
window = 252
shift_days = 2
#######################################################


##################################################### CODE #####################################################
st.set_page_config(page_title="Portfolio", layout="wide")
st.title("Portfolio Metrics and Allocation")

with st.form(key="params_form"):
    st.subheader("Input for the analysis:")

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", pd.to_datetime(start_date))
        window = st.number_input("Rolling Window", value=int(window), step=1)
        n_assets = st.number_input("Number of assets", min_value=2, max_value=30,
                                   value=len(Tickers), step=1)
    with col2:
        end_date = st.date_input("End Date", pd.to_datetime(end_date))
        shift_days = st.number_input("Shift Days", value=int(shift_days), step=1)

    st.markdown("### Tickers")
    ticker_inputs = []
    for i in range(int(n_assets)):
        default_val = Tickers[i] if i < len(Tickers) else ""
        ticker_inputs.append(
            st.text_input(f"Ticker {i+1}", value=default_val, key=f"ticker_{i}")
        )

    submitted = st.form_submit_button("Analyze")

if submitted:
    Tickers = [t.strip() for t in ticker_inputs if t.strip() != ""]
    if len(Tickers) < 2:
        st.error("Please enter at least 2 valid tickers.")
        st.stop()

    st.write("""---""")
    st.write("Loading data...")

    def get_data(ticker, start, end, interval='1D'):
        data = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=True)
        if data.empty:
            st.error(f"❌ Error: Download for {ticker} returned an empty DataFrame. Check the ticker symbol or API rate limits.")
            st.stop()
        data = data.ffill().dropna()
        data['Returns'] = data['Close'].pct_change()
        data = data.dropna()
        data = data[['Close', 'Returns']]
        data.columns = pd.MultiIndex.from_tuples([(col, ticker) for col in data.columns], names=['Price', 'Ticker'])
        return data


    def analyze_returns(returns: pd.DataFrame, title: str = "PTF"):
        mean_return = returns.mean().iloc[0]
        annualized_return = mean_return * 252
        sd = returns.std().iloc[0] * (252 ** 0.5)
        infra = annualized_return / sd
        downside_returns = returns[returns < 0]
        dsd = downside_returns.std().iloc[0] * (252 ** 0.5)
        infra_down = annualized_return / dsd
        cum_ret = (1 + returns).cumprod()
        max_cum = cum_ret.cummax()
        drawdown = (max_cum - cum_ret) / max_cum
        Max_draw = drawdown.max().iloc[0]
        cal = annualized_return / Max_draw

        plt.figure(figsize=(10, 6))
        plt.plot(cum_ret, label="Cumulative Return")
        plt.yscale("log")
        plt.plot(max_cum, label="Maximum Cumulative Return")
        plt.fill_between(drawdown.index, drawdown.values.T[0] + 1, color="red", label="Drawdown", alpha=0.3)
        plt.title(title)
        plt.ylabel("Cumulative Return")
        plt.xlabel("Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.clf()

        st.header(f"Performance Metrics – {title}")
        perf_metrics = pd.DataFrame({
            "Metric": ["Mean Return", "Annualized Return", "Std Dev", "Sharpe Ratio", "Downside Dev", "Sortino Ratio",
                       "Max Drawdown", "Calmar Ratio"],
            "Value": [mean_return, annualized_return, sd, infra, dsd, infra_down, Max_draw, cal]
        })
        perf_metrics = perf_metrics.set_index("Metric")
        percent_metrics = ["Mean Return", "Annualized Return", "Std Dev", "Downside Dev", "Max Drawdown"]
        perf_metrics["Formatted Value"] = perf_metrics.apply(
            lambda row: "{:.2%}".format(row["Value"]) if row.name in percent_metrics else "{:.2}".format(row["Value"]),
            axis=1)
        st.dataframe(perf_metrics[["Formatted Value"]], use_container_width=True)

        annual_returns = (1 + returns).resample('YE').prod() - 1
        annual_returns.index = annual_returns.index.year
        values = annual_returns.iloc[:, 0]
        colors_annual = ['green' if val >= 0 else 'red' for val in values]
        monthly_rp_returns = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
        col_name = monthly_rp_returns.columns[0]
        monthly_rp_returns = monthly_rp_returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)

        plt.figure(figsize=(10, 6))
        plt.title(title)
        plt.bar(annual_returns.index, values * 100, color=colors_annual, label='Annual')
        plt.ylabel("Return (%)")
        plt.xlabel("Year")
        plt.legend()
        st.pyplot(plt.gcf())
        plt.clf()

        monthly_matrix = monthly_rp_returns.rename(columns={col_name: 'Return'})
        monthly_matrix['Year'] = monthly_matrix.index.year
        monthly_matrix['Month'] = monthly_matrix.index.strftime('%b')
        monthly_matrix.index.name = "Date"
        pivot_table_rp = monthly_matrix.pivot(index='Year', columns='Month', values='Return')
        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        pivot_table_rp = pivot_table_rp.reindex(columns=month_order)
        plt.figure(figsize=(15, 10))
        sns.heatmap(pivot_table_rp * 100, annot=True, fmt=".1f", cmap='RdYlGn', center=0)
        styled_pivot = pivot_table_rp.style.format("{:.1%}").background_gradient(cmap="RdYlGn", axis=1)
        st.dataframe(styled_pivot, use_container_width=True, height=700)
        plt.title('Monthly Return Heatmap: ' + str(title))
        plt.ylabel('Year')
        plt.xlabel('Month')


    def optimize_monthly_minvar(pk_returns: pd.DataFrame):
        def min_variance(weights, cov_matrix):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        monthly_returns = pk_returns.groupby(pd.Grouper(freq='ME'))
        optimized_returns = []
        for date, group in monthly_returns:
            if len(group) < 10:
                continue
            cov_matrix = group.cov()
            n = cov_matrix.shape[0]
            init_guess = np.ones(n) / n
            bounds = [(-1, 1)] * n
            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            result = sop.minimize(
                min_variance,
                init_guess,
                args=(cov_matrix,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            if not result.success:
                continue
            opt_w = result.x
            start = date + pd.Timedelta(days=1)
            end = date + pd.offsets.MonthEnd(1)
            try:
                next_month_returns = pk_returns.loc[start:end]
                port_return = next_month_returns @ opt_w
                optimized_returns.append(port_return)
            except KeyError:
                continue
        if len(optimized_returns) == 0:
            return pd.Series(dtype=float)
        minvar_returns_df = pd.concat(optimized_returns).sort_index()
        minvar_cum_returns = (1 + minvar_returns_df).cumprod()
        return minvar_cum_returns, minvar_returns_df


    def optimize_risk_parity_dynamic(pk_returns: pd.DataFrame, window=window, shift_days=shift_days):
        rolling_vol = pk_returns.rolling(window).std()
        inverse_rolling_vol = 1 / rolling_vol
        weights = inverse_rolling_vol.div(inverse_rolling_vol.sum(axis=1), axis=0).shift(shift_days).dropna()

        weights_returns = (weights * pk_returns).sum(axis=1)
        weights_cum_returns = (1 + weights_returns).cumprod()
        return weights_cum_returns, weights_returns


    def optimize_monthly_sharpe(pk_returns: pd.DataFrame):
        def sharpe_opt_no_rf(weights, mean_returns, cov_matrix):
            port_return = np.dot(weights, mean_returns)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return - port_return / port_vol

        monthly_returns = pk_returns.groupby(pd.Grouper(freq='ME'))
        optimized_returns = []

        for date, group in monthly_returns:
            if len(group) < 10:
                continue

            mean_returns = group.mean()
            cov_matrix = group.cov()
            n = len(mean_returns)
            init_guess = np.ones(n) / n
            bounds = [(-1, 1)] * n
            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

            result = sop.minimize(
                sharpe_opt_no_rf,
                init_guess,
                args=(mean_returns, cov_matrix),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )

            if not result.success:
                continue

            opt_w = result.x

            start = date + pd.Timedelta(days=1)
            end = date + pd.offsets.MonthEnd(1)

            try:
                next_month_returns = pk_returns.loc[start:end]
                port_return = next_month_returns @ opt_w
                optimized_returns.append(port_return)
            except KeyError:
                continue

        if len(optimized_returns) == 0:
            return pd.Series(dtype=float)

        optimized_returns_df = pd.concat(optimized_returns).sort_index()
        optimized_cum_returns = (1 + optimized_returns_df).cumprod()
        return optimized_cum_returns, optimized_returns_df


    def optimize_monthly_utility(pk_returns: pd.DataFrame):
        def utility_opt(weights, mean_returns, cov_matrix):
            port_return = np.dot(weights, mean_returns)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            utility = port_return - 0.5 * port_vol
            return -utility

        monthly_returns = pk_returns.groupby(pd.Grouper(freq='ME'))
        optimized_returns = []

        for date, group in monthly_returns:
            if len(group) < 10:
                continue

            mean_returns = group.mean()
            cov_matrix = group.cov()
            n = len(mean_returns)
            init_guess = np.ones(n) / n
            bounds = [(-1, 1)] * n
            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

            result = sop.minimize(
                utility_opt,
                init_guess,
                args=(mean_returns, cov_matrix),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )

            if not result.success:
                continue

            opt_w = result.x

            start = date + pd.Timedelta(days=1)
            end = date + pd.offsets.MonthEnd(1)

            try:
                next_month_returns = pk_returns.loc[start:end]
                port_return = next_month_returns @ opt_w
                optimized_returns.append(port_return)
            except KeyError:
                continue

        if len(optimized_returns) == 0:
            return pd.Series(dtype=float)

        optimized_utility_df = pd.concat(optimized_returns).sort_index()
        optimized_utility_cum = (1 + optimized_utility_df).cumprod()
        return optimized_utility_cum, optimized_utility_df


    Prices = {}
    for ticker in Tickers:
        Prices[ticker] = get_data(ticker, start_date, end_date)
    Prices = pd.concat(Prices.values(), axis=1)
    Prices.columns = pd.MultiIndex.from_tuples([(price, ticker) for (price, _), ticker in Prices.columns],
                                               names=['Price', 'Ticker'])
    Prices = Prices.sort_index(axis=1)
    Prices = Prices.dropna()

    returns = Prices['Returns']
    cumulative_returns = (1 + returns).cumprod()
    cumulative_returns.columns = pd.MultiIndex.from_product([['Cumulative Returns'], cumulative_returns.columns],
                                                            names=['Price', 'Ticker'])
    Prices = pd.concat([Prices, cumulative_returns], axis=1)
    Prices = Prices.sort_index(axis=1)

    a = Prices['Returns']
    b = returns.columns

    rolling_corrs = {}

    for i in range(len(b)):
        for j in range(i + 1, len(b)):
            t1 = b[i]
            t2 = b[j]
            pair_name = f"{t1} ~ {t2}"
            rolling_corrs[pair_name] = a[t1].rolling(window).corr(a[t2])
    rolling_corrs = pd.DataFrame(rolling_corrs).dropna(how="all")

    port1 = [Tickers[0], Tickers[1]]
    port2 = [Tickers[1], Tickers[0], Tickers[2]]
    port3 = [Tickers[1], Tickers[0], Tickers[2], Tickers[3]]

    pa = Prices['Returns'][port1]
    pb = Prices['Returns'][port2]
    pk = Prices['Returns'][port3]

    optimized_strategies_ret = pd.concat([
        pd.concat({
            'Max Sharpe': optimize_monthly_sharpe(pa)[1],
            'Max Utility': optimize_monthly_utility(pa)[1],
            'Min Variance': optimize_monthly_minvar(pa)[1],
            'Risk Parity': optimize_risk_parity_dynamic(pa, window, shift_days)[1],
        }, axis=1, names=['Strategy']),

        pd.concat({
            'Max Sharpe': optimize_monthly_sharpe(pb)[1],
            'Max Utility': optimize_monthly_utility(pb)[1],
            'Min Variance': optimize_monthly_minvar(pb)[1],
            'Risk Parity': optimize_risk_parity_dynamic(pb, window, shift_days)[1],
        }, axis=1, names=['Strategy']),

        pd.concat({
            'Max Sharpe': optimize_monthly_sharpe(pk)[1],
            'Max Utility': optimize_monthly_utility(pk)[1],
            'Min Variance': optimize_monthly_minvar(pk)[1],
            'Risk Parity': optimize_risk_parity_dynamic(pk, window, shift_days)[1],
        }, axis=1, names=['Strategy'])
    ], axis=1, keys=[
        " + ".join(port1),
        " + ".join(port2),
        " + ".join(port3)
    ], names=['Portfolio'])
    optimized_strategies_ret = optimized_strategies_ret.dropna(how='any')

    optimized_strategies_cum_ret = pd.concat([
        pd.concat({
            'Max Sharpe': optimize_monthly_sharpe(pa)[0],
            'Max Utility': optimize_monthly_utility(pa)[0],
            'Min Variance': optimize_monthly_minvar(pa)[0],
            'Risk Parity': optimize_risk_parity_dynamic(pa, window, shift_days)[0],
        }, axis=1, names=['Strategy']),

        pd.concat({
            'Max Sharpe': optimize_monthly_sharpe(pb)[0],
            'Max Utility': optimize_monthly_utility(pb)[0],
            'Min Variance': optimize_monthly_minvar(pb)[0],
            'Risk Parity': optimize_risk_parity_dynamic(pb, window, shift_days)[0],
        }, axis=1, names=['Strategy']),

        pd.concat({
            'Max Sharpe': optimize_monthly_sharpe(pk)[0],
            'Max Utility': optimize_monthly_utility(pk)[0],
            'Min Variance': optimize_monthly_minvar(pk)[0],
            'Risk Parity': optimize_risk_parity_dynamic(pk, window, shift_days)[0],
        }, axis=1, names=['Strategy'])
    ], axis=1, keys=[
        " + ".join(port1),
        " + ".join(port2),
        " + ".join(port3)
    ], names=['Portfolio'])
    optimized_strategies_cum_ret = optimized_strategies_cum_ret.dropna(how='any')

    for ticker in Prices['Returns'].columns:
        title = f"Single Asset | {ticker}"
        st.subheader(f"\n=== Analyzing: {title} ===")
        analyze_returns(Prices['Returns'][[ticker]], title=title)

    for portfolio in optimized_strategies_ret.columns.levels[0]:
        for strategy in optimized_strategies_ret[portfolio].columns:
            title = f"{portfolio} | {strategy}"
            st.subheader(f"\n=== Analyzing: {title} ===")
            analyze_returns(optimized_strategies_ret[portfolio][[strategy]], title=title)

    for portfolio in optimized_strategies_cum_ret.columns.levels[0]:
        plt.figure(figsize=(10, 6))
        for strategy in optimized_strategies_cum_ret[portfolio].columns:
            plt.plot(optimized_strategies_cum_ret[portfolio][strategy], label=strategy)
        plt.title(f"Performance Comparison: {portfolio}")
        plt.ylabel("Cumulative Return")
        plt.xlabel("Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.clf()

    for strategy in optimized_strategies_cum_ret.columns.levels[1]:
        plt.figure(figsize=(10, 6))
        for portfolio in optimized_strategies_cum_ret.columns.levels[0]:
            plt.plot(optimized_strategies_cum_ret[portfolio][strategy], label=portfolio)
        plt.title(f"Performance Comparison: {strategy}")
        plt.ylabel("Cumulative Return")
        plt.xlabel("Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.clf()

    st.header("\nCorrelation: Full Sample and Rolling\n")

    for pair in rolling_corrs.columns:
        try:
            ticker1, ticker2 = pair.split(" ~ ")
            corr_static = Prices['Returns'][[ticker1, ticker2]].corr().loc[ticker1, ticker2]

            st.header(f"=== {pair} ===")
            st.markdown("**Rolling Correlation Table:**")
            df_corr = rolling_corrs[pair].dropna().to_frame(name="Rolling Corr")
            df_corr.index.name = "Date"
            df_corr["Rolling Corr"] = df_corr["Rolling Corr"].map("{:.2f}".format)
            st.dataframe(df_corr, use_container_width=True)

            plt.figure(figsize=(10, 6))
            plt.plot(rolling_corrs[pair].dropna(), label="Rolling Correlation")
            plt.axhline(corr_static, color='red', linestyle='--', label=f"Full Sample: {corr_static:.2f}")
            plt.title(f"Rolling vs Full Sample Correlation: {pair}")
            plt.ylabel("Correlation")
            plt.xlabel("Time")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            st.pyplot(plt.gcf())
            plt.clf()

        except Exception as e:
            st.markdown(f"Skipping {pair} due to error: {e}")

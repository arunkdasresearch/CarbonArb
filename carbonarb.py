import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os

###############################################################################
#                         HELPER FUNCTIONS
###############################################################################

def calculate_rho(beta):
    """Helper function for discount rate calculation."""
    rho_f = 0.0208
    carp = 0.0299 - 0.01  # Always subtract 1%
    _lambda = 0.5175273490449868  # Weighted leverage factor
    tax_rate = 0.15  # Corporate tax rate
    rho = _lambda * rho_f * (1 - tax_rate) + (1 - _lambda) * (rho_f + beta * carp)
    return rho

def calculate_discount(rho, deltat):
    """Discount factor for deltat years ahead at rate rho."""
    return (1 + rho) ** (-deltat)

def EJ2MWh(x):
    """Convert exajoules (EJ) to megawatt-hours (MWh)."""
    return x * 1e18 / 3600 / 1e6  # Convert to MWh

def EJ2Mcoal(x):
    """Convert exajoules (EJ) to million tonnes of coal (approx)."""
    return x * 1e9 / 29.3076 / 1e6

def calculate_emissions_and_production(scenario, df_ngfs, beta):
    """Returns emissions, 2022 coal production, and discounted coal production for a given scenario."""
    coal_emissions_2022_iea = 15.5  # GtCO2
    years_5_step = list(range(2010, 2101, 5))
    full_years = range(2023, 2101)

    sub = df_ngfs[df_ngfs.Scenario == scenario]
    emissions_row = sub[sub.Variable == "Emissions|CO2"].iloc[0]
    emissions_values = [emissions_row[str(y)] / 1e3 for y in years_5_step]
    f_e = interp1d(years_5_step, emissions_values)

    total_emissions = sum(f_e(y) for y in full_years)
    total_emissions *= coal_emissions_2022_iea / f_e(2022)

    production_row = sub[sub.Variable == "Primary Energy|Coal"].iloc[0]
    production_values = [production_row[str(y)] for y in years_5_step]
    f_p = interp1d(years_5_step, production_values)

    rho = calculate_rho(beta)
    production_discounted = sum(f_p(y) * calculate_discount(rho, y - 2022) for y in full_years)

    return {
        "emissions": total_emissions,
        "production_2022": EJ2Mcoal(f_p(2022)),
        "production_discounted": production_discounted,
    }

def calculate_cost_and_benefit(social_cost_of_carbon, global_lcoe_average, beta, df_ngfs):
    """Computes cost, benefit, and arbitrage opportunity."""
    ep_cps = calculate_emissions_and_production("NGFS2_Current Policies", df_ngfs, beta)
    ep_nz2050 = calculate_emissions_and_production("NGFS2_Net-Zero 2050", df_ngfs, beta)

    avoided_emissions = ep_cps["emissions"] - ep_nz2050["emissions"]
    discounted_production_increase = ep_cps["production_discounted"] - ep_nz2050["production_discounted"]
    discounted_production_increase_mwh = EJ2MWh(discounted_production_increase)

    cost = global_lcoe_average * discounted_production_increase_mwh / 1e12
    benefit = avoided_emissions * social_cost_of_carbon / 1e3

    return {
        "avoided_emissions": avoided_emissions,
        "cost": cost,
        "benefit": benefit,
        "arbitrage": benefit - cost,
        "coal_2022": ep_cps["production_2022"]
    }

###############################################################################
#                         MAIN STREAMLIT APP
###############################################################################

def main():
    st.title("My Carbon Arbitrage Opportunity Calculator")

    st.sidebar.header("Model Parameters")
    social_cost_of_carbon = st.sidebar.slider("Social Cost of Carbon (USD per ton CO₂)", 1, 200, 80)
    global_lcoe_average = st.sidebar.slider("Global average LCOE (USD/MWh for renewables)", 1.0, 200.0, 59.25)
    beta = st.sidebar.number_input("Unleveraged beta (advanced)", 0.0, 2.0, 0.91, step=0.01)

    data_path = os.path.join("data", "ar6_snapshot_1700882949.csv")
    if not os.path.isfile(data_path):
        st.error(f"CSV file not found at: {data_path}")
        return

    df_ngfs = pd.read_csv(data_path)
    results = calculate_cost_and_benefit(social_cost_of_carbon, global_lcoe_average, beta, df_ngfs)

    st.subheader("Results")
    st.write(f"**Total emissions prevented**: {results['avoided_emissions']:.2f} GtCO₂")
    st.write(f"**Cost**: {results['cost']:.2f} trillion dollars")
    st.write(f"**Benefit**: {results['benefit']:.2f} trillion dollars")
    st.write(f"**Carbon arbitrage opportunity**: {results['arbitrage']:.2f} trillion dollars")

    st.subheader("Parameter Sweep Plots")
    param_choice = st.selectbox("Which parameter would you like to sweep?",
                                ["Social Cost of Carbon", "Global LCOE", "Beta"])

    if param_choice == "Social Cost of Carbon":
        param_values = np.linspace(10, 200, 20)
    elif param_choice == "Global LCOE":
        param_values = np.linspace(10, 200, 20)
    else:
        param_values = np.linspace(0.0, 2.0, 20)

    emissions, cost, benefit, arbitrage = [], [], [], []
    for val in param_values:
        res = calculate_cost_and_benefit(
            val if param_choice == "Social Cost of Carbon" else social_cost_of_carbon,
            val if param_choice == "Global LCOE" else global_lcoe_average,
            val if param_choice == "Beta" else beta,
            df_ngfs
        )
        emissions.append(res["avoided_emissions"])
        cost.append(res["cost"])
        benefit.append(res["benefit"])
        arbitrage.append(res["arbitrage"])

    st.write("### 📊 Sweeping **{}**".format(param_choice))

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    axs[0, 0].plot(param_values, emissions, label="Total Emissions Prevented", color="blue")
    axs[0, 0].set_title("Total Emissions Prevented (GtCO2)")
    axs[0, 0].set_xlabel(param_choice)
    axs[0, 0].set_ylabel("GtCO₂")

    axs[0, 1].plot(param_values, cost, label="Cost", color="red")
    axs[0, 1].set_title("Cost (trillion $)")
    axs[0, 1].set_xlabel(param_choice)
    axs[0, 1].set_ylabel("Trillion USD")

    axs[1, 0].plot(param_values, benefit, label="Benefit", color="green")
    axs[1, 0].set_title("Benefit (trillion $)")
    axs[1, 0].set_xlabel(param_choice)
    axs[1, 0].set_ylabel("Trillion USD")
 
    axs[1, 1].plot(param_values, arbitrage, label="Carbon Arbitrage", color="purple")
    axs[1, 1].set_title("Carbon Arbitrage (trillion $)")
    axs[1, 1].set_xlabel(param_choice)
    axs[1, 1].set_ylabel("Trillion USD")

    plt.tight_layout()
    st.pyplot(fig)

if __name__ == "__main__":
    main()

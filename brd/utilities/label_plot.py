from prettyPlot.plotting import plt, pretty_labels, pretty_legend


def label_conv(input_string):
    if input_string.lower() == "width":
        return "width [mm]"
    elif input_string.lower() == "spacing":
        return "spacing [mm]"
    elif input_string.lower() == "height":
        return "height [mm]"
    elif (
        input_string.lower() == "co2_liq"
        or input_string.lower() == "co2.liquid"
    ):
        return r"$Y_{CO_2, liq}$"
    elif (
        input_string.lower() == "co_liq" or input_string.lower() == "co.liquid"
    ):
        return r"$Y_{CO, liq}$"
    elif (
        input_string.lower() == "h2_liq" or input_string.lower() == "h2.liquid"
    ):
        return r"$Y_{H_2, liq}$"
    elif (
        input_string.lower() == "co2_gas" or input_string.lower() == "co2.gas"
    ):
        return r"$Y_{CO_2, gas}$"
    elif input_string.lower() == "co_gas" or input_string.lower() == "co.gas":
        return r"$Y_{CO, gas}$"
    elif input_string.lower() == "h2_gas" or input_string.lower() == "h2.gas":
        return r"$Y_{H_2, gas}$"
    elif input_string.lower() == "kla_h2":
        return r"$KLA_{H_2}$"
    elif input_string.lower() == "kla_co":
        return r"$KLA_{CO}$"
    elif input_string.lower() == "kla_co2":
        return r"$KLA_{CO_2}$"
    elif input_string.lower() == "alpha.gas":
        return r"$\alpha_{gas}$"
    elif (
        input_string.lower() == "d.gas"
        or input_string.lower() == "d"
        or input_string.lower() == "bubblediam"
    ):
        return "Mean bubble diam [m]$"
    elif input_string.lower() == "y":
        return "y [m]"
    elif input_string.lower() == "t":
        return "t [s]"
    elif input_string.lower() == "gh":
        return "Gas holdup"
    elif input_string.lower() == "gh_height":
        return "Height-based gas holdup"
    else:
        print(input_string)
        return input_string

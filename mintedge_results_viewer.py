import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import Button


POWER_COLUMNS = ["dynamic_W_servers", "idle_W_servers", "W_links"]


def positive_integer(value):
    """Accept only positive integer values."""
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("The group size must be an integer.") from error

    if number <= 0:
        raise argparse.ArgumentTypeError("The group size must be greater than 0.")

    return number


def load_data(csv_path):
    """Load the CSV file and validate the required columns."""
    data = pd.read_csv(csv_path)

    required_columns = ["time"] + POWER_COLUMNS
    missing_columns = [
        column for column in required_columns if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required CSV columns: " + ", ".join(missing_columns)
        )

    if data.empty:
        raise ValueError("The CSV file does not contain data.")

    data = data.sort_values("time").reset_index(drop=True)

    for column in required_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    if data[required_columns].isna().any().any():
        raise ValueError(
            "The columns time, dynamic_W_servers, idle_W_servers and "
            "W_links must contain numeric values."
        )

    return data


def calculate_hourly_energy(data):
    """
    Calculate energy for each 3600-second execution hour.

    MintEDGE stores one power measurement per second, so:
        energy in Wh = sum of power samples in W / 3600
    """
    hourly = data.copy()
    hourly["total_power_W"] = hourly[POWER_COLUMNS].sum(axis=1)
    hourly["execution_hour"] = ((hourly["time"] - 1) // 3600 + 1).astype(int)

    result = hourly.groupby("execution_hour").agg(
        energy_Wh=("total_power_W", lambda values: values.sum() / 3600),
        average_power_W=("total_power_W", "mean"),
        minimum_power_W=("total_power_W", "min"),
        maximum_power_W=("total_power_W", "max"),
        samples=("total_power_W", "size"),
    )

    result["change_percent"] = result["energy_Wh"].pct_change() * 100
    result["difference_Wh"] = result["energy_Wh"].diff()

    return result.reset_index()


def calculate_average_utilization(data, group_size):
    """Calculate the mean utilization of all available servers."""
    utilization_columns = [
        column for column in data.columns if column.startswith("server_util_")
    ]

    if not utilization_columns:
        raise ValueError(
            "No server utilization columns were found. "
            "Expected columns starting with 'server_util_'."
        )

    utilization = data[utilization_columns].apply(
        pd.to_numeric, errors="coerce"
    ).mean(axis=1)

    if group_size is None:
        return data["time"].to_numpy(), utilization.to_numpy(), 1

    groups = np.arange(len(data)) // group_size
    grouped_time = data.groupby(groups)["time"].mean()
    grouped_utilization = utilization.groupby(groups).mean()

    return (
        grouped_time.to_numpy(),
        grouped_utilization.to_numpy(),
        group_size,
    )


class PlotNavigator:
    """Show the three result screens with Previous and Next buttons."""

    def __init__(self, hourly, utilization_time, utilization, group_size):
        self.hourly = hourly
        self.utilization_time = utilization_time
        self.utilization = utilization
        self.group_size = group_size
        self.current_screen = 0

        self.figure, self.axis = plt.subplots(figsize=(13, 7))
        plt.subplots_adjust(bottom=0.16)

        previous_axis = plt.axes([0.36, 0.04, 0.12, 0.06])
        next_axis = plt.axes([0.52, 0.04, 0.12, 0.06])

        self.previous_button = Button(previous_axis, "Previous")
        self.next_button = Button(next_axis, "Next")

        self.previous_button.on_clicked(self.show_previous)
        self.next_button.on_clicked(self.show_next)

        self.draw_current_screen()

    def show_previous(self, _event):
        self.current_screen = (self.current_screen - 1) % 3
        self.draw_current_screen()

    def show_next(self, _event):
        self.current_screen = (self.current_screen + 1) % 3
        self.draw_current_screen()

    def draw_current_screen(self):
        self.axis.clear()

        if self.current_screen == 0:
            self.draw_hourly_energy()
        elif self.current_screen == 1:
            self.draw_hour_comparison()
        else:
            self.draw_average_utilization()

        self.figure.canvas.draw_idle()

    def draw_hourly_energy(self):
        hours = self.hourly["execution_hour"].to_numpy()
        energy = self.hourly["energy_Wh"].to_numpy()

        self.axis.bar(hours, energy, label="Energy per hour")

        if len(hours) == 1:
            trend = np.array([energy[0]])
        else:
            trend_function = np.poly1d(np.polyfit(hours, energy, 1))
            trend = trend_function(hours)

        self.axis.plot(
            hours,
            trend,
            marker="o",
            linewidth=2,
            label="Linear trend",
        )

        for hour, value in zip(hours, energy):
            self.axis.text(
                hour,
                value,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        self.axis.set_title("Energy consumed during each execution hour")
        self.axis.set_xlabel("Execution hour")
        self.axis.set_ylabel("Energy consumed (Wh)")
        self.axis.set_xticks(hours)
        self.axis.grid(axis="y", alpha=0.3)
        self.axis.legend()
        self.axis.text(
            0.99,
            1.02,
            "Screen 1 of 3",
            transform=self.axis.transAxes,
            ha="right",
        )

    def draw_hour_comparison(self):
        report = self.hourly.copy()

        report["execution_hour"] = report["execution_hour"].map(
            lambda value: f"Hour {value}"
        )
        report["energy_Wh"] = report["energy_Wh"].map(lambda value: f"{value:.2f}")
        report["difference_Wh"] = report["difference_Wh"].map(
            lambda value: "-" if pd.isna(value) else f"{value:+.2f}"
        )
        report["change_percent"] = report["change_percent"].map(
            lambda value: "-" if pd.isna(value) else f"{value:+.2f}%"
        )
        report["average_power_W"] = report["average_power_W"].map(
            lambda value: f"{value:.2f}"
        )
        report["minimum_power_W"] = report["minimum_power_W"].map(
            lambda value: f"{value:.2f}"
        )
        report["maximum_power_W"] = report["maximum_power_W"].map(
            lambda value: f"{value:.2f}"
        )

        columns = [
            "execution_hour",
            "energy_Wh",
            "difference_Wh",
            "change_percent",
            "average_power_W",
            "minimum_power_W",
            "maximum_power_W",
            "samples",
        ]
        labels = [
            "Hour",
            "Energy\n(Wh)",
            "Difference\n(Wh)",
            "Change",
            "Average\npower (W)",
            "Minimum\npower (W)",
            "Maximum\npower (W)",
            "Seconds",
        ]

        self.axis.axis("off")
        table = self.axis.table(
            cellText=report[columns].values,
            colLabels=labels,
            cellLoc="center",
            colLoc="center",
            loc="center",
        )

        row_count = max(len(report), 1)
        table.auto_set_font_size(False)
        table.set_fontsize(max(6, min(10, 220 / row_count)))
        table.scale(1, max(0.75, min(1.5, 18 / row_count)))

        self.axis.set_title(
            "Comparison between execution hours\n"
            "Positive changes mean that energy consumption increased",
            pad=25,
        )
        self.axis.text(
            0.99,
            1.02,
            "Screen 2 of 3",
            transform=self.axis.transAxes,
            ha="right",
        )

    def draw_average_utilization(self):
        self.axis.plot(
            self.utilization_time,
            self.utilization * 100,
            marker="o",
            markersize=3,
        )

        if self.group_size == 1:
            group_text = "No grouping: every CSV point is shown"
        else:
            group_text = (
                f"Each point is the average of {self.group_size} CSV points"
            )

        self.axis.set_title(f"Average server utilization over time\n{group_text}")
        self.axis.set_xlabel("Simulation time (seconds)")
        self.axis.set_ylabel("Average server utilization (%)")
        self.axis.set_ylim(0, 100)
        self.axis.grid(alpha=0.3)
        self.axis.text(
            0.99,
            1.02,
            "Screen 3 of 3",
            transform=self.axis.transAxes,
            ha="right",
        )

    def show(self):
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Display MintEDGE energy and utilization results."
    )
    parser.add_argument("csv_file", help="Path to the MintEDGE CSV result file")
    parser.add_argument(
        "group_size",
        nargs="?",
        type=positive_integer,
        default=None,
        help=(
            "Optional number of CSV points used for each utilization average. "
            "Only positive integers are accepted."
        ),
    )
    args = parser.parse_args()

    data = load_data(args.csv_file)
    hourly = calculate_hourly_energy(data)
    utilization_time, utilization, used_group_size = (
        calculate_average_utilization(data, args.group_size)
    )

    navigator = PlotNavigator(
        hourly,
        utilization_time,
        utilization,
        used_group_size,
    )
    navigator.show()


if __name__ == "__main__":
    main()

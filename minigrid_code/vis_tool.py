import pandas as pd

pd.options.mode.chained_assignment = (
    None  # ignore all warnings like SettingWithCopyWarning
)

import numpy as np
from typing import Callable
import os, glob
import shutil
import json


def walk_through(
    path: str,
    metric: str,
    query_fn: Callable,
    start: int,
    end: int,
    steps: int,
    window: int,
    cutoff: float = 0.95,
    extrapolate: bool = False,
    delete: bool = False,
):
    def isnan(number):
        return np.isnan(float(number))

    def smooth(df):
        try:
            df = df.dropna(subset=[metric])  # remove NaN rows
        except KeyError:
            print("!!key error csv", run)
            if delete:
                shutil.rmtree(run)
                print("deleted")
            return None

        if isnan(df["env_steps"].iloc[-1]) or df["env_steps"].iloc[-1] < cutoff * end:
            # an incomplete run
            print("!!incomplete csv", run, df["env_steps"].iloc[-1], end=" ")
            if delete:
                shutil.rmtree(run)
                print("deleted")
            else:
                print("\n")
            return None

        # smooth by moving average
        df[metric] = df[metric].rolling(window=window, min_periods=1).mean()

        # update the columns with interpolated values and aligned steps
        aligned_step = np.linspace(start, end, steps).astype(np.int32)
        if not extrapolate:
            ## we only do interpolation, not extrapolation
            aligned_step = aligned_step[aligned_step <= df["env_steps"].iloc[-1]]
        aligned_value = np.interp(aligned_step, df["env_steps"], df[metric])

        # enlarge or reduce to same number of rows
        print(run, df.shape[0], df["env_steps"].iloc[-1])

        extracted_df = pd.DataFrame(
            data={
                "env_steps": aligned_step,
                metric: aligned_value,
            }
        )

        return extracted_df

    dfs = []
    i = 0

    runs = sorted(glob.glob(os.path.join(path, "*")))

    for run in runs:
        with open(os.path.join(run, "config.json")) as f:
            flags = json.load(f)

        if not query_fn(flags):
            continue

        csv_path = os.path.join(run, "progress.csv")
        try:
            df = pd.read_csv(open(csv_path))
        except pd.errors.EmptyDataError:
            print("!!empty csv", run)
            if delete:
                shutil.rmtree(run)
                print("deleted")
            continue

        df = smooth(df)
        if df is None:
            continue
        i += 1

        # concat flags (dot)
        pd_flags = pd.json_normalize(flags)
        df_flag = pd.concat([pd_flags] * df.shape[0], axis=0)  # repeat rows
        df_flag.index = df.index  # copy index
        df = pd.concat([df, df_flag], axis=1)
        dfs.append(df)

    print("\n in total:", i)
    dfs = pd.concat(dfs, ignore_index=True)
    return dfs

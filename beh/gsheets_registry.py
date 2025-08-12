import time
import gspread
import jax
from google.oauth2.service_account import Credentials

from beh.core.registry import *

sheet_id = "1b7f7NInSCiL8j7LPke36DgO3PDBNEITAP3ytK95m_RM"
scopes = ["https://www.googleapis.com/auth/spreadsheets"]
creds = Credentials.from_service_account_file("./keys/gsheets_credentials.json", scopes=scopes)
gc = gspread.authorize(creds)

def format_res(x):
    return f"{round(x, 2):.2f}"

def gsheet_log_row(
    data_name,
    accelerator,
    worksheet_name,
    dimension, 
    metric, 
    n = "-", 
    mlp4 = "-",
    mlp16 = "-",
    mlp128 = "-",
    moe4_d = "-",
    moe16_d = "-",
    moe128_d = "-",
    moe4_s = "-",
    moe16_s = "-",
    moe128_s = "-"):
    '''
    Log a row of results in google sheets using google sheets api.
    '''
    row = [
        metric,
        n,
        mlp4,
        mlp16,
        mlp128,
        moe4_d,
        moe16_d,
        moe128_d,
        moe4_s,
        moe16_s,
        moe128_s,
        dimension,
        data_name,
        accelerator, # GPU / CPU
        time.strftime("%Y-%m-%d %H:%M:%S"), # Timestamp
        "FALSE"  # Set approve flag initially to false to avoid push to overleaf
    ]
    ws = gc.open_by_key(sheet_id).worksheet(worksheet_name)
    ws.append_row(row, value_input_option="RAW")

def gsheet_log_results(dimension : int, reg : CoreRegistry, configs, data_name : str):
    '''
    Log all results in google sheets using google sheets api.
    '''
    
    # Format for main table
    worksheet_name = 'table1'
    time_1_key, time_2_key = "Time - ms", "Time - factor"
    fp_1_key = "False Positive - Ratio"
    para_1_key, para_2_key = "Active Parameter - Abs.", "Active / Total Parameter"
    metrics = {
        time_1_key : [],
        time_2_key : [],
        fp_1_key : [],
        para_1_key : [],
        para_2_key: [],
    }

    # First model in config is baseline speed
    baseline_key = list(configs)[1]
    baselin_speed = reg.get(baseline_key + core_keys['inf_speed_key']) / 1000 # Divide to convert microseconds to miliseconds

    def append_metrics(model_key, metrics):
        # Inference Speed
        speed = reg.get(model_key + core_keys['inf_speed_key']) / 1000 # Divide to convert microseconds to miliseconds
        metrics[time_1_key].append(speed)
        metrics[time_2_key].append(speed / baselin_speed)

        # False Positive Rate
        metrics[fp_1_key].append(reg.get(model_key + core_keys['fp_key']))

        # Parameters
        metrics[para_1_key].append(reg.get(model_key + core_keys['active_parameters_key']))
        metrics[para_2_key].append(reg.get(model_key + core_keys['active_parameters_key']) / reg.get(model_key + core_keys['total_parameters_key']))

        return metrics 

    ## Loop over all configured models for that dimension
    for model_key, _ in configs.items():
        if model_key == 'general': continue
        if configs[model_key]['type'] == 'moe':
            model_key = f'{model_key}_dense'
        metrics = append_metrics(model_key, metrics)
    
    ## Ensure we append all sparse results after all other to have them 
    ## as one block of results together
    for model_key, _ in configs.items():
        if model_key == 'general': continue
        if configs[model_key]['type'] == 'moe':
            model_key = f'{model_key}_sparse'
            metrics = append_metrics(model_key, metrics)
    
    ## Log all metric rows for each model
    for metric_name, value in metrics.items():
        # Format numbers if neccesary 
        if metric_name in [fp_1_key, time_1_key, time_2_key, para_2_key] :
            value = [format_res(x) for x in value]

        gsheet_log_row(
            data_name = data_name,
            accelerator = jax.devices()[0].device_kind,
            worksheet_name = worksheet_name,
            dimension = dimension, 
            metric = metric_name, 

            n = 1, 

            mlp4 = value[0],
            mlp16 = value[1],
            mlp128 = value[2],

            moe4_d = value[3],
            moe16_d = value[4],
            moe128_d = value[5],

            moe4_s = value[6],
            moe16_s = value[7],
            moe128_s = value[8])
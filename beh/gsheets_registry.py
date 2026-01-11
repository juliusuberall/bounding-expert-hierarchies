import time
import gspread
import jax

from beh.core.shared import pe_dim
from google.oauth2.service_account import Credentials

from beh.core.registry import *

sheet_id = "1bK4UXi6Orcgq8jQwv1Ikeq5sa5qIWIk-ZQxrH71axX8"
scopes = ["https://www.googleapis.com/auth/spreadsheets"]
creds = Credentials.from_service_account_file("./keys/gsheets_credentials.json", scopes=scopes)
gc = gspread.authorize(creds)

def gsheet_log_row(
    worksheet_name,
    dimension, 
    data_name,
    accelerator,
    chart_key,
    model_type,
    pattern,
    size,
    architecture,
    pe_num_freq,
    total_p,
    active_p,
    epochs,
    batch_speeds,
    timing_setup,
    fp,
    fn,
    train_time,
    exp_time):
    '''
    Log a row of results in google sheets using google sheets api.
    '''
    row = [
        dimension,
        data_name,
        accelerator, # GPU / CPU
        chart_key,
        model_type,
        pattern,
        size,
        architecture,
        pe_num_freq,
        total_p,
        active_p,
        epochs,
        round(float(batch_speeds[0][1]), 2),
        timing_setup,
        round(float(fp), 4),
        fn,
        train_time,
        exp_time,
        time.strftime("%Y-%m-%d %H:%M:%S") # Timestamp
    ]
    ws = gc.open_by_key(sheet_id).worksheet(worksheet_name)
    ws.append_row(row, value_input_option="RAW")

def generate_key(type:str, pattern:str, size:str) -> str:
    pattern = ""
    if type == 'moe':
        pattern = "Sp" if pattern == "Sparse" else "Dn"
        pattern = "-" + pattern
    key = f"{type.upper()}{pattern}-{size.upper()}"
    return key

def gsheet_log_results(model_key : str, dimension : int, reg : CoreRegistry, configs, data_name : str):
    '''
    Log all results in google sheets using google sheets api.
    '''
    # Set basic row arguments
    worksheet_name = 'last_run'
    accelerator = jax.devices()[0].device_kind
    model_type = configs[model_key]['type']
    pe_num_freq = configs['general']['pe_num_freq']
    
    # Extract model type specific results 
    if model_type == 'moe':
        m_key = f'{model_key}_dense'
    else:
        m_key = model_key

    if model_type in ['moe', 'moeg', 'mlp']:
        epochs = reg.get(model_key + core_keys['total_epochs'])
        arch = reg.get(model_key + core_keys['architecture'])
        pattern = 'Dense'
        if model_type == 'moeg':
            pattern = 'Sparse'
    
    elif model_type == 'bvh':
        pattern = 'Sparse'
        pe_num_freq = 0
        arch = f"Max-Depth: {configs[model_key]['max_depth']}"
        epochs = "None"
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Log results online 
    size = model_key[-1].capitalize()
    gsheet_log_row(
        worksheet_name   = worksheet_name,
        dimension        = dimension,
        data_name        = data_name,
        accelerator      = accelerator,
        chart_key        = generate_key(model_type, pattern, size),
        model_type       = model_type,
        pattern          = pattern,
        size             = size,
        architecture     = arch,
        pe_num_freq      = pe_num_freq,
        total_p          = float(reg.get(m_key + core_keys['total_parameters_key'])),
        active_p         = float(reg.get(m_key + core_keys['active_parameters_key'])),
        epochs           = epochs,
        batch_speeds     = reg.get(m_key + core_keys['inf_speed_key']),
        timing_setup     = f"{configs['general']['inf_bench_query_size']/1e06}M queries | {configs['general']['inf_bench_repitions']/1e03}K reps",
        fp               = float(reg.get(m_key + core_keys['fp_key'])),
        fn               = float(reg.get(m_key + core_keys['fn_key'])),
        train_time       = round(float(reg.get(model_key + core_keys['training_time'])), 2),
        exp_time         = round(float((time.perf_counter_ns() - reg.get(model_key + core_keys['experiment_start_time_key'])) / 6e10), 2),
    )

    # Second result logging to add sparse MoE results immediatly after the dense results
    if model_type == 'moe':
        m_key = f'{model_key}_sparse'
        gsheet_log_row(
            worksheet_name   = worksheet_name,
            dimension        = dimension,
            data_name        = data_name,
            accelerator      = accelerator,
            chart_key        = generate_key(model_type, pattern, size),
            model_type       = model_type,
            pattern          = 'Sparse',
            size             = size,
            architecture     = arch,
            pe_num_freq      = pe_num_freq,
            total_p          = float(reg.get(m_key + core_keys['total_parameters_key'])),
            active_p         = float(reg.get(m_key + core_keys['active_parameters_key'])),
            epochs           = epochs,
            batch_speeds     = reg.get(m_key + core_keys['inf_speed_key']),
            timing_setup     = f"{configs['general']['inf_bench_query_size']/1e06}M queries | {configs['general']['inf_bench_repitions']/1e03}K reps",
            fp               = float(reg.get(m_key + core_keys['fp_key'])),
            fn               = float(reg.get(m_key + core_keys['fn_key'])),
            train_time       = '"',
            exp_time         = '"',
        )
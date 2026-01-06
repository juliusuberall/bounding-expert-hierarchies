import time
import gspread
import jax
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
    total_p,
    active_p,
    epochs,
    batch_speeds,
    timing_setup,
    fp,
    fn,
    train_time):
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
        total_p,
        active_p,
        epochs,
        float(batch_speeds[0][1]),
        timing_setup,
        fp,
        fn,
        train_time,
        "FALSE",  # Set approve flag initially to false to avoid push to overleaf
        time.strftime("%Y-%m-%d %H:%M:%S") # Timestamp
    ]
    ws = gc.open_by_key(sheet_id).worksheet(worksheet_name)
    ws.append_row(row, value_input_option="RAW")

def gsheet_log_results(model_key : str, dimension : int, reg : CoreRegistry, configs, data_name : str):
    '''
    Log all results in google sheets using google sheets api.
    '''
    # Set basic row arguments
    worksheet_name = 'last_run'
    accelerator = jax.devices()[0].device_kind
    model_type = configs[model_key]['type']
    m_key = model_key
    
    # Extract model type specific results 
    if model_type in ['moe', 'moe_grid', 'mlp']:
        epochs = reg.get(model_key + core_keys['total_epochs'])

    if model_type == 'moe':
        nex = configs[model_key]['nex']
        gate_hid_lay = configs[model_key]['gate_hidden_layer']
        expert_hid_lay = configs[model_key]['expert_hidden_layer']

        m_key = f'{model_key}_dense'
        gate_arch = [dimension] + gate_hid_lay + [nex]
        expert_arch = [dimension] + expert_hid_lay + [1]
        arch = f'G {str(gate_arch).replace(" ", "")}    {nex}x E {str(expert_arch).replace(" ", "")}'
        pattern = 'Dense'

    elif model_type == 'moe_grid':
        nex = configs[model_key]['grid_dim'] ** dimension
        expert_hid_lay = configs[m_key]['expert_hidden_layer']
        expert_arch = [dimension] + expert_hid_lay + [1]
        arch = f'{nex}x E {str(expert_arch).replace(" ", "")}'
        pattern = 'Sparse'

    elif model_type == 'mlp':
        hid_lay = configs[m_key]['hidden_layer']
        arch = str([dimension] + hid_lay + [1]).replace(" ", "")
        pattern = 'Dense'
    
    elif model_type == 'bvh':
        pattern = 'Sparse'
        arch = f"Max-Depth: {configs[model_key]['max_depth']}"
        epochs = "None"
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Log results online 
    gsheet_log_row(
        worksheet_name   = worksheet_name,
        dimension        = dimension,
        data_name        = data_name,
        accelerator      = accelerator,
        chart_key        = m_key.split("_")[0]  + "+" if pattern == 'Sparse' else m_key.split("_")[0] ,
        model_type       = model_type,
        pattern          = pattern,
        size             = model_key[-1].capitalize(),
        architecture     = arch,
        total_p          = float(reg.get(m_key + core_keys['total_parameters_key'])),
        active_p         = float(reg.get(m_key + core_keys['active_parameters_key'])),
        epochs           = epochs,
        batch_speeds     = reg.get(m_key + core_keys['inf_speed_key']),
        timing_setup     = f"{configs['general']['inf_bench_query_size']/1e06}M queries | {configs['general']['inf_bench_repitions']/1e03}K reps",
        fp               = float(reg.get(m_key + core_keys['fp_key'])),
        fn               = float(reg.get(m_key + core_keys['fn_key'])),
        train_time       = round(float(reg.get(model_key + core_keys['training_time'])), 2)
    )

    # Second result logging to add sparse MoE results immediatly after the dense results
    if model_type == 'moe':
        m_key = f'{model_key}_sparse'
        gsheet_log_row(
            worksheet_name   = worksheet_name,
            dimension        = dimension,
            data_name        = data_name,
            accelerator      = accelerator,
            chart_key        = m_key.split("_")[0] + "+",
            model_type       = model_type,
            pattern          = 'Sparse',
            size             = model_key[-1].capitalize(),
            architecture     = arch,
            total_p          = float(reg.get(m_key + core_keys['total_parameters_key'])),
            active_p         = float(reg.get(m_key + core_keys['active_parameters_key'])),
            epochs           = epochs,
            batch_speeds     = reg.get(m_key + core_keys['inf_speed_key']),
            timing_setup     = f"{configs['general']['inf_bench_query_size']/1e06}M queries | {configs['general']['inf_bench_repitions']/1e03}K reps",
            fp               = float(reg.get(m_key + core_keys['fp_key'])),
            fn               = float(reg.get(m_key + core_keys['fn_key'])),
            train_time       = round(float(reg.get(model_key + core_keys['training_time'])), 2)
        )